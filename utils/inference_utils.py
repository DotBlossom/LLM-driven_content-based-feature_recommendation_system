import torch
import os
from tqdm import tqdm
from sqlalchemy import select
from torch.utils.data import DataLoader, Dataset

from database import ProductInferenceInput, TrainingItem
from utils.dependencies import get_global_batch_size, get_global_encoder

# =========================================================
# 1. 공통 전처리 함수 (학습/추론 양쪽에서 import하여 사용)
# =========================================================
def parse_db_row(row) -> TrainingItem:
    """
    DB Row(Dictionary)를 받아 학습/추론에 사용할 TrainingItem 객체로 변환.
    이 함수 하나로 Feature Flattening과 Name Tagging을 통합 관리합니다.
    """
    # 1. Feature Data 복사
    raw_feats = dict(row['feature_data'])
    
    # 2. Reinforced Feature Flattening
    if 'reinforced_feature' in raw_feats:
        re_dict = raw_feats['reinforced_feature']
        if isinstance(re_dict, dict):
            for key, val in re_dict.items():
                # Key 포맷팅: "MAT" -> "[MAT]"
                vocab_key = key if key.startswith("[") and key.endswith("]") else f"[{key}]"
                raw_feats[vocab_key] = val

    # 3. Name Tagging Logic
    base_name = row['product_name']
    product_type = raw_feats.get('product_type_name', "").strip()
    
    final_name = ""
    if base_name:
        if product_type:
            final_name = f"{base_name} (Category: {product_type})"
        else:
            final_name = base_name
    else:
        # Fallback: 타입명 + 외형
        appearance = raw_feats.get('graphical_appearance_name', "").strip()
        final_name = f"{product_type} {appearance}".strip()
        if not final_name:
            final_name = "Unknown Product"

    return TrainingItem(
        product_id=str(row['product_id']), 
        feature_data=raw_feats, 
        product_name=final_name
    )

# =========================================================
# 2. Inference Dataset & Utils
# =========================================================
class InferenceDataset(Dataset):
    def __init__(self, products):
        self.products = products
        
    def __len__(self):
        return len(self.products)

    def __getitem__(self, idx):
        return self.products[idx]

import os
import torch
from torch.utils.data import DataLoader, Dataset
from sqlalchemy import select
from tqdm import tqdm

# ... (parse_db_row, InferenceDataset 등의 위쪽 코드는 그대로 유지) ...

def generate_and_save_item_vectors(
    db_session, 
    save_dir="models", 
    safe_mode=False, 
    checkpoint_path: str = None  # 👈 [New] 체크포인트 경로 인자 추가
):
    """
    checkpoint_path: 특정 .pth 파일을 지정하면 그 가중치를 로드하여 임베딩을 생성합니다.
                     None이면 현재 get_global_encoder()에 로드된 상태 그대로 사용합니다.
    """
    save_tensor_path = os.path.join(save_dir, "pretrained_item_matrix.pt")
    save_ids_path = os.path.join(save_dir, "item_ids.pt") 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Target Device: {device}")

    # 1. 모델 아키텍처 가져오기
    try:
        # 껍데기(아키텍처)를 가져옵니다. 
        # (만약 get_global_encoder가 싱글톤이라도, load_state_dict로 가중치를 바꾸면 영향이 갈 수 있으니
        #  안전하게 하려면 새로 생성하는 것이 좋으나, 여기서는 편의상 가져와서 덮어씌웁니다.)
        model = get_global_encoder()
        
        # 🌟 [핵심] 지정된 체크포인트가 있으면 가중치 로드
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                print(f"♻️ Loading weights from Checkpoint: {checkpoint_path}")
                # map_location으로 디바이스 호환성 확보
                state_dict = torch.load(checkpoint_path, map_location=device)
                
                # 모델에 가중치 덮어씌우기 (strict=False는 혹시 모를 미세한 키 불일치 무시용, 보통은 True 권장)
                model.load_state_dict(state_dict, strict=True)
                print("✅ Successfully loaded checkpoint weights!")
            else:
                print(f"❌ [Error] Checkpoint path not found: {checkpoint_path}")
                return None, None
        else:
            print("⚠️ No checkpoint_path provided. Using current model weights.")

        model = model.to(device)
        model.eval() # 추론 모드 필수

    except Exception as e:
        print(f"❌ Model Setup Failed: {e}")
        return None, None

    # 2. DB 데이터 로드
    print("🚀 Fetching ALL products from DB for Inference...")
    stmt = select(
        ProductInferenceInput.product_id, 
        ProductInferenceInput.feature_data, 
        ProductInferenceInput.product_name 
    )
    result = db_session.execute(stmt).mappings().all()
    
    if not result:
        print("❌ No products found in DB.")
        return None, None

    inference_items = [parse_db_row(row) for row in result]
    # ID 순으로 정렬 (나중에 찾기 쉽게)
    inference_items.sort(key=lambda x: x.product_id)
    ordered_ids = [item.product_id for item in inference_items]
    
    print(f"✅ Prepared {len(inference_items)} items for vector extraction.")

    # 3. DataLoader 설정
    dataset = InferenceDataset(inference_items)
    
    # Collator 가져오기 (전역 함수 혹은 클래스 인스턴스)
    from item_tower import SimCSECollator
    collator_instance = SimCSECollator()

    def inference_collate_fn(batch):
        # is_first_view=True 옵션으로 1개의 뷰만 생성
        return collator_instance.process_batch_items(batch, is_first_view=True)

    batch_size = get_global_batch_size()
    dataloader = DataLoader(
        dataset, 
        # 안전 모드면 배치 사이즈를 줄임
        batch_size=batch_size * 4 if not safe_mode else batch_size, 
        shuffle=False, 
        collate_fn=inference_collate_fn,
        num_workers=0
    )

    # 4. Inference (Full GPU or Safe CPU)
    all_vectors = []
    
    print(f"⚡ Starting Vector Extraction (Mode: {'Safe/CPU-bound' if safe_mode else 'Fast/GPU-bound'})...")
    
    try:
        with torch.no_grad():
            for batch_inputs in tqdm(dataloader):
                # batch_inputs는 리스트 형태 [input_ids, masks, ...]
                inputs = [t.to(device) for t in batch_inputs]
                
                # Forward
                vectors = model(*inputs) # (Batch, 128)
                
                if safe_mode:
                    # [안전 모드] 즉시 CPU로 내림 (VRAM 절약)
                    all_vectors.append(vectors.cpu())
                else:
                    # [고속 모드] GPU에 둠
                    all_vectors.append(vectors)
        
        # 5. Merge & Save
        print("🧩 Merging tensors...")
        if safe_mode:
            final_tensor = torch.cat(all_vectors, dim=0)
        else:
            final_tensor = torch.cat(all_vectors, dim=0).cpu() # 마지막에 한 번에 내림

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("🚨 [OOM Error] GPU Memory Full! Retry with 'safe_mode=True'.")
            torch.cuda.empty_cache()
            return None, None
        raise e

    # 6. 파일 저장
    os.makedirs(save_dir, exist_ok=True)
    
    # 텐서 저장
    torch.save(final_tensor, save_tensor_path)
    # ID 리스트 저장
    torch.save(ordered_ids, save_ids_path)

    print(f"💾 Saved Vectors: {final_tensor.shape} -> {save_tensor_path}")
    print(f"💾 Saved IDs: {len(ordered_ids)} -> {save_ids_path}")
    
    return final_tensor, ordered_ids