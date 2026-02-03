# inference_utils.py (또는 적절한 위치)

from typing import Any, Dict
from pydantic import BaseModel
from torch.utils.data import Dataset, DataLoader
from database import ProductInferenceInput # DB 스키마
from main import get_global_encoder, get_global_batch_size # 의존성 주입 함수 임포트
from utils import vocab # vocab 관련 유틸
import torch
import os
from tqdm import tqdm
from sqlalchemy import select

# 1. 추론용 데이터셋 (No Dropout, No Corruption)
class InferenceDataset(Dataset):
    def __init__(self, products):
        self.products = products # List[TrainingItem]
        
    def __len__(self):
        return len(self.products)

    def __getitem__(self, idx):
        # 학습 때와 달리 데이터를 복제하거나 변형하지 않고 그대로 반환
        return self.products[idx]

class TrainingItem(BaseModel):
    product_id: str
    feature_data: Dict[str, Any] # DB에서 긁어온 Raw JSON
    product_name: str            # Text Embedding용

# 필요에 맞는 범주(train, valid, test, real)에 맞게 Item set 관리. 후 case 맞게 loading
# 2. 벡터 추출 함수
def generate_and_save_item_vectors(db_session, save_path="models/pretrained_item_matrix.pt"):
    """
    전역 메모리에 로드된 Item Tower를 사용하여 전체 아이템 벡터를 생성하고 저장합니다.
    """
    
    # ---------------------------------------------------------
    # A. 전역 모델 및 설정 가져오기 (Dependency Injection)
    # ---------------------------------------------------------
    try:
        model = get_global_encoder()     # 이미 로드된 HybridItemTower 인스턴스
        batch_size = get_global_batch_size()
        device = next(model.parameters()).device # 모델이 있는 장치(cuda/cpu) 확인
        print(f"✅ Loaded Global Encoder on {device}")
    except Exception as e:
        print(f"❌ Global Model Load Failed: {e}")
        return

    # ---------------------------------------------------------
    # B. DB에서 전체 아이템 로드
    # Query의 대상이 되는 Loader를 미리 설정 
    # ---------------------------------------------------------
    print("🚀 Fetching ALL products from DB for Inference...")
    stmt = select(
        ProductInferenceInput.product_id, 
        ProductInferenceInput.feature_data, 
        ProductInferenceInput.product_name 
    )
    result = db_session.execute(stmt).mappings().all()
    
    if not result:
        print("❌ No items found in DB.")
        return

    # 데이터 변환 (TrainingItem 객체로)
    # (학습 코드의 전처리 로직과 동일하게 유지 - RE Flattening, Name Tagging 등)
    inference_items = []
    for row in result:
        raw_feats = dict(row['feature_data'])
        
        # [중요] 학습 데이터 전처리 로직 복사 (함수로 분리하는게 좋음)
        if 'reinforced_feature' in raw_feats:
            re_dict = raw_feats['reinforced_feature']
            if isinstance(re_dict, dict):
                for key, val in re_dict.items():
                    k = key if key.startswith("[") else f"[{key}]"
                    raw_feats[k] = val
        
        # Name Tagging
        base_name = row['product_name'] or ""
        p_type = raw_feats.get('product_type_name', "").strip()
        final_name = f"{base_name} (Category: {p_type})" if base_name and p_type else base_name
        
        inference_items.append(TrainingItem(
            product_id=str(row['product_id']),
            feature_data=raw_feats,
            product_name=final_name
        ))

    print(f"✅ Prepared {len(inference_items)} items for inference.")

    # ---------------------------------------------------------
    # C. DataLoader 준비
    # ---------------------------------------------------------
    dataset = InferenceDataset(inference_items)
    
    # 기존 SimCSECollator 활용
    # 주의: SimCSECollator는 클래스이므로 인스턴스화 필요
    from item_tower import SimCSECollator # (Collator가 정의된 파일에서 임포트)
    collator_instance = SimCSECollator() 

    # 커스텀 Collate 함수: 단일 뷰(View)만 처리하도록 래핑
    def inference_collate_fn(batch):
        # batch는 List[TrainingItem]
        # SimCSECollator의 process_batch_items 메서드 직접 호출
        return collator_instance.process_batch_items(batch, is_first_view=True)

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size * 2, # 추론은 역전파가 없으므로 배치를 2배로 키워도 됨
        shuffle=False,             # 순서대로 뽑아야 ID 매핑이 쉬움
        collate_fn=inference_collate_fn,
        num_workers=0
    )

    # ---------------------------------------------------------
    # D. Inference Loop
    # ---------------------------------------------------------
    all_vectors = []
    all_product_ids = [] # 나중에 ID 매핑을 위해 저장
    
    model.eval() # 🚨 필수: Dropout 비활성화

    print("⚡ Starting Vector Extraction...")
    
    with torch.no_grad(): # 🚨 필수: Gradient 계산 끄기 (속도/메모리 최적화)
        for batch_inputs in tqdm(dataloader):
            # batch_inputs는 (std, re_ids, re_mask, txt_ids, txt_mask) 튜플
            inputs = [t.to(device) for t in batch_inputs]
            
            # 모델 통과 (HybridItemTower.forward)
            # 결과: (Batch, 128)
            vectors = model(*inputs)
            
            # CPU로 내려서 리스트에 저장
            all_vectors.append(vectors.cpu())
            
            # 현재 배치의 Product ID 추적 (필요 시)
            # (DataLoader의 batch 순서와 dataset 순서가 같으므로 별도 처리 없어도 됨)

    # ---------------------------------------------------------
    # E. 병합 및 저장
    # ---------------------------------------------------------
    # (Num_Items, 128)
    final_tensor = torch.cat(all_vectors, dim=0)
    
    # 폴더 확보
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(final_tensor, save_path)
    print(f"💾 Lookup Table Saved: {save_path}")
    print(f"   - Shape: {final_tensor.shape}")
    print(f"   - Device: {final_tensor.device} (Should be cpu)")
    
    return final_tensor