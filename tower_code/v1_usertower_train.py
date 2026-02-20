            
import os
import torch
import torch.nn as nn
import numpy as np
import random
from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm

from v1_refine_usertower import FeatureProcessor, SASRecDataset, SASRecUserTower, dataset_peek, duorec_loss_refined, inbatch_corrected_logq_loss


# =====================================================================
# [Config] 파이프라인 설정 
# =====================================================================
@dataclass
class PipelineConfig:
    # Paths
    base_dir: str = r"D:\trainDataset\localprops"
    model_dir: str = r"C:\Users\candyform\Desktop\inferenceCode\models"
    
    # Hyperparameters
    batch_size: int = 896
    lr: float = 5e-5
    weight_decay: float = 1e-4
    epochs: int = 5
    
    # Model Args (SASRecUserTower용)
    d_model: int = 128
    max_len: int = 50
    dropout: float = 0.3
    pretrained_dim: int = 128 # 사전학습 아이템 벡터 차원 
    nhead: int = 4
    num_layers: int = 2
    
    # Loss Penalties
    lambda_logq: float = 0.1
    lambda_sup: float = 0.1
    lambda_cl: float = 0.1

    # 자동 할당될 메타데이터 크기
    num_items: int = 0
    num_prod_types: int = 0
    num_colors: int = 0
    num_graphics: int = 0
    num_sections: int = 0
    num_age_groups: int = 10

# =====================================================================
# Phase 1: Environment Setup
# =====================================================================
def setup_environment(seed: int = 42):
    """난수 고정 및 디바이스 설정 (Airflow Task 독립성 보장)"""
    print("\n⚙️ [Phase 1] Setting up environment...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device set to: {device}")
    return device

# =====================================================================
# Phase 2: Data Preparation
# =====================================================================
def prepare_features(cfg: PipelineConfig):
    """FeatureProcessor 초기화 및 메타데이터 업데이트"""
    print("\n📊 [Phase 2] Loading Processors...")
    
    # 경로 설정
    user_path = os.path.join(cfg.base_dir, "features_user_w_meta.parquet") 
    item_path = os.path.join(cfg.base_dir, "features_item.parquet")
    seq_path = os.path.join(cfg.base_dir, "features_sequence_cleaned.parquet")
    
    
    # 평가할때, 이거로 val_proc 만들어야함.그라고 val용 user파일도 다시 만들어야함
    TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")
    USER_VAL_FEAT_PATH = os.path.join(cfg.base_dir, "features_user_val.parquet")
    SEQ_VAL_DATA_PATH = os.path.join(cfg.base_dir, "features_sequence_val.parquet")
    # val 경로 만들기 -> processor 초기화를 val_proc으로 따로 하고, return에 추가함.
    # Processor 초기화 
    processor = FeatureProcessor(user_path, item_path, seq_path)
    
    # Config에 임베딩 레이어 생성을 위한 메타데이터 업데이트
    cfg.num_items = processor.num_items
    
    ####### 실제 item metadata id랑 묶인상태로 가져와야하고 연결 필요 #######

    cfg.num_prod_types = int(processor.items['type_id'].max()) if 'type_id' in processor.items else 50
    cfg.num_colors = int(processor.items['color_id'].max()) if 'color_id' in processor.items else 50
    cfg.num_graphics = int(processor.items['graphic_id'].max()) if 'graphic_id' in processor.items else 50
    cfg.num_sections = int(processor.items['section_id'].max()) if 'section_id' in processor.items else 50

    print(f"✅ Features Loaded. Total Items: {cfg.num_items}")
    return processor, cfg

# =====================================================================
# Phase 3: Embedding Alignment & DataLoader
# =====================================================================
def load_aligned_pretrained_embeddings(processor, model_dir, pretrained_dim):
    """Dataset에서 사용할 수 있도록 정렬된 사전학습 벡터(N+1, Dim) 생성"""
    print(f"\n🔄 [Phase 3-1] Aligning Pretrained Item Embeddings...")
    emb_path = os.path.join(model_dir, "pretrained_item_matrix.pt")
    ids_path = os.path.join(model_dir, "item_ids.pt")

    num_embeddings = processor.num_items + 1 
    aligned_weight = torch.randn(num_embeddings, pretrained_dim) * 0.01 
    aligned_weight[0] = 0.0 # Padding
    
    try:
        pretrained_emb = torch.load(emb_path, map_location='cpu')
        if isinstance(pretrained_emb, dict):
            pretrained_emb = pretrained_emb.get('weight', pretrained_emb.get('item_content_emb.weight'))
        pretrained_ids = torch.load(ids_path, map_location='cpu')
        
        pretrained_map = {str(iid.item()) if isinstance(iid, torch.Tensor) else str(iid): pretrained_emb[idx] 
                          for idx, iid in enumerate(pretrained_ids)}
        
        matched = 0
        for i, current_id_str in enumerate(processor.item_ids):
            if current_id_str in pretrained_map:
                aligned_weight[i + 1] = pretrained_map[current_id_str]
                matched += 1
                
        print(f"✅ Matched: {matched}/{len(processor.item_ids)}")
    except Exception as e:
        print(f"⚠️ [Warning] Failed to load Pretrained files: {e}. Using random init.")
        
    return aligned_weight

def create_dataloaders(processor, cfg: PipelineConfig, aligned_pretrained_vecs=None):
    """Dataset 및 DataLoader 인스턴스화"""
    print("\n📦 [Phase 3-2] Creating DataLoaders...")
    
    # SASRecDataset 내부에서 aligned_pretrained_vecs를 참조하게끔 
    
    train_dataset = SASRecDataset(processor, max_len=cfg.max_len, is_train=True)
    
    # Dataset 인스턴스에 정렬된 pretrained vector 룩업 테이블 주입 (동적 바인딩)
    train_dataset.pretrained_lookup = aligned_pretrained_vecs 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True,
        drop_last=True
    )
    
    print(f"✅ Train Loader Ready: {len(train_loader)} batches/epoch")
    return train_loader


import hashlib
import json

def get_hash_id(text, hash_size):
    """문자열을 일관된 정수 ID(1 ~ hash_size)로 해싱 (0은 Padding)"""
    if not text or str(text).lower() in ['unknown', 'nan', 'none']:
        return 0
    # MD5를 사용하여 파이썬 세션이 바뀌어도 항상 동일한 해시값 보장
    hash_obj = hashlib.md5(str(text).strip().lower().encode('utf-8'))
    # 16진수를 정수로 변환 후 hash_size로 나눈 나머지 + 1
    return (int(hash_obj.hexdigest(), 16) % hash_size) + 1

def load_item_metadata_hashed(processor, base_dir, hash_size=1000):
    """JSON 파일을 읽어 정렬된 메타데이터 해시 텐서(N+1, 4)를 생성"""
    print("\n🏷️ [Phase 3-2] Loading and Hashing Item Metadata...")
    json_path = os.path.join(base_dir, "filtered_data_reinforced.json")
    
    num_items = processor.num_items + 1
    # 0번 인덱스는 패딩을 위해 0으로 유지 (N+1, 4차원 배열)
    item_side_arr = np.zeros((num_items, 4), dtype=np.int64)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            item_data = json.load(f)
    except Exception as e:
        print(f"❌ [Error] Failed to load JSON: {e}")
        return torch.tensor(item_side_arr, dtype=torch.long)
    
    # 빠른 검색을 위해 O(1) Lookup Dictionary 생성
    # int형 article_id를 string으로 변환하여 매핑
    metadata_dict = {str(item.get('article_id', '')): item for item in item_data}
    
    matched = 0
    for i, current_id_str in enumerate(processor.item_ids):
        idx = i + 1 # 1-based indexing
        
        if current_id_str in metadata_dict:
            meta = metadata_dict[current_id_str]
            
            # 카테고리 매핑 및 해싱 (해당 키가 없으면 빈 문자열 반환)
            type_val = meta.get("product_type_name", "")
            color_val = meta.get("colour_group_name", "")
            graphic_val = meta.get("graphical_appearance_name", "")
            section_val = meta.get("section_name", "")
            
            item_side_arr[idx, 0] = get_hash_id(type_val, hash_size)
            item_side_arr[idx, 1] = get_hash_id(color_val, hash_size)
            item_side_arr[idx, 2] = get_hash_id(graphic_val, hash_size)
            item_side_arr[idx, 3] = get_hash_id(section_val, hash_size)
            
            matched += 1

    print(f"✅ Metadata Matched & Hashed: {matched}/{len(processor.item_ids)} (Hash Size: {hash_size})")
    
    return torch.tensor(item_side_arr, dtype=torch.long)
# =====================================================================
# Phase 4: Model Setup
# =====================================================================
class DummyItemTower(nn.Module):
    """실행 테스트용 더미 아이템 타워"""
    def __init__(self, num_items, dim):
        super().__init__()
        self.emb = nn.Embedding(num_items + 1, dim)
        self.log_q = nn.Parameter(torch.zeros(num_items + 1), requires_grad=False)
    def get_all_embeddings(self): return self.emb.weight
    def get_log_q(self): return self.log_q

def setup_models(cfg: PipelineConfig, device):
    """User Tower 초기"""
    print("\n🧠 [Phase 4] Initializing Models...")
    
    user_tower = SASRecUserTower(cfg).to(device)
    


    print("✅ Models initialized and moved to device.")
    return user_tower

# =====================================================================
# Phase 5: Training Loop (1 Epoch Runner)
# =====================================================================
def train_one_epoch(epoch, model, full_item_embeddings, log_q_tensor, dataloader, optimizer, scaler, cfg, device):
    """단일 에포크 훈련 함수 (실제 Loss 계산 및 로그 모니터링 적용)"""
    model.train()
    total_loss_accum = 0.0
    main_loss_accum = 0.0
    cl_loss_accum = 0.0
    
        
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        # -------------------------------------------------------
        # 1. Data Unpacking (Dictionary to Device)
        # -------------------------------------------------------
        item_ids = batch['item_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        time_bucket_ids = batch['time_bucket_ids'].to(device)
        
        type_ids = batch['type_ids'].to(device)
        color_ids = batch['color_ids'].to(device)
        graphic_ids = batch['graphic_ids'].to(device)
        section_ids = batch['section_ids'].to(device)
        
        age_bucket = batch['age_bucket'].to(device)
        price_bucket = batch['price_bucket'].to(device)
        cnt_bucket = batch['cnt_bucket'].to(device)
        recency_bucket = batch['recency_bucket'].to(device)
        
        channel_ids = batch['channel_ids'].to(device)
        club_status_ids = batch['club_status_ids'].to(device)
        news_freq_ids = batch['news_freq_ids'].to(device)
        fn_ids = batch['fn_ids'].to(device)
        active_ids = batch['active_ids'].to(device)
        
        cont_feats = batch['cont_feats'].to(device)
        
        # Pretrained Vector 룩업 처리
        if 'pretrained_vecs' in batch:
            pretrained_vecs = batch['pretrained_vecs'].to(device)
        else:
            pretrained_vecs = dataloader.dataset.pretrained_lookup[item_ids.cpu()].to(device)
            
        forward_kwargs = {
            'pretrained_vecs': pretrained_vecs,
            'item_ids': item_ids,
            'time_bucket_ids': time_bucket_ids,
            'type_ids': type_ids,
            'color_ids': color_ids,
            'graphic_ids': graphic_ids,
            'section_ids': section_ids,
            'age_bucket': age_bucket,
            'price_bucket': price_bucket,
            'cnt_bucket': cnt_bucket,
            'recency_bucket': recency_bucket,
            'channel_ids': channel_ids,
            'club_status_ids': club_status_ids,
            'news_freq_ids': news_freq_ids,
            'fn_ids': fn_ids,
            'active_ids': active_ids,
            'cont_feats': cont_feats,
            'padding_mask': padding_mask,
            'training_mode': True
        }

        # =======================================================
        # [모니터링 로그] 첫 배치에서만 데이터 상태 점검
        # =======================================================
        if batch_idx == 0:
            print(f"\n📦 [Batch 0 Monitor]")
            print(f"   - Item IDs: Shape {item_ids.shape} | Min {item_ids.min()} | Max {item_ids.max()}")
            print(f"   - Time Buckets: Min {time_bucket_ids.min()} | Max {time_bucket_ids.max()}")
            pad_ratio = (padding_mask.sum().item() / padding_mask.numel()) * 100
            print(f"   - Padding Ratio: {pad_ratio:.1f}%")
            print(f"   - Cont Feats Mean: {cont_feats.mean().item():.3f} | Std: {cont_feats.std().item():.3f}")
            
            print("\n🎯 [First User Data State Check]")
            print("-" * 50)
            print(f"👤 [User Profile]")
            print(f"   - Age Bucket ID:    {age_bucket[0].item()} (Target Age Group)")
            print(f"   - Price Bucket ID:  {price_bucket[0].item()} (Spending Power)")
            print(f"   - News Freq ID:     {news_freq_ids[0].item()} (Marketing Sensitivity)")
            
            valid_indices = torch.where(~padding_mask[0])[0]
            if len(valid_indices) > 0:
                print(f"\n🛍️ [Item History - Last 3 Items]")
                sample_indices = valid_indices[-3:] 
                sample_types = type_ids[0][sample_indices].tolist()
                sample_times = time_bucket_ids[0][sample_indices].tolist()
                for i, (t_id, time_id) in enumerate(zip(sample_types, sample_times)):
                    print(f"   - Item {i+1}: Type Hash ID [{t_id}] | Time Bucket ID [{time_id}]")
            else:
                print("\n⚠️ [Warning] This user has NO valid sequence (All Padded).")
            print("-" * 50)

        # -------------------------------------------------------
        # 2. Forward & Real Loss Calculation (AMP)
        # -------------------------------------------------------
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            # A. First View
            output_1 = model(**forward_kwargs)
            # B. Second View (Dropout 마스크가 달라짐)
            output_2 = model(**forward_kwargs)

            # (1) Main Loss (All Time Steps)
            valid_mask = ~padding_mask.view(-1)
            flat_output = output_1.view(-1, cfg.d_model)[valid_mask]
            flat_targets = target_ids.view(-1)[valid_mask]
            
            

            main_loss = inbatch_corrected_logq_loss(
                user_emb=flat_output,
                item_tower_emb=full_item_embeddings,
                target_ids=flat_targets,
                log_q_tensor=log_q_tensor,
                lambda_logq=cfg.lambda_logq
            )
            
            # (2) DuoRec Loss (Last Time Step Only)
            last_output_1 = output_1[:, -1, :] 
            last_output_2 = output_2[:, -1, :]
            last_targets = target_ids[:, -1]

            cl_loss = duorec_loss_refined(
                user_emb_1=last_output_1,
                user_emb_2=last_output_2,
                target_ids=last_targets,
                lambda_sup=cfg.lambda_sup
            )

            # 최종 Loss 조합
            total_loss = main_loss + (cfg.lambda_cl * cl_loss)

        # -------------------------------------------------------
        # 3. Backward & Optimizer Step
        # -------------------------------------------------------
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        # 기울기 폭발 방지를 위한 정규화 (5.0은 트랜스포머에서 많이 쓰이는 여유있는 값)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        # 누적
        total_loss_accum += total_loss.item()
        main_loss_accum += main_loss.item()
        cl_loss_accum += cl_loss.item()
        
        pbar.set_postfix({
            'Loss': f"{total_loss.item():.4f}",
            'Main': f"{main_loss.item():.4f}",
            'CL': f"{cl_loss.item():.4f}"
        })
        
        # 100배치마다 로깅
        if batch_idx % 100 == 0:
            print(f"   [Epoch {epoch}] Batch {batch_idx:04d}/{len(dataloader)} | Total Loss: {total_loss.item():.4f} (Main: {main_loss.item():.4f}, CL: {cl_loss.item():.4f})")

    avg_loss = total_loss_accum / len(dataloader)
    avg_main = main_loss_accum / len(dataloader)
    avg_cl = cl_loss_accum / len(dataloader)
    
    print(f"🏁 Epoch {epoch} Completed | Avg Total: {avg_loss:.4f} (Main: {avg_main:.4f}, CL: {avg_cl:.4f})")
    return avg_loss
# =====================================================================
# Main Execution Pipeline
# =====================================================================
def run_pipeline():
    """Airflow DAG나 MLflow Run에서 직접 호출하는 엔트리 포인트"""
    print("🚀 Starting User Tower Training Pipeline...")
    
    
    
    
    
    # 1. Config & Env
    cfg = PipelineConfig()
    device = setup_environment()
    processor, cfg = prepare_features(cfg)
    # item metadata cfg
    HASH_SIZE = 1000
    cfg.num_prod_types = HASH_SIZE
    cfg.num_colors = HASH_SIZE
    cfg.num_graphics = HASH_SIZE
    cfg.num_sections = HASH_SIZE
    
    # 2. Data

    aligned_vecs = load_aligned_pretrained_embeddings(processor, cfg.model_dir, cfg.pretrained_dim)
    
    full_item_embeddings = aligned_vecs.to(device)
    log_q_tensor = processor.get_logq_probs(device)
    
    
    item_metadata_tensor = load_item_metadata_hashed(processor, cfg.base_dir, hash_size=HASH_SIZE)
    processor.i_side_arr = item_metadata_tensor.numpy()
    train_loader = create_dataloaders(processor, cfg, aligned_vecs)
    dataset_peek(train_loader.dataset, processor)
    
    # 3. Models & Optimizer
    user_tower = setup_models(cfg, device)
    optimizer = torch.optim.AdamW(user_tower.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # 4. Training Loop (Phase 5)
    # mlflow.start_run() 블록용
    for epoch in range(1, cfg.epochs + 1):
        avg_loss = train_one_epoch(
            epoch=epoch,
            model=user_tower,
            full_item_embeddings=full_item_embeddings,
            log_q_tensor=log_q_tensor,

            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device
        )
        # mlflow.log_metric("train_loss", avg_loss, step=epoch)
        
    print("🎉 Pipeline Execution Finished Successfully!")

if __name__ == "__main__":
    run_pipeline()