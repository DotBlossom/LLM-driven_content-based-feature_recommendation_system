            
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from dataclasses import dataclass
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

import wandb
from v1_refine_usertower import FeatureProcessor, SASRecDataset, SASRecUserTower, dataset_peek, duorec_loss_refined, full_batch_hard_emphasis_loss, inbatch_corrected_logq_loss, inbatch_hnm_corrected_loss_with_stats,inbatch_mixed_hnm_loss_with_stats


# =====================================================================
# [Config] 파이프라인 설정 
# =====================================================================
@dataclass
class PipelineConfig:
    # Paths
    base_dir: str = r"D:\trainDataset\localprops"
    model_dir: str = r"C:\Users\candyform\Desktop\inferenceCode\models"
    
    # Hyperparameters
    batch_size: int = 768
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 15
    
    # Model Args (SASRecUserTower용)
    d_model: int = 128
    max_len: int = 50
    dropout: float = 0.2
    pretrained_dim: int = 128 # 사전학습 아이템 벡터 차원 
    nhead: int = 4
    num_layers: int = 2
    
    # Loss Penalties
    lambda_logq: float = 1.0
    lambda_sup: float = 0.1
    lambda_cl: float = 0.2
   
    # [신규] HNM 제어 파라미터
    top_k_percent: float = 0.01 # 상위 15% 하드 네거티브 사용 (10~20% 사이 권장)
    hnm_threshold: float = 0.90
    hard_margin: float = 0.01

    # model 관리
    freeze_item_tower: bool = True
    item_tower_pth_name: str = "encoder_ep03_loss0.8129.pth"
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
    """FeatureProcessor 초기화 및 메타데이터 업데이트 (로컬 캐싱 적용)"""
    print("\n📊 [Phase 2] Loading Processors...")
    
    # 1. 캐시 파일 경로 설정
    cache_path = os.path.join(cfg.base_dir, "processor_cache.pkl")
    
    # 2. 캐시 존재 여부 확인 및 로드
    if os.path.exists(cache_path):
        print(f"   ✅ [Cache Hit] Found cached processors at {cache_path}")
        print("   ⏳ Loading from local storage...")
        with open(cache_path, 'rb') as f:
            train_proc, val_proc = pickle.load(f)
            
    # 3. 캐시가 없을 경우: 원본 파라켓 로드 및 생성
    else:
        print("   ⚠️ [Cache Miss] Cache not found. Processing from Parquet files...")
        
        # 경로 설정
        user_path = os.path.join(cfg.base_dir, "features_user_w_meta.parquet") 
        item_path = os.path.join(cfg.base_dir, "features_item.parquet")
        seq_path = os.path.join(cfg.base_dir, "features_sequence_cleaned.parquet")
        
        TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")
        USER_VAL_FEAT_PATH = os.path.join(cfg.base_dir, "features_user_w_meta_val.parquet")
        SEQ_VAL_DATA_PATH = os.path.join(cfg.base_dir, "features_sequence_val.parquet")
        
        # Processor 초기화 
        train_proc = FeatureProcessor(user_path, item_path, seq_path)
        val_proc = FeatureProcessor(USER_VAL_FEAT_PATH, item_path, SEQ_VAL_DATA_PATH, base_processor=train_proc)
        
        # [신규] 생성된 Processor 객체를 로컬 파일로 저장 (HIGHEST_PROTOCOL로 속도/용량 최적화)
        print("   💾 Saving processors to local cache for future use...")
        with open(cache_path, 'wb') as f:
            pickle.dump((train_proc, val_proc), f, protocol=pickle.HIGHEST_PROTOCOL)

    # 4. Config 업데이트 (캐시에서 불러왔든 새로 만들었든 동일하게 적용)
    cfg.num_items = train_proc.num_items
    
    ####### 실제 item metadata id랑 묶인상태로 가져와야하고 연결 필요 #######
    cfg.num_prod_types = int(train_proc.items['type_id'].max()) if 'type_id' in train_proc.items else 50
    cfg.num_colors = int(train_proc.items['color_id'].max()) if 'color_id' in train_proc.items else 50
    cfg.num_graphics = int(train_proc.items['graphic_id'].max()) if 'graphic_id' in train_proc.items else 50
    cfg.num_sections = int(train_proc.items['section_id'].max()) if 'section_id' in train_proc.items else 50

    print(f"✅ Features Loaded. Total Items: {cfg.num_items}")
    return train_proc, val_proc, cfg
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

def create_dataloaders(processor, cfg: PipelineConfig, aligned_pretrained_vecs=None, is_train=True):
    """Dataset 및 DataLoader 인스턴스화"""
    mode_str = "Train" if is_train else "Validation"
    print(f"\n📦 [Phase 3-2] Creating {mode_str} DataLoaders...")
    
    # 💡 1. is_train 파라미터 전달
    dataset = SASRecDataset(processor, max_len=cfg.max_len, is_train=is_train)
    
    # Dataset 인스턴스에 정렬된 pretrained vector 룩업 테이블 주입
    dataset.pretrained_lookup = aligned_pretrained_vecs 
    
    loader = DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        # 💡 2. 검증 시에는 셔플을 끄고, 자투리 데이터(마지막 배치)도 버리지 않고 모두 평가
        shuffle=is_train, 
        num_workers=0, 
        pin_memory=True,
        drop_last=is_train 
    )
    
    print(f"✅ {mode_str} Loader Ready: {len(loader)} batches/epoch")
    return loader

def load_item_tower_state_dict(model_dir: str, pth_filename: str, device):
    """
    [Data/IO] 물리적 파일(.pth)을 읽어 메모리(state_dict)로 올리는 순수 IO 역할.
    모델 구조나 학습 상태(Freeze 여부)에는 절대 관여하지 않음.
    """
    file_path = os.path.join(model_dir, pth_filename)
    
    if not os.path.exists(file_path):
        print(f"⚠️ [IO Warning] Item Tower file not found: {file_path}")
        print("   -> Random initialization will be used.")
        return None
        
    print(f"📥 [IO] Loading Item Tower weights from {pth_filename}...")
    
    try:
        # map_location을 통해 CPU/GPU 메모리 매핑 최적화
        state_dict = torch.load(file_path, map_location=device)
        return state_dict
    except Exception as e:
        print(f"❌ [IO Error] Failed to load .pth file: {e}")
        return None
    
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
class SASRecItemTower(nn.Module):
    def __init__(self, num_items, d_model, log_q_tensor=None):
        super().__init__()
        
        # 💡 단순히 임베딩이라기보다 '미세조정 가능한 아이템 벡터 행렬'임을 명시
        self.item_matrix = nn.Embedding(num_items + 1, d_model, padding_idx=0)
        
        if log_q_tensor is not None:
            self.register_buffer('log_q', log_q_tensor)
        else:
            self.register_buffer('log_q', torch.zeros(num_items + 1))

    def get_all_embeddings(self):
        return self.item_matrix.weight

    def get_log_q(self):
        return self.log_q
        
    def set_freeze_state(self, freeze: bool):
        for param in self.parameters():
            param.requires_grad = not freeze
            
    # 💡 [핵심] 밖에서 억지로 쑤셔넣지 않고, 클래스 스스로 추론 벡터를 받아 초기화하는 메서드
    def init_from_pretrained(self, pretrained_vecs):
        """추론된 사전학습 벡터를 미세조정 가능한 파라미터(Weight)로 초기화"""
        with torch.no_grad():
            self.item_matrix.weight.copy_(pretrained_vecs)
        print("✅ Pretrained item vectors successfully loaded into learnable matrix!")
    
def setup_models(cfg: PipelineConfig, device, item_state_dict=None, log_q_tensor=None):
    print(f"\n🧠 [Phase 4] Initializing Models...")
    
    # 1. User Tower 생성
    user_tower = SASRecUserTower(cfg).to(device)
    
    # 2. Item Tower 뼈대 생성
    item_tower = SASRecItemTower(
        num_items=cfg.num_items, 
        d_model=cfg.d_model, 
        log_q_tensor=log_q_tensor
    ).to(device)
    
    # 3. Data 주입 (IO 데이터 -> Architecture)
    if item_state_dict is not None:
        try:
            # strict=False 옵션: 저장된 모델과 현재 구조의 키 이름이 조금 달라도 유연하게 로드
            missing, unexpected = item_tower.load_state_dict(item_state_dict, strict=False)
            print(f"✅ Item Tower weights successfully loaded!")
            if unexpected:
                print(f"   - Ignored extra keys from .pth: {unexpected[:3]}...")
            if missing:
                print(f"   ⚠️ [CRITICAL WARNING] Missing keys: {missing}")
        except Exception as e:
            print(f"❌ [Error] Weight injection failed: {e}")

    # 4. 학습 상태(Freeze/Unfreeze) 통제 적용
    item_tower.set_freeze_state(cfg.freeze_item_tower)
    
    # 직관적인 로깅
    mode_str = "FROZEN ❄️ (Speed Optimized)" if cfg.freeze_item_tower else "UNFROZEN 🔥 (Joint Fine-tuning)"
    print(f"✅ Item Tower State: {mode_str}")
    
    return user_tower, item_tower

# =====================================================================
# Phase 5: Training Loop 
# =====================================================================
def train_user_tower(epoch, model, item_tower, log_q_tensor, dataloader, optimizer, scaler, cfg, device, seq_labels, static_labels):
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
            #valid_mask = ~padding_mask.view(-1)
            #flat_output = output_1.view(-1, cfg.d_model)[valid_mask]
            #flat_targets = target_ids.view(-1)[valid_mask]
            
            last_output_1 = output_1[:, -1, :] # (Batch, Dim)
            last_targets = target_ids[:, -1]   # (Batch,)
            last_valid_mask = ~padding_mask[:, -1]
            
            valid_user_emb = last_output_1[last_valid_mask]
            valid_targets = last_targets[last_valid_mask]
            
            hnm_stats = {}
            
            if valid_user_emb.size(0) > 0:
                valid_user_emb = F.normalize(valid_user_emb, p=2, dim=1)
                
                # 💡 [핵심 추가] 평가때와 동일하게 item_tower에서 실시간으로 벡터 추출!
                # 나중에 Joint Training을 켤 때 아이템 벡터가 업데이트되려면 여기서 뽑아야 합니다.
                full_item_embeddings = item_tower.get_all_embeddings()
                norm_item_embeddings = F.normalize(full_item_embeddings, p=2, dim=1)
                main_loss, hnm_stats = full_batch_hard_emphasis_loss(
                    user_emb=valid_user_emb,
                    item_tower_emb=norm_item_embeddings, 
                    target_ids=valid_targets,
                    log_q_tensor=log_q_tensor,
                    top_k_percent=cfg.top_k_percent,
                    hard_margin=cfg.hard_margin,
                    hnm_threshold=cfg.hnm_threshold,   # Config에서 가져온 Threshold (예: 0.85)
                    temperature=0.15, 
                    lambda_logq=cfg.lambda_logq        # 상향된 1.0 적용
                )
            else:
                main_loss = torch.tensor(0.0, device=device)
                hnm_stats = {"avg_hn_similarity": 0.0, "num_active_hard_negs": 0}
            
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
        if batch_idx % 100 == 0:
            wandb.log({
                "Train/Main_Loss_Step": main_loss.item(),
                "HNM/Avg_Hard_Negative_Sim": hnm_stats.get("avg_hn_similarity", 0),
                "HNM/Num_K": hnm_stats.get("num_active_hard_negs", 0),
                "Step": epoch * len(dataloader) + batch_idx
            })
        # -------
        
    avg_loss = total_loss_accum / len(dataloader)
    avg_main = main_loss_accum / len(dataloader)
    avg_cl = cl_loss_accum / len(dataloader)

    with torch.no_grad():
        s_weights = torch.sigmoid(model.seq_gate).cpu().numpy()
        u_weights = torch.sigmoid(model.static_gate).cpu().numpy()
            
            # 딕셔너리 형태로 변환하여 WandB에 전송
    gate_log = {f"Gate/Seq_{label}": w for label, w in zip(seq_labels, s_weights)}
    gate_log.update({f"Gate/Static_{label}": w for label, w in zip(static_labels, u_weights)})
    wandb.log(gate_log)

    
    print(f"🏁 Epoch {epoch} Completed | Avg Total: {avg_loss:.4f} (Main: {avg_main:.4f}, CL: {avg_cl:.4f})")
    return avg_loss


import torch
import torch
import torch.nn.functional as F
from tqdm import tqdm



# 💡 인자에 processor를 추가했습니다.
def evaluate_model(model, item_tower, dataloader, target_df_path, device, processor, k_list=[20, 100, 500]):
    """
    Validation 데이터셋과 정답지(target_dict)를 이용해 Recall@K를 평가하는 함수
    """
    model.eval()
    item_tower.eval()
    print(f"🎯 Loading targets from: {target_df_path}")
    target_df = pd.read_parquet(target_df_path)
    target_dict = target_df.set_index('customer_id')['target_ids'].to_dict()
    
    # K값 중 가장 큰 값을 기준으로 한 번만 Top-K 연산을 수행하여 GPU 연산 절약
    max_k = max(k_list)
    
    total_hits = {k: 0.0 for k in k_list}
    total_valid_users = 0
    
    with torch.no_grad():
        # 1. 전체 아이템 임베딩 로드 및 정규화 (루프 밖에서 한 번만 수행)
        full_item_embeddings = item_tower.get_all_embeddings()
        norm_item_embeddings = F.normalize(full_item_embeddings, p=2, dim=1)
        
        '''
        print("\n🔍 [Eval Monitor] Item Tower Check")
        print(f"   - Shape: {full_item_embeddings.shape}")
        print(f"   - Mean: {full_item_embeddings.mean().item():.6f} | Std: {full_item_embeddings.std().item():.6f}")
            # 인덱스 1번(첫 번째 실제 아이템)의 앞 5개 차원 값 출력
        if full_item_embeddings.size(0) > 1:
            print(f"   - Item [1] Sample: {full_item_embeddings[1][:5].tolist()}")
        '''
        
        
        
        
        # tqdm을 이용해 진행 시간 및 상태 표시
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Dataloader에서 'user_ids'를 문자열 리스트로 바로 가져옴
            user_ids = batch['user_ids'] 
            
            item_ids = batch['item_ids'].to(device)
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
                print("pretrained_vecs has been loaded")
            else:
                pretrained_vecs = dataloader.dataset.pretrained_lookup[item_ids.cpu()].to(device)
            
            # =======================================================
            '''
            if batch_idx == 0:
                print(f"\n🔍 [Eval Monitor] Pretrained Vecs Check (Batch 0)")
                print(f"   - Shape: {pretrained_vecs.shape}")
                print(f"   - Mean: {pretrained_vecs.mean().item():.6f} | Std: {pretrained_vecs.std().item():.6f}")
                
                # 패딩(0)이 아닌 실제 아이템 ID 하나를 찾아 해당 벡터의 값 확인
                valid_mask = item_ids[0] != 0
                if valid_mask.any():
                    valid_idx = valid_mask.nonzero(as_tuple=True)[0][0]
                    sample_item_id = item_ids[0][valid_idx].item()
                    print(f"   - Item [{sample_item_id}] Sample: {pretrained_vecs[0][valid_idx][:5].tolist()}")
            '''
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
                'training_mode': False # Dropout 비활성화
            }

            # 2. User Tower Forward
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                output = model(**forward_kwargs) # (Batch, Seq_Len, Dim)
                
            # 3. 실제 마지막 유효 시점의 벡터 추출 (단순히 -1이 아니라 Padding을 고려)
            if output.dim() == 3:
                lengths = (~padding_mask).sum(dim=1)
                last_indices = (lengths - 1).clamp(min=0)
                batch_range = torch.arange(output.size(0), device=device)
                last_user_emb = output[batch_range, last_indices]
            else:
                last_user_emb = output
                
            # L2 정규화
            last_user_emb = F.normalize(last_user_emb, p=2, dim=1)
            
            # 4. 정답지(target_dict)에 존재하는 유효한 유저만 필터링
            valid_idx_list = [i for i, uid in enumerate(user_ids) if uid in target_dict and len(target_dict[uid]) > 0]
            if not valid_idx_list: 
                continue 
                
            v_idx = torch.tensor(valid_idx_list, device=device)
            valid_user_emb = last_user_emb[v_idx]
            
            # 5. 전체 아이템과 내적하여 Score 계산
            scores = torch.matmul(valid_user_emb, norm_item_embeddings.T)
            
            # 6. Top-K 인덱스 추출
            _, topk_indices = torch.topk(scores, k=max_k, dim=-1)
            pred_ids = topk_indices.cpu().numpy() 
            
            # 7. 실제 정답(Set)과 교집합 비교하여 Recall@K 측정
            for i, original_idx in enumerate(valid_idx_list):
                u_id = user_ids[original_idx]
                
                # 💡 [안전 장치] 정답이 단일 문자열이든 리스트든 무조건 리스트로 취급하게 만듦
                raw_targets = target_dict[u_id]
                if isinstance(raw_targets, str) or not hasattr(raw_targets, '__iter__'):
                    raw_targets = [raw_targets]
                
                # 💡 리스트로 만들어진 raw_targets를 순회
                actual_indices = set(processor.item2id[iid] for iid in raw_targets if iid in processor.item2id)
                
                # 만약 정답 아이템들이 모델이 모르는(OOT/Unseen) 아이템이라 매핑 후 세트가 비어있다면, 
                # 맞출 가능성이 0이므로 평가 타겟 유저에서 제외 (분모 증가 방지)
                if not actual_indices:
                    continue
                
                total_valid_users += 1
                for k in k_list:
                    # 예측한 Top-K 리스트(pred_ids) 중 단 하나라도 실제 구매 목록(actual_indices)에 포함되어 있다면 Hit
                    if not actual_indices.isdisjoint(pred_ids[i, :k]):
                        total_hits[k] += 1

    # 최종 Recall 퍼센티지 계산
    results = {}
    if total_valid_users > 0:
        for k in k_list:
            results[f'Recall@{k}'] = (total_hits[k] / total_valid_users) * 100
            
    print(f"\n📈 [Validation Results] Valid Users: {total_valid_users}")
    for k in k_list:
        print(f"   - Recall@{k:03d}: {results.get(f'Recall@{k}', 0):.2f}%")
        
    return results


from tqdm import tqdm
import wandb

def train_user_tower_all_time(epoch, model, item_tower, log_q_tensor, dataloader, optimizer, scaler, cfg, device, seq_labels=None, static_labels=None):
    """단일 에포크 훈련 함수 (All Time Steps + Same-User Masking 적용)"""
    model.train()
    total_loss_accum = 0.0
    main_loss_accum = 0.0
    cl_loss_accum = 0.0
    
    # 안전을 위해 labels가 None일 경우 빈 리스트로 초기화
    seq_labels = seq_labels or []
    static_labels = static_labels or []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        # -------------------------------------------------------
        # 1. Data Unpacking
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

        # -------------------------------------------------------
        # 2. Forward & Loss Calculation (AMP)
        # -------------------------------------------------------
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            output_1 = model(**forward_kwargs)
            output_2 = model(**forward_kwargs)

            # =======================================================
            # 💡 [핵심] (1) Main Loss (All Time Steps -> 1D Flattening)
            # =======================================================
            valid_mask = ~padding_mask # (Batch, Seq) True면 유효 데이터
            
            # 1. 2D 텐서를 유효한 타임스텝만 1D로 필터링 (N, Dim) 및 (N,)
            flat_output = output_1[valid_mask] 
            flat_targets = target_ids[valid_mask]
            
            # 2. 유저 ID 매핑 트릭: 문자열 ID 대신 현재 배치의 행(Row) 인덱스를 고유 ID로 사용
            # (Batch, 1) 사이즈의 인덱스를 Seq 길이만큼 늘린 뒤 똑같이 Flatten 합니다.
            batch_size, seq_len = item_ids.shape
            batch_row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
            flat_user_ids = batch_row_indices[valid_mask] # (N,) -> Same-User Masking에 사용됨
            
            if flat_output.size(0) > 0:
                flat_user_emb = F.normalize(flat_output, p=2, dim=1)
                
                # 실시간 아이템 벡터 추출
                full_item_embeddings = item_tower.get_all_embeddings()
                norm_item_embeddings = F.normalize(full_item_embeddings, p=2, dim=1)
                
                # 베이스라인 Loss 호출 (Same-User Masking 적용)
                main_loss = inbatch_corrected_logq_loss(
                    user_emb=flat_user_emb,
                    item_tower_emb=norm_item_embeddings,
                    target_ids=flat_targets,
                    user_ids=flat_user_ids,  # 배치 내 로컬 고유 ID 전달
                    log_q_tensor=log_q_tensor,
                    temperature=0.1,         # Baseline 온도
                    lambda_logq=cfg.lambda_logq
                )
            else:
                main_loss = torch.tensor(0.0, device=device)
            
            # =======================================================
            # (2) DuoRec Loss (여전히 Last Time Step Only 적용)
            # =======================================================
            # DuoRec은 시퀀스의 '최종 의도' 안정화에 목적이 있으므로 마지막 스텝만 사용하는 것이 맞습니다.
            last_indices = (valid_mask.sum(dim=1) - 1).clamp(min=0)
            batch_range = torch.arange(batch_size, device=device)
            
            last_output_1 = output_1[batch_range, last_indices]
            last_output_2 = output_2[batch_range, last_indices]
            last_targets = target_ids[batch_range, last_indices]
            
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        # 누적 및 로깅
        total_loss_accum += total_loss.item()
        main_loss_accum += main_loss.item()
        cl_loss_accum += cl_loss.item()
        
        pbar.set_postfix({
            'Loss': f"{total_loss.item():.4f}",
            'Main': f"{main_loss.item():.4f}",
            'CL': f"{cl_loss.item():.4f}"
        })
        
        if batch_idx % 100 == 0:
            wandb.log({
                "Train/Main_Loss_Step": main_loss.item(),
                "Train/CL_Loss_Step": cl_loss.item(),
                "Step": epoch * len(dataloader) + batch_idx
            })

    avg_loss = total_loss_accum / len(dataloader)
    avg_main = main_loss_accum / len(dataloader)
    avg_cl = cl_loss_accum / len(dataloader)

    # Gate Weights Logging
    with torch.no_grad():
        s_weights = torch.sigmoid(model.seq_gate).cpu().numpy()
        u_weights = torch.sigmoid(model.static_gate).cpu().numpy()
        
        gate_log = {}
        if seq_labels and len(seq_labels) == len(s_weights):
            gate_log.update({f"Gate/Seq_{label}": w for label, w in zip(seq_labels, s_weights)})
        if static_labels and len(static_labels) == len(u_weights):
            gate_log.update({f"Gate/Static_{label}": w for label, w in zip(static_labels, u_weights)})
            
        if gate_log:
            wandb.log(gate_log)

    print(f"🏁 Epoch {epoch} Completed | Avg Total: {avg_loss:.4f} (Main: {avg_main:.4f}, CL: {avg_cl:.4f})")
    return avg_loss
# =====================================================================
# Main Execution Pipeline
# =====================================================================
def run_pipeline():
    """Airflow DAG나 MLflow Run에서 직접 호출하는 엔트리 포인트"""
    print("🚀 Starting User Tower Training Pipeline...")
    
    
    SEQ_LABELS = ['item_id', 'time', 'type', 'color', 'graphic', 'section']
    STATIC_LABELS = ['age', 'price', 'cnt', 'recency', 'channel', 'club', 'news', 'fn', 'active', 'cont']
    # 1. Config & Env
    cfg = PipelineConfig()
    device = setup_environment()
    processor, val_processor, cfg = prepare_features(cfg)
    
    # item metadata cfg
    HASH_SIZE = 1000
    cfg.num_prod_types = HASH_SIZE
    cfg.num_colors = HASH_SIZE
    cfg.num_graphics = HASH_SIZE
    cfg.num_sections = HASH_SIZE
    
    # 2. Data 가져오기
    aligned_vecs = load_aligned_pretrained_embeddings(processor, cfg.model_dir, cfg.pretrained_dim)
    # ❌ full_item_embeddings = aligned_vecs.to(device) # 더 이상 사용하지 않음
    
    item_state_dict = load_item_tower_state_dict(cfg.model_dir, cfg.item_tower_pth_name, device)
    log_q_tensor = processor.get_logq_probs(device)
    
    item_metadata_tensor = load_item_metadata_hashed(processor, cfg.base_dir, hash_size=HASH_SIZE)
    processor.i_side_arr = item_metadata_tensor.numpy()
    
    train_loader = create_dataloaders(processor, cfg, aligned_vecs, is_train=True)
    val_loader = create_dataloaders(val_processor, cfg, aligned_vecs, is_train=False)
    dataset_peek(train_loader.dataset, processor)
    
    
    
        
    wandb.init(
        project="SASRec-User-Tower-causality-Optimization", # 프로젝트명
        name=f"run_lr_{cfg.lr}_epoch_{cfg.epochs}", # 실험 이름
        config=cfg.__dict__ # 하이퍼파라미터 저장
    )
    
    
    # -----------------------------------------------------------
    # 3. Models & Optimizer Setup (초기 상태: Epoch 1용 세팅)
    # -----------------------------------------------------------
    user_tower, item_tower = setup_models(cfg, device, item_state_dict, log_q_tensor)
    TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")
    
    # 💡 [핵심 반영] 아까 만든 깔끔한 메서드로 사전학습 벡터 강제 주입!
    item_tower.init_from_pretrained(aligned_vecs.to(device))
    
    # 💡 [초기화] Epoch 1에서는 User Tower만 학습하도록 Item Tower 완전 동결
    item_tower.set_freeze_state(True)
    print(f"❄️ Epoch 1: Item Tower FROZEN! (User Tower LR: {cfg.lr})")
    
    # User Tower만 포함된 Optimizer 생성
    optimizer = torch.optim.AdamW(user_tower.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # 💡 [스케줄러] Validation 지표(Recall@100)를 보고 정체 시 학습률 감소 (patience=1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=1
    )
    
    # Best Model 트래킹 변수
    best_recall_100 = 0.0

    # -----------------------------------------------------------
    # 4. Training Loop
    # -----------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):
        
        # 💡 [동적 Unfreeze] Epoch 2 진입 시 딱 한 번 실행하여 Joint Training 시작
        if epoch == 2:
            print("\n🔥 [Dynamic Unfreeze] Epoch 2: Item Tower Joint Training 시작!")
            item_tower.set_freeze_state(False)
            item_finetune_lr = cfg.lr * 0.05 # 아이템은 매우 미세하게만 조정 (User LR의 5%)
            
            # 기존 옵티마이저에 아이템 타워의 파라미터 그룹을 런타임에 동적으로 추가
            optimizer.add_param_group({
                'params': item_tower.parameters(), 
                'lr': item_finetune_lr
            })
            print(f"   - User Tower LR: {cfg.lr}")
            print(f"   - Item Tower LR: {item_finetune_lr} (Fine-tuning mode)")

        # ------------------- 훈련 (Train) -------------------
        avg_loss = train_user_tower_all_time(
            epoch=epoch,
            model=user_tower,
            item_tower=item_tower, # 정적 벡터 대신 모델 객체 자체를 넘김
            log_q_tensor=log_q_tensor,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            seq_labels = SEQ_LABELS,
            static_labels = STATIC_LABELS
        )
        
        # ------------------- 평가 (Evaluate) -------------------
        val_metrics = evaluate_model(
            model=user_tower, 
            item_tower=item_tower, 
            dataloader=val_loader,
            target_df_path=TARGET_VAL_PATH,
            device=device,
            processor=processor,
            k_list=[20, 100, 500]
        )
        
        current_recall_100 = val_metrics.get('Recall@100', 0.0)
        
        # ------------------- 스케줄러 & Best Model 저장 -------------------
        scheduler.step(current_recall_100)
        
        if current_recall_100 > best_recall_100:
            print(f"🌟 [New Best!] Recall@100 updated: {best_recall_100:.2f}% -> {current_recall_100:.2f}%")
            best_recall_100 = current_recall_100
            
            # 최고 성능 달성 시 파라미터 덮어쓰기 저장
            torch.save(user_tower.state_dict(), os.path.join(cfg.model_dir, "best_user_tower_c.pth"))
            torch.save(item_tower.state_dict(), os.path.join(cfg.model_dir, "best_item_tower_c.pth"))
            print("   💾 Best model weights saved to disk.")
        else:
            print(f"   - (Current Best: {best_recall_100:.2f}%)")
            
    print("\n🎉 Pipeline Execution Finished Successfully!")

def run_resume_pipeline(resume_epoch=6, last_best_recall=9.69):
    """저장된 모델을 불러와 Epoch 6부터 재학습을 진행하는 엔트리 포인트"""
    print(f"🚀 Resuming User Tower Training from Epoch {resume_epoch}...")
    # 모델 구조와 일치하는 이름표 정의
    
    
    
    
    
    
    SEQ_LABELS = ['item_id', 'time', 'type', 'color', 'graphic', 'section']
    STATIC_LABELS = ['age', 'price', 'cnt', 'recency', 'channel', 'club', 'news', 'fn', 'active', 'cont']
    
    cfg = PipelineConfig()
    device = setup_environment()
    processor, val_processor, cfg = prepare_features(cfg)
    # processor.analyze_distributions()
    HASH_SIZE = 1000 
    cfg.num_prod_types = HASH_SIZE
    cfg.num_colors = HASH_SIZE
    cfg.num_graphics = HASH_SIZE
    cfg.num_sections = HASH_SIZE
    # 아이템 개수도 processor에서 가져와서 정확히 매칭 (매우 중요)
    cfg.num_items = len(processor.item2id)
    aligned_vecs = load_aligned_pretrained_embeddings(processor, cfg.model_dir, cfg.pretrained_dim)
    log_q_tensor = processor.get_logq_probs(device)
    item_metadata_tensor = load_item_metadata_hashed(processor, cfg.base_dir, hash_size=HASH_SIZE)
    processor.i_side_arr = item_metadata_tensor.numpy()
    
    
    wandb.init(
        project="SASRec-User-Tower-Optimization", # 프로젝트명
        name=f"run_lr_{cfg.lr}_epoch_{cfg.epochs}", # 실험 이름
        config=cfg.__dict__ # 하이퍼파라미터 저장
    )
    
    
    train_loader = create_dataloaders(processor, cfg, aligned_vecs, is_train=True)
    val_loader = create_dataloaders(val_processor, cfg, aligned_vecs, is_train=False)
    
    # 2. 모델 생성
    # item_state_dict는 초기화용이므로 비워두거나 기본 로드 후 가중치를 덮어씌웁니다.
    user_tower, item_tower = setup_models(cfg, device, {}, log_q_tensor)
    
    # 3. [핵심] 가중치 불러오기 (Best 모델 로드)
    print("📂 Loading best weights for Resume...")
    user_weight_path = os.path.join(cfg.model_dir, "best_user_tower_fout.pth")
    item_weight_path = os.path.join(cfg.model_dir, "best_item_tower_fout.pth")

    if os.path.exists(user_weight_path) and os.path.exists(item_weight_path):
        # torch.load는 파일만 읽고, strict 옵션은 load_state_dict에 줍니다.
        user_state_dict = torch.load(user_weight_path, map_location=device)
        user_tower.load_state_dict(user_state_dict, strict=False) 
        
        item_state_dict = torch.load(item_weight_path, map_location=device)
        item_tower.load_state_dict(item_state_dict, strict=False)
        
        print("✅ Successfully loaded best weights from disk (Feature Gates Initialized).")
        # 4. Optimizer & Scheduler 설정
        # 재학습 시에는 Item Tower를 바로 학습 가능 상태로 둡니다.
        item_tower.set_freeze_state(True)
    
    # 두 타워의 파라미터를 처음부터 나누어 관리
    user_lr = 5e-4   # 재학습이므로 기존 LR보다 절반 정도로 낮게 시작하는 것을 추천
    item_lr = user_lr * 0.05
    
    optimizer = torch.optim.AdamW([
        {'params': user_tower.parameters(), 'lr': user_lr},
        {'params': item_tower.parameters(), 'lr': item_lr}
    ], weight_decay=cfg.weight_decay)
    
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',          # Recall@100 기준
        factor=0.3,          # [조정] 0.5보다 조금 더 과감하게 깎아서 정착 유도
        patience=3,          # [조정] 2에서 3으로 증가. HNM은 적응 기간이 필요함
        threshold=1e-4,      # 미세한 개선도 인정
        min_lr=1e-6,         # 최소 학습률 하한선
    )
    
    best_recall_100 = last_best_recall # 9.69% 부터 시작
    TARGET_VAL_PATH = os.path.join(cfg.base_dir, "features_target_val.parquet")

    # 5. Training Loop (Epoch 6 ~ 10 등)
    total_epochs = resume_epoch + 4 # 예: 5에포크 더 학습
    for epoch in range(resume_epoch, total_epochs + 1):
        
        avg_loss = train_user_tower(
            epoch=epoch,
            model=user_tower,
            item_tower=item_tower,
            log_q_tensor=log_q_tensor,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device,
            seq_labels = SEQ_LABELS,
            static_labels = STATIC_LABELS
        )
        
        val_metrics = evaluate_model(
            model=user_tower, 
            item_tower=item_tower, 
            dataloader=val_loader,
            target_df_path=TARGET_VAL_PATH,
            device=device,
            processor=processor,
            k_list=[20, 100, 500]
        )
        
        current_recall_100 = val_metrics.get('Recall@100', 0.0)
        scheduler.step(current_recall_100)
        
        if current_recall_100 > best_recall_100:
            print(f"🌟 [New Best!] Recall@100 updated: {best_recall_100:.2f}% -> {current_recall_100:.2f}%")
            best_recall_100 = current_recall_100
            torch.save(user_tower.state_dict(), os.path.join(cfg.model_dir, "best_user_tower_hmn.pth"))
            torch.save(item_tower.state_dict(), os.path.join(cfg.model_dir, "best_item_tower_hmn.pth"))
            print("💾 Best model weights updated.")
        else:
            print(f" - (Current Best: {best_recall_100:.2f}%)")

    print("\n🎉 Resume Training Finished!")

if __name__ == "__main__":
    # 5에포크까지 학습했으므로 6번부터 재개
    #run_resume_pipeline(resume_epoch=26, last_best_recall=17.55)
    run_pipeline()