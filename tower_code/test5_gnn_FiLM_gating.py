import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup
import pandas as pd
import numpy as np
import os
import random
import math
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import gc
import warnings
import logging

warnings.filterwarnings("ignore", message="Support for mismatched src_key_padding_mask and mask is deprecated")

# ==========================================
# ⚙️ 설정 & 경로
# ==========================================
#TEMPERATURE = 0.2
LAMBDA_LOGQ = 0.1
BATCH_SIZE = 896
EMBED_DIM = 128
MAX_SEQ_LEN = 50
DROPOUT = 0.3
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = r"D:\trainDataset\localprops"
MODEL_DIR = r"C:\Users\candyform\Desktop\inferenceCode\models"
CACHE_DIR = os.path.join(BASE_DIR, "cache")

ITEM_FEAT_PATH_PQ = os.path.join(BASE_DIR, "features_item.parquet")
USER_FEAT_PATH_PQ = os.path.join(BASE_DIR, "features_user.parquet")
SEQ_DATA_PATH_PQ = os.path.join(BASE_DIR, "features_sequence_cleaned.parquet")
TARGET_VAL_PATH = os.path.join(BASE_DIR, "features_target_val.parquet")
USER_VAL_FEAT_PATH = os.path.join(BASE_DIR, "features_user_val.parquet")
SEQ_VAL_DATA_PATH = os.path.join(BASE_DIR, "features_sequence_val.parquet")

SAVE_PATH_BEST = os.path.join(MODEL_DIR, "user_tower_phase3_best_Film.pth")

class SmartLogger:
    def __init__(self, verbosity=1): self.verbosity = verbosity
    def log(self, level, msg):
        if self.verbosity >= level: print(f"[{'ℹ️' if level==1 else '📊'}] {msg}")

logger = SmartLogger(verbosity=1)

# ==========================================
# 1. Feature Processor & Dataset
# ==========================================
class FeatureProcessor:
    def __init__(self, user_path, item_path, seq_path, scaler=None, num_bins=10):
        self.users = pd.read_parquet(user_path)
        # 중복 제거 및 인덱스 설정
        self.users = self.users.drop_duplicates(subset=['customer_id']).set_index('customer_id')
        self.items = pd.read_parquet(item_path).set_index('article_id')
        self.seqs = pd.read_parquet(seq_path).set_index('customer_id')
        
        # 인덱스 타입 강제 (String)
        self.users.index = self.users.index.astype(str)
        self.items.index = self.items.index.astype(str)
        self.seqs.index = self.seqs.index.astype(str)

        self.user_ids = self.users.index.tolist()
        self.user2id = {uid: i + 1 for i, uid in enumerate(self.user_ids)}
        self.item_ids = self.items.index.tolist()
        self.item2id = {iid: i + 1 for i, iid in enumerate(self.item_ids)}
        
        self.u_dense_cols = ['user_avg_price_log', 'total_cnt_log', 'recency_log']
        self.users_scaled = self.users.copy()
        self.user_scaler = StandardScaler()
        
        # -------------------------------------------------------
        # ⭐ [Refactor] 1. Scaler 처리 (Dense Feature용)
        # -------------------------------------------------------
        if scaler is None: 
            self.user_scaler = StandardScaler()
            scaled_data = self.user_scaler.fit_transform(self.users[self.u_dense_cols])
        else: 
            self.user_scaler = scaler
            scaled_data = self.user_scaler.transform(self.users[self.u_dense_cols])
        
        self.users_scaled[self.u_dense_cols] = np.nan_to_num(scaled_data, nan=0.0)

        # -------------------------------------------------------
        # ⭐ [Refactor] 2. Bucketing (Binning) -> Index화
        # -------------------------------------------------------
        # (1) Activity Index (Total Count) -> 0~9 등급
        # qcut: 데이터 분포에 따라 균등하게 N등분 (헤비유저/라이트유저 구분 용이)
        # duplicates='drop': 데이터가 쏠려있을 경우 구간 병합 방지
        try:
            self.users_scaled['activity_idx'] = pd.qcut(
                self.users['total_cnt_log'], q=num_bins, labels=False, duplicates='drop'
            ).fillna(0).astype(int)
        except ValueError:
            # 데이터가 너무 적거나 중복이 많아 qcut 실패 시 rank 기반 처리
            self.users_scaled['activity_idx'] = 0

        # (2) Price Index (Avg Price) -> 0~9 등급 (필요 시 사용)
        try:
            self.users_scaled['price_idx'] = pd.qcut(
                self.users['user_avg_price_log'], q=num_bins, labels=False, duplicates='drop'
            ).fillna(0).astype(int)
        except ValueError:
            self.users_scaled['price_idx'] = 0
            
        print(f"✅ [FeatureProcessor] Bucketing Complete (Bins={num_bins})")
        
        # NaN 방어
        self.users_scaled[self.u_dense_cols] = np.nan_to_num(scaled_data, nan=0.0)
    def get_user_tensor(self, user_id):
        # Dense Features (Float, Scaled)
        dense = torch.tensor(self.users_scaled.loc[user_id, self.u_dense_cols].values, dtype=torch.float32)
        
        # Categorical Features (Int)
        # preferred_channel은 1, 2로 되어있으므로 0, 1로 변환
        cat_channel = torch.tensor(int(self.users_scaled.loc[user_id, 'preferred_channel']) - 1, dtype=torch.long)
        
        # ⭐ 추가된 Bucketed Features
        cat_activity = torch.tensor(int(self.users_scaled.loc[user_id, 'activity_idx']), dtype=torch.long)
        cat_price = torch.tensor(int(self.users_scaled.loc[user_id, 'price_idx']), dtype=torch.long)
        
        return dense, cat_channel, cat_activity, cat_price
    def get_logq_probs(self, device):
        """
        모델의 Embedding(N+1, D) 구조와 일치하도록 인덱스 보정된 log_q 생성
        """
        # 1. raw_probability 추출 (0-based)
        raw_probs = self.items['raw_probability'].reindex(self.item_ids).values
        
        # 2. Smoothing 및 처리
        eps = 1e-6
        sorted_probs = np.nan_to_num(raw_probs, nan=0.0) + eps
        sorted_probs /= sorted_probs.sum()
        
        # 3. 로그 계산
        log_q_values = np.log(sorted_probs).astype(np.float32)
        
        # 4. [중요] 1-based 인덱싱 대응을 위한 Padding 추가
        # 0번 인덱스는 사용하지 않으므로 아주 작은 확률(또는 0)의 로그값으로 채움
        full_log_q = np.zeros(len(self.item_ids) + 1, dtype=np.float32)
        full_log_q[1:] = log_q_values  # 1번 인덱스부터 실제 값 채우기
        full_log_q[0] = -20.0          # 0번 인덱스(Padding)는 낮은 값으로 설정
    
        return torch.tensor(full_log_q, dtype=torch.float32).to(device)
class UserTowerDataset(Dataset):
    def __init__(self, processor, max_seq_len=50, is_training=True):
        self.processor = processor
        self.user_ids = processor.user_ids 
        self.max_len = max_seq_len
        self.is_training = is_training
        self.min_cut_len = 3      

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        u_id_str = self.user_ids[idx]
        u_dense, u_cat_channel, u_cat_activity, u_cat_price = self.processor.get_user_tensor(u_id_str)
        
        processed_tokens = []
        processed_deltas = []
        
        if u_id_str in self.processor.seqs.index:
            seq_row = self.processor.seqs.loc[u_id_str]
            # Series일 경우 처리
            if isinstance(seq_row, pd.DataFrame): seq_row = seq_row.iloc[0]
                
            for i, d in zip(seq_row['sequence_ids'], seq_row['sequence_deltas']):
                 token = self.processor.item2id.get(str(i), 0) # str 변환 안전장치
                 if token == 0: continue
                 processed_tokens.append(token)
                 processed_deltas.append(d)

        seq_len = len(processed_tokens)
        input_seq = []
        target_seq = [] 

        if seq_len > 0:
            if self.is_training:
                can_sample = seq_len > self.min_cut_len
                if not can_sample or random.random() < 0.8:
                    input_seq = processed_tokens[:-1]
                    target_seq = processed_tokens[1:]
                else:
                    max_cut = seq_len - 1
                    cut_idx = seq_len if max_cut < self.min_cut_len else random.randint(self.min_cut_len, max_cut)
                    full_slice = processed_tokens[:cut_idx+1]
                    input_seq = full_slice[:-1]
                    target_seq = full_slice[1:]
            else:
                input_seq = processed_tokens[:]
                target_seq = [0] * len(input_seq)

        input_ids = input_seq[-self.max_len:]
        target_ids = target_seq[-self.max_len:]
        input_deltas = processed_deltas[:len(input_seq)][-self.max_len:]

        return {
            'user_idx': torch.tensor(idx + 1, dtype=torch.long),
            'user_dense': u_dense,           # (3,) Float (Scaled)
            'user_cat': u_cat_channel,       # (1,) Long (0 or 1)
            'activity_idx': u_cat_activity,  # (1,) Long (0 ~ 9) ⭐ New
            'price_idx': u_cat_price,        # (1,) Long (0 ~ 9) ⭐ New (옵션)
            'seq_ids': torch.tensor(input_ids, dtype=torch.long),
            'seq_deltas': torch.tensor(input_deltas, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }

def user_tower_collate_fn(batch):
    u_idx = torch.stack([b['user_idx'] for b in batch])
    u_dense = torch.stack([b['user_dense'] for b in batch])
    u_cat = torch.stack([b['user_cat'] for b in batch])
        
        # ⭐ New: Activity & Price Index Batching
    activity_idx = torch.stack([b['activity_idx'] for b in batch])
    price_idx = torch.stack([b['price_idx'] for b in batch]) # 필요 시 사용
        
    seq_ids = pad_sequence([b['seq_ids'] for b in batch], batch_first=True, padding_value=0)
    seq_deltas = pad_sequence([b['seq_deltas'] for b in batch], batch_first=True, padding_value=0)
    target_ids = pad_sequence([b['target_ids'] for b in batch], batch_first=True, padding_value=0)
        
    seq_mask = (seq_ids != 0).long()
    last_target = torch.tensor([b['target_ids'][-1] if len(b['target_ids']) > 0 else 0 for b in batch], dtype=torch.long)
        
    # 리턴값에 activity_idx 추가
    return u_idx, u_dense, u_cat, activity_idx, seq_ids, seq_deltas, seq_mask, target_ids, last_target
# ==========================================
# 2. Alignment Functions (Alignment)
# ==========================================
def load_and_align_embeddings(model, processor, model_dir, device):
    """ Content Item Embedding Alignment (Pretrained -> model.item_content_emb) """
    print(f"\n🔄 [Content Alignment] Starting Item Embedding Alignment...")
    emb_path = os.path.join(model_dir, "pretrained_item_matrix.pt")
    ids_path = os.path.join(model_dir, "item_ids.pt")

    try:
        pretrained_emb = torch.load(emb_path, map_location='cpu')
        if isinstance(pretrained_emb, dict):
            pretrained_emb = pretrained_emb.get('weight', pretrained_emb.get('item_content_emb.weight'))
        pretrained_ids = torch.load(ids_path, map_location='cpu')
    except Exception as e:
        print(f"❌ [Error] Failed to load Content files: {e}")
        return model

    pretrained_map = {str(item_id.item()) if isinstance(item_id, torch.Tensor) else str(item_id): pretrained_emb[idx] for idx, item_id in enumerate(pretrained_ids)}
    
    num_embeddings = len(processor.item_ids) + 1 
    new_weight = torch.randn(num_embeddings, pretrained_emb.shape[1]) * 0.01 
    new_weight[0] = 0.0 
    
    matched = 0
    for i, current_id_str in enumerate(processor.item_ids):
        if current_id_str in pretrained_map:
            new_weight[i + 1] = pretrained_map[current_id_str]
            matched += 1
            
    with torch.no_grad():
        model.item_content_emb = nn.Embedding.from_pretrained(new_weight.to(device), freeze=False)
        
    print(f"✅ [Content Alignment] Matched: {matched}/{len(processor.item_ids)}")
    return model

def load_and_align_gnn_items(model, processor, base_dir, device):
    """ GNN Item Embedding Alignment (GNN -> model.gnn_item_emb) """
    print(f"\n🔄 [GNN Item Alignment] Starting...")
    cache_dir = os.path.join(base_dir, "cache")
    model_path = os.path.join(MODEL_DIR , "simgcl_trained.pth")
    maps_path = os.path.join(cache_dir, "id_maps_train.pt")

    try:
        maps = torch.load(maps_path, map_location='cpu')
        gnn_item2id = maps['item2id']
        gnn_state_dict = torch.load(model_path, map_location='cpu')
        gnn_emb_weight = gnn_state_dict['embedding_item.weight']
    except Exception as e:
        print(f"❌ [Error] Failed to load GNN Item files: {e}")
        return model

    num_embeddings = len(processor.item_ids) + 1 
    new_weight = torch.randn(num_embeddings, gnn_emb_weight.shape[1]) * 0.01
    new_weight[0] = 0.0

    matched = 0
    for i, current_id_str in enumerate(processor.item_ids):
        if current_id_str in gnn_item2id:
            new_weight[i + 1] = gnn_emb_weight[gnn_item2id[current_id_str]]
            matched += 1
            
    with torch.no_grad():
        # [중요] 반드시 gnn_item_emb 에 넣어야 함!
        model.gnn_item_emb = nn.Embedding.from_pretrained(new_weight.to(device), freeze=False)
        print(f"✅ Injected into 'model.gnn_item_emb'")

    print(f"✅ [GNN Item Alignment] Matched: {matched}/{len(processor.item_ids)}")
    return model

def load_and_align_gnn_user_embeddings(model, processor, base_dir, device):
    """ GNN User Embedding Alignment (GNN -> model.gnn_user_emb) """
    print(f"\n🔄 [GNN User Alignment] Starting...")
    cache_dir = os.path.join(base_dir, "cache")
    model_path = os.path.join(MODEL_DIR , "simgcl_trained.pth")
    maps_path = os.path.join(cache_dir, "id_maps_train.pt")

    try:
        maps = torch.load(maps_path, map_location='cpu')
        gnn_user2id = maps['user2id']
        gnn_state_dict = torch.load(model_path, map_location='cpu')
        
        # 유저 가중치 찾기
        gnn_user_weight = None
        for key, tensor in gnn_state_dict.items():
            if 'embedding_user' in key:
                gnn_user_weight = tensor
                break
        if gnn_user_weight is None: raise Exception("User embedding not found in state dict")

    except Exception as e:
        print(f"❌ [Error] Failed to load GNN User files: {e}")
        return model

    num_users = len(processor.user_ids) + 1
    new_weight = torch.randn(num_users, gnn_user_weight.shape[1]) * 0.01
    new_weight[0] = 0.0
    
    matched = 0
    for i, current_id_str in enumerate(processor.user_ids):
        if current_id_str in gnn_user2id:
            new_weight[i + 1] = gnn_user_weight[gnn_user2id[current_id_str]]
            matched += 1
            
    with torch.no_grad():
        # [중요] 반드시 gnn_user_emb 에 넣어야 함! (크기 96만)
        model.gnn_user_emb = nn.Embedding.from_pretrained(new_weight.to(device), freeze=False)
        print(f"✅ Injected into 'model.gnn_user_emb'")

    print(f"✅ [GNN User Alignment] Matched: {matched}/{len(processor.user_ids)}")
    return model

def verify_embedding_alignment(model, processor, model_dir):
    # (생략: 기존 코드와 동일, 필요시 추가)
    pass

# ==========================================
# 3. Model Definition (Fixed)
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class ComprehensiveGatedFiLM(nn.Module):
    """
    [Gated FiLM Architecture]
    1. Base: Sequence Vector (Main Signal)
    2. Context: GNN + Meta + Price (Condition)
    3. Modulation: Context가 Sequence의 Scale(Gamma)과 Shift(Beta)를 조절
    4. Gating: 변조된 결과가 좋은지 판단하여, 원본 Sequence와 섞음 (Residual)
    """
    def __init__(self, dim=128, 
                 gnn_dim=64, 
                 dense_dim=35,        # u_dense (Price, Cnt, Recency 등)
                 cat_cardinality=2,   # u_cat (Channel 등)
                 active_cardinality=10 # total_cnt 등급
                 ):
        super().__init__()
        
        # ==========================================
        # 1. Feature Encoders (Static Context 준비)
        # ==========================================
        
        # (A) Activity (신뢰도 Gate용 - 단순 참고용이 아니라 Context에 포함)
        self.activity_emb = nn.Embedding(active_cardinality, 16)
        
        # (B) Price (정규화된 Float 값 -> 벡터화)
        self.price_encoder = nn.Sequential(
            nn.Linear(1, 32), 
            nn.Tanh() # -1 ~ 1 정규화 가정
        )

        # (C) Cat & Dense Processor
        self.cat_emb = nn.Embedding(cat_cardinality, 16)
        
        # Context Dimension = GNN(Projected) + Meta + Price + Activity
        # 여기서는 내부에서 합쳐서 dim(128)으로 만듦
        self.context_encoder = nn.Sequential(
            # Input: GNN(64) + Dense(35) + Cat(16) + Price(32) + Activity(16)
            # 주의: Dense에 이미 Price/Cnt 등이 포함되어 있다면 중복 제거 필요하지만,
            # 여기서는 편의상 다 연결하고 Linear가 알아서 거르도록 함
            nn.Linear(gnn_dim + dense_dim + 16 + 32 + 16, dim), 
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # ==========================================
        # 2. FiLM Generator (Gamma, Beta)
        # ==========================================
        self.film_gen = nn.Linear(dim, dim * 2) # Input: Context -> Output: Gamma, Beta

        # Sequence Normalization (Affine=False 필수! Gamma/Beta를 우리가 만드니까)
        self.seq_ln = nn.LayerNorm(dim, elementwise_affine=False)

        # ==========================================
        # 3. Gating Network (The Judge)
        # ==========================================
        # 입력: [Original_Seq, Context] -> 출력: 0~1 (Channel-wise)
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim), 
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid() 
        )

        # ==========================================
        # 4. 초기화 (매우 중요: Start from Seq Only)
        # ==========================================
        self.apply(self._init_weights)
        
        # FiLM 초기화: Gamma=1, Beta=0 (Identity)
        nn.init.zeros_(self.film_gen.weight)
        with torch.no_grad():
            self.film_gen.bias[:dim].fill_(1.0) # Gamma
            self.film_gen.bias[dim:].fill_(0.0) # Beta

        # Gate 초기화: 0에 가깝게 (원본 Sequence 우선)
        # Bias를 음수로 밀어서 Sigmoid 통과 시 0.1 이하가 되도록 설정
        nn.init.constant_(self.gate_net[-2].bias, -3.0) 

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, v_seq, v_gnn, price_norm, u_dense, u_cat, activity_idx):
        """
        v_seq: (B, L, D) - Dynamic
        Others: (B, ...) - Static
        """
        B, L, D = v_seq.shape
        
        # 1. Static Features Encoding
        v_act = self.activity_emb(activity_idx)  # (B, 16)
        v_price = self.price_encoder(price_norm) # (B, 32)
        v_cat = self.cat_emb(u_cat)              # (B, 16)
        
        # 2. Context Integration (User Static Profile)
        # 모든 정적 정보를 모음
        raw_context = torch.cat([v_gnn, u_dense, v_cat, v_price, v_act], dim=1) # (B, Total_In)
        context = self.context_encoder(raw_context) # (B, D)

        # 3. Broadcasting (Static -> Sequence Length만큼 복사)
        # (B, D) -> (B, L, D)
        context_expanded = context.unsqueeze(1).expand(-1, L, -1)

        # 4. FiLM Modulation
        # Gamma, Beta 생성 (B, L, D)
        film_params = self.film_gen(context_expanded)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)

        # 변조 적용: Gamma * Norm(Seq) + Beta
        v_seq_norm = self.seq_ln(v_seq)
        v_modulated = gamma * v_seq_norm + beta
        
        # 5. Gating (Selective Fusion)
        # Gate 입력: "원래 시퀀스"와 "유저 컨텍스트"를 보고 결정
        gate_input = torch.cat([v_seq, context_expanded], dim=-1) # (B, L, D*2)
        gate = self.gate_net(gate_input) # (B, L, D) -> 0.0 ~ 1.0

        # 최종 출력: Gate가 열리면 변조된 값, 닫히면 원본 시퀀스
        output = gate * v_modulated + (1 - gate) * v_seq
        
        return output, gate.mean()

# ==========================================
# 🧩 3. Parallel Adapter (유지)
# ==========================================
class ParallelAdapter(nn.Module):
    def __init__(self, content_dim=128, gnn_dim=64, out_dim=128, dropout=0.2):
        super().__init__()
        self.content_proj = nn.Sequential(
            nn.Linear(content_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gnn_proj = nn.Sequential(
            nn.Linear(gnn_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, v_content, v_gnn):
        # [수정] Content Embedding에 Residual Connection 추가 (+ v_content)
        # v_content(원본)가 Adapter를 통과한 결과와 더해짐 -> 원본 정보 보존
        merged = (self.content_proj(v_content) + v_content) + self.gnn_proj(v_gnn)
        return merged

# ==========================================
# 🏰 Hybrid User Tower (수정됨)
# ==========================================
class HybridUserTower(nn.Module):
    def __init__(self, num_users, num_items, gnn_user_init, gnn_item_init, item_content_init):
        super().__init__()
        self.embed_dim = 128
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 1. Embeddings
        self.gnn_user_emb = nn.Embedding.from_pretrained(gnn_user_init, freeze=False)
        self.gnn_item_emb = nn.Embedding.from_pretrained(gnn_item_init, freeze=False)
        self.item_content_emb = nn.Embedding.from_pretrained(item_content_init, freeze=False)
        
        # 2. Adapters (Shared)
        # GNN Projector (For User GNN)
        self.gnn_projector = nn.Sequential(
            nn.Linear(gnn_user_init.shape[1], 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(256, 128), nn.LayerNorm(128)
        )
        
        # Item & Seq Projector
        self.seq_adapter = ParallelAdapter(
            content_dim=128, 
            gnn_dim=64, 
            out_dim=128, 
            dropout=DROPOUT
        )
        
        # 3. Sequence Modeling
        self.time_emb = nn.Embedding(1001, 128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=2, dim_feedforward=512, 
            dropout=DROPOUT, batch_first=True, norm_first=True
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # 4. Fusion Layer (Gated FiLM 교체) ✅
        self.fusion_layer = ComprehensiveGatedFiLM(
            dim=128, 
            gnn_dim=128, # gnn_projector 통과 후 차원
            dense_dim=3, # price_log, cnt_log, recency_log
            cat_cardinality=2,
            active_cardinality=10
        )

    def get_current_temperature(self, clamp_min):
        scale = self.logit_scale.exp().clamp(clamp_min, max=100.0)
        return 1.0 / scale
    
    # Target Item 생성 함수 (Shared Adapter 적용) ✅
    def get_item_representation(self, item_ids):
        raw_content = self.item_content_emb(item_ids)
        raw_gnn = self.gnn_item_emb(item_ids)
        item_vec = self.seq_adapter(raw_content, raw_gnn)
        return F.normalize(item_vec, p=2, dim=-1)

    def forward(self, u_idx, seq_ids, seq_deltas, seq_mask, u_dense, u_cat, activity_idx):
        B, L = seq_ids.shape
        
        # 1. GNN User (Static)
        v_gnn = self.gnn_projector(self.gnn_user_emb(u_idx)) # (B, 128)
        
        # Training 시 GNN Dropout (Aux Loss 학습 강제용)
        if self.training:
            drop_prob = 0.5 # 50% 확률로 GNN 끄기
            mask = torch.bernoulli(torch.full((B, 1), 1 - drop_prob, device=v_gnn.device))
            v_gnn = v_gnn * mask # Inverted 아님. 그냥 0으로 날림.
            
            # GNN이 꺼지면 Activity Idx도 영향을 받아야 하므로
            # 아래 Fusion Layer로 넘길 때 주의 필요하지만, 
            # FiLM은 Context 전체를 보기 때문에 괜찮음.

        # 2. Sequence Input Processing
        raw_content = self.item_content_emb(seq_ids)
        raw_gnn = self.gnn_item_emb(seq_ids)
        
        # Shared Adapter
        seq_input = self.seq_adapter(raw_content, raw_gnn)
        
        # Time Embedding
        seq_input = seq_input * math.sqrt(self.embed_dim) + self.time_emb(seq_deltas.clamp(max=1000))
        
        # Transformer
        causal_mask = torch.triu(torch.ones(L, L, device=seq_ids.device) * float('-inf'), diagonal=1)
        key_padding_mask = (seq_mask == 0)
        seq_out = self.seq_encoder(seq_input, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        
        # 3. Pure Sequence Vector (Aux Loss용) ✅
        v_seq = F.normalize(seq_out, p=2, dim=2)

        # 4. Pre-processing for FiLM Inputs
        # u_dense에서 price, activity 분리 (FeatureProcessor 로직에 따름)
        # u_dense 순서: ['user_avg_price_log', 'total_cnt_log', 'recency_log'] 가정
        price_norm = u_dense[:, 0:1] # (B, 1)
        
        # 5. Gated FiLM Fusion
        output, gate_avg = self.fusion_layer(
                    v_seq=v_seq,
                    v_gnn=v_gnn,
                    price_norm=price_norm,
                    u_dense=u_dense,
                    u_cat=u_cat,
                    activity_idx=activity_idx # ⭐ 전달
        )
                
        output = F.normalize(output, p=2, dim=2)

        # Return: Final, Seq_Only, Gate_Log
        return output, v_seq, gate_avg
    
    
    
    def get_meta_feature_importance(self):
        """
        Meta MLP의 첫 번째 Linear Layer 가중치를 분석하여
        어떤 Feature가 가장 영향력이 큰지 계산합니다.
        """
        # 첫 번째 Linear Layer의 가중치: (Out_Dim, In_Dim) -> (128, 35)
        weight_matrix = self.meta_mlp[0].weight.abs().detach().cpu()
        
        # Input Dimension Slicing
        # Price: 0~32, Cnt: 32~64, Recency: 64~96, Channel: 96~112
        imp_price = weight_matrix[:, 0:32].mean().item()
        imp_cnt = weight_matrix[:, 32:64].mean().item()
        imp_recency = weight_matrix[:, 64:96].mean().item()
        imp_channel = weight_matrix[:, 96:112].mean().item()
        
        # 정규화 (비율로 보기 위해)
        total = imp_price + imp_cnt + imp_recency + imp_channel + 1e-9
        return {
            "Price": imp_price / total,
            "Count": imp_cnt / total,
            "Recency": imp_recency / total,
            "Channel": imp_channel / total
        }
# ==========================================
# 4. Loss & Eval
# ==========================================
def logq_correction_loss(user_emb, item_emb, pos_item_ids, item_probs, temperature=0.07, lambda_logq=0.0):
    scores = torch.matmul(user_emb, item_emb.T)
    if lambda_logq > 0.0:
        
        log_q = torch.log(item_probs[pos_item_ids] + 1e-4).view(1, -1)
        scores = scores - (lambda_logq * log_q)
    logits = scores / temperature
    is_collision = (pos_item_ids.unsqueeze(1) == pos_item_ids.unsqueeze(0))
    mask = is_collision.fill_diagonal_(False)
    logits = logits.masked_fill(mask, -1e4)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)

def efficient_corrected_logq_loss(
    user_emb, 
    item_emb, 
    pos_item_ids, 
    precomputed_log_q, 
    temperature=0.1, 
    lambda_logq=0.1
):
    # 인덱스 범위 체크 (디버깅용, 실제 학습시 성능 영향 미미)
    assert pos_item_ids.max() < precomputed_log_q.size(0), "pos_item_ids contains out-of-bounds index!"
    logits = torch.matmul(user_emb, item_emb.T)
    logits.div_(temperature) # logits /= temperature (In-place)
    
    if lambda_logq > 0.0:
        # 2. LogQ Correction (In-place)
        # precomputed_log_q에서 현재 배치의 값만 슬라이싱 (View 생성)
        batch_log_q = precomputed_log_q[pos_item_ids].view(1, -1)
        
        # In-place subtraction: 새로운 텐서 할당 최소화
        logits.sub_(batch_log_q * lambda_logq)
        
        # 3. Positive Recovery (RecSys 2025)
        # torch.sum 대신 einsum을 쓰면 가끔 특정 CUDA 버전에서 더 효율적입니다.
        pos_logits_raw = torch.einsum('bd,bd->b', user_emb, item_emb).div_(temperature)
        logits.diagonal().copy_(pos_logits_raw)

    # 4. Collision Masking (메모리 절약형)
    with torch.no_grad():
        is_collision = (pos_item_ids.unsqueeze(1) == pos_item_ids.unsqueeze(0))
        mask = is_collision.fill_diagonal_(False)
    
    # FP16 AMP 사용 시 -3e4가 안전 (Underflow 방지)
    mask_value = -30000.0 if logits.dtype == torch.float16 else -1e9
    logits.masked_fill_(mask, mask_value)

    # 5. Labels 생성 (매번 생성하지 않고 재사용 가능하지만, 이 정도는 미미함)
    labels = torch.arange(logits.size(0), device=logits.device)
    
    return F.cross_entropy(logits, labels)

def evaluate_recall_multi_k(model, processor, target_df_path, k_list=[20, 100, 500], batch_size=256):
    model.eval()
    target_df = pd.read_parquet(target_df_path)
    target_dict = target_df.set_index('customer_id')['target_ids'].to_dict()
    
    val_loader = DataLoader(UserTowerDataset(processor, is_training=False), batch_size=batch_size, shuffle=False, collate_fn=user_tower_collate_fn)
    
    with torch.no_grad():
        all_item_ids = torch.arange(1, len(processor.item_ids)+1).to(DEVICE)
        
        # 1. 배치 단위로 처리 (아이템이 많으면 OOM 날 수 있으므로)
        all_item_vecs_list = []
        chunk_size = 4096 
        
        for i in range(0, len(all_item_ids), chunk_size):
            chunk_ids = all_item_ids[i : i + chunk_size]
            
            # Content + GNN 가져오기
            chunk_content = model.item_content_emb(chunk_ids)
            chunk_gnn = model.gnn_item_emb(chunk_ids)
            
            # Adapter 통과
            chunk_vecs = model.seq_adapter(chunk_content, chunk_gnn)
            chunk_vecs = F.normalize(chunk_vecs, p=2, dim=1)
            
            all_item_vecs_list.append(chunk_vecs)
            
        all_item_vecs = torch.cat(all_item_vecs_list, dim=0)
    
    hit_counts = {k: 0 for k in k_list}
    total_users = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # ⭐ [수정 1] activity_idx 언패킹 추가
            # collate_fn 리턴 순서: u_idx, u_dense, u_cat, activity_idx, seq_ids, seq_deltas, seq_mask, target_ids, last_target
            u_idx, u_dense, u_cat, activity_idx, seq_ids, seq_deltas, seq_mask, _, _ = [x.to(DEVICE) for x in batch]

            batch_uids = [processor.user_ids[i-1] for i in u_idx.cpu().numpy()]
            valid_idx_list = [i for i, uid in enumerate(batch_uids) if uid in target_dict]
            if not valid_idx_list: continue
            
            v_idx = torch.tensor(valid_idx_list).to(DEVICE)
            
            # ⭐ [수정 2] 모델 Forward에 activity_idx 전달
            # 리턴값: output(FiLM 적용), v_seq(순수 시퀀스), gate_avg
            # 평가시에는 FiLM이 적용된 최종 성능(output)을 봐야 하므로 첫 번째 리턴값을 사용합니다.
            output, _, _ = model(
                u_idx[v_idx], 
                seq_ids[v_idx], 
                seq_deltas[v_idx], 
                seq_mask[v_idx], 
                u_dense[v_idx], 
                u_cat[v_idx],
                activity_idx[v_idx] # <-- 필수 전달!
            )
            
            lengths = seq_mask[v_idx].sum(dim=1)
            last_indices = (lengths - 1).clamp(min=0)
            batch_range = torch.arange(output.size(0), device=DEVICE)
            
            # FiLM이 적용된 최종 벡터 추출
            last_user_vecs = output[batch_range, last_indices]
            
            scores = torch.matmul(last_user_vecs, all_item_vecs.T)
            _, topk_indices = torch.topk(scores, k=max(k_list), dim=1)
            pred_ids = (topk_indices + 1).cpu().numpy()
            
            for i, original_idx in enumerate(valid_idx_list):
                u_id = batch_uids[original_idx]
                actual_indices = set(processor.item2id[tid] for tid in target_dict[u_id] if tid in processor.item2id)
                if not actual_indices: continue
                for k in k_list:
                    if not actual_indices.isdisjoint(pred_ids[i, :k]): hit_counts[k] += 1
                total_users += 1
    
    metrics = {f"R@{k}": (hit_counts[k] / total_users if total_users > 0 else 0.0) for k in k_list}
    logger.log(1, f"📊 Eval Result: {metrics}")
    model.train()
    return metrics

def sanity_check_indices(pos_item_ids, num_embeddings):
    max_idx = pos_item_ids.max().item()
    min_idx = pos_item_ids.min().item()
    
    if max_idx >= num_embeddings or min_idx < 0:
        raise ValueError(f"ID Mapping Error! "
                         f"Batch Max Index: {max_idx}, "
                         f"Embedding Size: {num_embeddings}. "
                         f"인덱스가 범위를 벗어났습니다.")
    else:
        print(f"Check Passed: Index {min_idx} ~ {max_idx} is safe for Embedding({num_embeddings})")
# ==========================================
# 5. Main Training Function (Refactored)
# ==========================================
def train_phase_3_FiLM():
    logger.log(1, "🚀 Phase 3: 2-Stage Training / FiLM (Warm-up -> Fine-tuning)")
    
    # ------------------------------------------------------------------
    # 1. Data & Model Setup (기존과 동일)
    # ------------------------------------------------------------------
    train_proc = FeatureProcessor(USER_FEAT_PATH_PQ, ITEM_FEAT_PATH_PQ, SEQ_DATA_PATH_PQ, scaler=None)
    valid_proc = FeatureProcessor(USER_VAL_FEAT_PATH, ITEM_FEAT_PATH_PQ, SEQ_VAL_DATA_PATH, scaler=train_proc.user_scaler)
    valid_proc.item2id, valid_proc.item_ids = train_proc.item2id, train_proc.item_ids

    num_users = len(train_proc.user_ids) + 1
    num_items = len(train_proc.item_ids) + 1
    
    dummy_gnn_user = torch.zeros((num_users, 64))
    dummy_gnn_item = torch.zeros((num_items, 64))
    dummy_content  = torch.zeros((num_items, 128))

    # [초기화] Freeze=True 상태로 시작 (Warm-up 준비)
    model = HybridUserTower(
        num_users=num_users, 
        num_items=num_items, 
        gnn_user_init=dummy_gnn_user, 
        gnn_item_init=dummy_gnn_item,
        item_content_init=dummy_content
        
    ).to(DEVICE)
   
    # Injection (수술)
    model = load_and_align_embeddings(model, train_proc, model_dir=MODEL_DIR, device=DEVICE)
    model = load_and_align_gnn_items(model, train_proc, base_dir=BASE_DIR, device=DEVICE)
    model = load_and_align_gnn_user_embeddings(model, train_proc, base_dir=BASE_DIR, device=DEVICE)

    # ------------------------------------------------------------------
    # 2. Stage 1: Warm-up Setup (Epoch 1~2)
    # ------------------------------------------------------------------
    logger.log(1, "❄️ [Stage 1] Freezing Embeddings for Warm-up...")
    
    # 강제로 얼리기 (Safety Lock)
    model.gnn_user_emb.weight.requires_grad = False
    model.gnn_item_emb.weight.requires_grad = False
    model.item_content_emb.weight.requires_grad = False
    
    # 학습 가능한 파라미터만 골라냄 (Tower, Adapter, Fusion 등)
    warmup_params = filter(lambda p: p.requires_grad, model.parameters())
    
    optimizer = optim.AdamW(warmup_params, lr=5e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')
    item_probs = train_proc.get_logq_probs(DEVICE)
    
    train_loader = DataLoader(UserTowerDataset(train_proc, is_training=True), batch_size=BATCH_SIZE, shuffle=True, collate_fn=user_tower_collate_fn)
    

    
    # 전체 스케줄러 (10 Epoch 기준)
    total_steps = len(train_loader) * EPOCHS 
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_r100 = 0.0

    # ------------------------------------------------------------------
    # 3. Training Loop
    # ------------------------------------------------------------------
    for epoch in range(EPOCHS):
        
        # ==============================================================
        # [Stage 2 Switch] Epoch 3부터 Unfreeze & 차등 학습률 적용
        # ==============================================================
        if epoch == 2: # 0, 1 (Warmup) -> 2 (Fine-tuning Start)
            logger.log(1, "\n🔥 [Stage 2] Unfreezing Embeddings & Differential LR Start!")
            
            # 1. 녹이기 (Unfreeze)
            model.gnn_user_emb.weight.requires_grad = True
            model.gnn_item_emb.weight.requires_grad = True
            model.item_content_emb.weight.requires_grad = True
            
            # 2. 파라미터 그룹 분리
            tower_params = []
            embedding_params = []
            
            for name, param in model.named_parameters():
                if not param.requires_grad: continue
                if 'emb' in name:
                    embedding_params.append(param)
                else:
                    tower_params.append(param)
            
            # 3. Optimizer 재생성 (차등 학습률 적용)
            # Scheduler도 새로 연결해야 함
            optimizer = optim.AdamW([
                {'params': tower_params, 'lr': 5e-4},       # 뇌: 유지
                {'params': embedding_params, 'lr': 3e-5}    # 몸: 아주 천천히 (Fine-tuning)
            ], weight_decay=0.01)
            
            # 남은 Step에 맞춰 스케줄러 재설정
            remaining_steps = len(train_loader) * (EPOCHS - epoch)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=int(remaining_steps * 0.1), # 짧은 웜업
                num_training_steps=remaining_steps
            )
            
            # Scaler 상태는 유지하는 게 좋음 (선택사항)

        # ==============================================================
        # Standard Training Loop
        # ==============================================================
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            u_idx, u_dense, u_cat, activity_idx, seq_ids, seq_deltas, seq_mask, target_ids, _ = [x.to(DEVICE) for x in batch]
            optimizer.zero_grad() 
            with torch.amp.autocast('cuda'):
                user_seq_vecs,user_seq_only , gate_weights = model(
                    u_idx, seq_ids, seq_deltas, seq_mask, u_dense, u_cat, activity_idx
                )
                
                valid_mask = (target_ids != 0) 
                active_user_vecs = user_seq_vecs[valid_mask] 
                active_target_ids = target_ids[valid_mask]
                target_content = model.item_content_emb(active_target_ids)
                target_gnn = model.gnn_item_emb(active_target_ids)
                target_vecs = model.seq_adapter(target_content, target_gnn)
                
                active_item_vecs = F.normalize(target_vecs, p=2, dim=1)
                
                current_temp = model.get_current_temperature(6.67)

                loss_main = efficient_corrected_logq_loss(
                    active_user_vecs, active_item_vecs, active_target_ids, 
                    item_probs, temperature=current_temp, lambda_logq=LAMBDA_LOGQ 
                )
                    
                # 2. ⭐ Aux Loss (Sequence Only vs Item)
                # GNN/Meta 없이 맞추도록 강제
                # user_seq_only에서 valid_mask 적용 필요
                active_seq_only = user_seq_only[valid_mask]
                    
                    # LogQ 보정은 Main Loss에만 적용하거나 둘 다 적용해도 됨 (여기선 간단히 CrossEntropy만)
                    # 하지만 efficient_corrected_logq_loss를 재사용하는 것이 좋음 (lambda=0으로 두더라도)
                loss_aux = efficient_corrected_logq_loss(
                    active_seq_only, active_item_vecs, active_target_ids,
                    item_probs, temperature=current_temp, lambda_logq=0.0 # Aux는 LogQ 굳이 안 써도 됨
                )
                    
                    # 3. Final Loss
                loss = loss_main + (0.3 * loss_aux)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() 
            
            total_loss += loss.item()
            # LR 로깅 (그룹이 2개일 땐 첫 번째 그룹 LR 표시)
            curr_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{curr_lr:.2e}"})
            #del loss, user_seq_vecs, active_user_vecs, active_item_vecs

 

        #gc.collect()              # Python 쓰레기 수거
        #torch.cuda.empty_cache()  # GPU 캐시 반환 (공유 메모리도 같이 정리됨)
        avg_loss = total_loss / len(train_loader)
        logger.log(1, f"📊 Epoch {epoch+1} Result: Avg Loss {avg_loss:.4f}")

        # Evaluation
        metrics = evaluate_recall_multi_k(model, valid_proc, TARGET_VAL_PATH, k_list=[20, 100, 500], batch_size=256)

        gate_avg = gate_weights.item() 
        
        print(f"   🏛️  FiLM Gate Status:")
        print(f"      - Avg Gate Open Rate : {gate_avg:.4f}")
                        
        
        # Best Model Save
        if metrics['R@100'] > best_r100 and epoch >= 2:
            best_r100 = metrics['R@100']
            torch.save(model.state_dict(), SAVE_PATH_BEST)
            logger.log(1, f"🌟 New Best R@100: {best_r100:.4f} - Model Saved!")

        if epoch <= 1:
            SAVE_PATH_FREEZE = os.path.join(MODEL_DIR, "user_tower_phase3_freeze_Film.pth")
            torch.save(model.state_dict(), SAVE_PATH_FREEZE)
            logger.log(1, f"🌟 user tower freeze fin: {avg_loss:.4f} - Model Saved!")
if __name__ == "__main__":
    train_phase_3_FiLM()
