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

SAVE_PATH_BEST = os.path.join(MODEL_DIR, "user_tower_phase3_best_ft_0.19x.pth")

class SmartLogger:
    def __init__(self, verbosity=1): self.verbosity = verbosity
    def log(self, level, msg):
        if self.verbosity >= level: print(f"[{'ℹ️' if level==1 else '📊'}] {msg}")

logger = SmartLogger(verbosity=1)

# ==========================================
# 1. Feature Processor & Dataset
# ==========================================
class FeatureProcessor:
    def __init__(self, user_path, item_path, seq_path, scaler=None):
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
        
        

        if scaler is None: 
            scaled_data = self.user_scaler.fit_transform(self.users[self.u_dense_cols])
        else: 
            self.user_scaler = scaler
            scaled_data = self.user_scaler.transform(self.users[self.u_dense_cols])
        
        # NaN 방어
        self.users_scaled[self.u_dense_cols] = np.nan_to_num(scaled_data, nan=0.0)

    def get_user_tensor(self, user_id):
        dense = torch.tensor(self.users_scaled.loc[user_id, self.u_dense_cols].values, dtype=torch.float32)
        cat = torch.tensor(int(self.users_scaled.loc[user_id, 'preferred_channel']) - 1, dtype=torch.long)
        return dense, cat
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
        u_dense, u_cat = self.processor.get_user_tensor(u_id_str)
        
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
            'user_dense': u_dense, 'user_cat': u_cat,
            'seq_ids': torch.tensor(input_ids, dtype=torch.long),
            'seq_deltas': torch.tensor(input_deltas, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }

def user_tower_collate_fn(batch):
    u_idx = torch.stack([b['user_idx'] for b in batch])
    u_dense = torch.stack([b['user_dense'] for b in batch])
    u_cat = torch.stack([b['user_cat'] for b in batch])
    seq_ids = pad_sequence([b['seq_ids'] for b in batch], batch_first=True, padding_value=0)
    seq_deltas = pad_sequence([b['seq_deltas'] for b in batch], batch_first=True, padding_value=0)
    target_ids = pad_sequence([b['target_ids'] for b in batch], batch_first=True, padding_value=0)
    seq_mask = (seq_ids != 0).long()
    last_target = torch.tensor([b['target_ids'][-1] if len(b['target_ids']) > 0 else 0 for b in batch], dtype=torch.long)
    return u_idx, u_dense, u_cat, seq_ids, seq_deltas, seq_mask, target_ids, last_target

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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceCentricFusion(nn.Module):
    """
    [설계 철학]
    1. 경쟁(Softmax)을 제거합니다. Sequence는 무조건 1.0의 비중을 가집니다.
    2. GNN과 Meta는 Sequence 벡터를 Query로 사용하여, 
       Sequence가 '필요하다고 판단할 때만' 정보가 더해(Add)집니다.
    3. 초기에는 GNN/Meta 반영률을 0에 수렴하게 하여 Sequence 학습을 강제합니다.
    """
    def __init__(self, dim=128):
        super().__init__()
        
        # Sequence가 GNN/Meta를 얼마나 가져올지 결정하는 Gate
        # 입력: Sequence (Context)
        # 출력: 2 (GNN gate, Meta gate) -> Softmax 아님! Sigmoid 사용
        self.context_gate = nn.Sequential(
            nn.Linear(dim, 64),
            nn.GELU(),
            nn.Linear(64, 2), # [0]: GNN Gate, [1]: Meta Gate
            nn.Sigmoid()      # 0.0 ~ 1.0 독립적인 확률
        )
        
        # 차원 투영 (Projector)
        self.gnn_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )
        
        self.meta_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )
        
        # 최종 정리는 LayerNorm만 (MLP 통과 X -> 정보 희석 방지)
        self.final_ln = nn.LayerNorm(dim)

        # 🔥 [핵심 초기화]
        # Gate의 마지막 레이어 바이어스를 음수로 설정하여
        # 초기 Sigmoid 출력이 0에 가깝게 만듦 (예: -5 -> sigmoid(-5) ≈ 0.006)
        # 이렇게 하면 첫 Epoch에는 GNN/Meta가 거의 반영되지 않고 Sequence만 학습됨.
        nn.init.zeros_(self.context_gate[-2].weight)
        nn.init.constant_(self.context_gate[-2].bias, -5.0) 

    def forward(self, v_gnn, v_seq, v_meta):
        # 1. Gate 계산 (Sequence가 결정함)
        # gates: (Batch, Seq_Len, 2)
        gates = self.context_gate(v_seq)
        
        g_gnn = gates[..., 0:1]
        g_meta = gates[..., 1:2]
        
        # 2. Residual Addition (경쟁하지 않고 더하기만 함)
        # v_seq (Main) + (Gate * GNN) + (Gate * Meta)
        # Sequence는 계수가 1로 고정이므로 절대 무시되지 않음
        fused = v_seq + (g_gnn * self.gnn_proj(v_gnn)) + (g_meta * self.meta_proj(v_meta))
        
        # 3. Norm & Return
        # Gate 가중치도 리턴하여 로깅 (평균값)
        gnn_ratio = g_gnn.mean().item()
        meta_ratio = g_meta.mean().item()
        gate_weights = [gnn_ratio, meta_ratio]

        return self.final_ln(fused), gate_weights

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
        
        # 2. Adapters
        self.gnn_projector = nn.Sequential(
            nn.Linear(gnn_user_init.shape[1], 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(256, 128), nn.LayerNorm(128)
        )
        
        # [수정] ParallelAdapter 사용
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
        
        # 4. Meta & Fusion
        self.channel_emb = nn.Embedding(2, 32)
        self.meta_mlp = nn.Sequential(
            nn.Linear(35, 128), nn.GELU(),  # Target Layer Monitoring
            nn.Linear(128, 128), nn.LayerNorm(128)
        )
        self.fusion_layer = SequenceCentricFusion(dim=128)
        
        
        
        
    def get_current_temperature(self, clamp_min):
        # 사용할 때는 exp를 취해서 양수로 만듦
        # 1 / exp(scale) = temperature
        # 하지만 보통 계산 효율을 위해 (Cosine Sim * Scale) 방식으로 곱해버림
        # 여기서는 기존 Loss 함수와의 호환성을 위해 Temperature 값으로 변환해서 리턴
        
        # logit_scale을 최대 100(exp(4.6))까지만 커지게 제한 (CLIP 논문 테크닉 - 발산 방지)
        scale = self.logit_scale.exp().clamp(clamp_min, max=100.0)
        
        #clamp(min=14.3)
        # Scale = 1 / Temperature 이므로,
        # Temperature = 1 / Scale
        return 1.0 / scale
    
    def forward(self, u_idx, seq_ids, seq_deltas, seq_mask, u_dense, u_cat):
        B, L = seq_ids.shape
        
        # 1. GNN User
        v_gnn = self.gnn_projector(self.gnn_user_emb(u_idx))
        v_gnn_seq = F.normalize(v_gnn, p=2, dim=1).unsqueeze(1).expand(-1, L, -1)
        v_gnn_seq = torch.zeros_like(v_gnn_seq)
        if self.training:
            drop_prob = 0.4  # 40% 확률로 GNN을 버림
            keep_prob = 1 - drop_prob
            
            # 배치별 마스크 생성 (B, 1, 1)
            mask = torch.bernoulli(torch.full((B, 1, 1), keep_prob, device=v_gnn_seq.device))
            
            # Inverted Dropout: 살아남은 신호는 keep_prob로 나눠서 스케일 유지
            v_gnn_seq = (v_gnn_seq * mask) / keep_prob
        
        # =========================================================
        # [수정된 부분] 2. Dual-View Sequence (Parallel Adapter)
        # =========================================================
        # (1) 임베딩 꺼내기
        raw_content = self.item_content_emb(seq_ids) # (B, L, 128)
        raw_gnn = self.gnn_item_emb(seq_ids)         # (B, L, 64)
        
        # (2) Adapter 통과 (인자 2개 전달!)
        # 기존에는 cat으로 합쳐서 넣었지만, 이제는 따로 넣어야 합니다.
        seq_input = self.seq_adapter(raw_content, raw_gnn) # <--- 여기가 수정됨!
        
        # (3) Time Embedding
        seq_input = seq_input  * math.sqrt(self.embed_dim) + self.time_emb(seq_deltas.clamp(max=1000))
        
        # =========================================================
        
        causal_mask = torch.triu(torch.ones(L, L, device=seq_ids.device) * float('-inf'), diagonal=1)
        key_padding_mask = (seq_mask == 0)
        
        seq_out = self.seq_encoder(seq_input, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        v_seq = F.normalize(seq_out, p=2, dim=2)

        cat_vec = self.channel_emb(u_cat)
        v_meta = self.meta_mlp(torch.cat([u_dense, cat_vec], dim=1))
        v_meta_seq = F.normalize(v_meta, p=2, dim=1).unsqueeze(1).expand(-1, L, -1)
        
        output, gate_weights = self.fusion_layer(v_gnn_seq, v_seq, v_meta_seq)
        output = F.normalize(output, p=2, dim=2)
        return output, v_seq, gate_weights
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



class EnsembleGate(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        # 아주 가벼운 2층 MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() # 결과는 무조건 0~1 사이 (Alpha)
        )
        
    def forward(self, seq_len, u_dense):
        # seq_len 정규화 (대략 100으로 나눔)
        len_feat = seq_len.unsqueeze(1).float() / 100.0
        
        # 입력 벡터 결합: [길이, 유저정보1, 유저정보2, ...]
        # u_dense는 이미 log/scale 되어 있다고 가정
        features = torch.cat([len_feat, u_dense], dim=1) 
        
        # Alpha 예측
        alpha = self.mlp(features)
        return alpha

def save_gate_model(gate_model, save_dir, filename="gate_model_best.pth"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)
    torch.save(gate_model.state_dict(), save_path)
    print(f"💾 Gate Model Saved: {save_path}")

def load_gate_model(save_dir, device, filename="gate_model_best.pth"):
    save_path = os.path.join(save_dir, filename)
    
    # 모델 초기화
    gate_model = EnsembleGate().to(device)
    
    if os.path.exists(save_path):
        gate_model.load_state_dict(torch.load(save_path, map_location=device))
        gate_model.eval() # 평가는 무조건 eval 모드
        print(f"📂 Gate Model Loaded from: {save_path}")
        return gate_model
    else:
        print(f"⚠️ Warning: No Gate model found at {save_path}. Initializing Randomly.")
        return gate_model





def train_gate_only(
    gate_model, 
    seq_model, 
    processor, 
    gnn_user_matrix, 
    gnn_item_matrix,
    train_loader, # 학습 데이터 로더
    epochs=3
):
    print("\n🚀 Training Ensemble Gate (Freezing Base Models)...")
    
    # 1. 기존 모델 Freeze (절대 건드리지 않음)
    seq_model.eval()
    for param in seq_model.parameters():
        param.requires_grad = False
        
    # 2. Gate 모델만 학습
    gate_model.train()
    optimizer = torch.optim.Adam(gate_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss() # 혹은 Contrastive Loss
    
    # GNN/Seq Item Vector 미리 계산 (고정값)
    with torch.no_grad():
        all_item_ids = torch.arange(1, len(processor.item_ids)+1).to(DEVICE)
        
        # (Seq Item Vec)
        seq_item_vecs = []
        for i in range(0, len(all_item_ids), 4096):
            chunk = all_item_ids[i:i+4096]
            c_vec = seq_model.seq_adapter(
                seq_model.item_content_emb(chunk), 
                seq_model.gnn_item_emb(chunk)
            )
            seq_item_vecs.append(F.normalize(c_vec, p=2, dim=1))
        all_seq_vecs = torch.cat(seq_item_vecs, dim=0)
        
        # (GNN Item Vec)
        all_gnn_vecs = F.normalize(gnn_item_matrix[1:].to(DEVICE), p=2, dim=1)

    # 3. 학습 루프
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Gate Epoch {epoch+1}")
        
        for batch in pbar:
            u_idx, u_dense, u_cat, seq_ids, seq_deltas, seq_mask, target_ids, _ = [x.to(DEVICE) for x in batch]
            
            with torch.no_grad():
                # (A) Seq Score 계산
                # Seq Only 모드이므로 output만 받음
                output = seq_model(u_idx, seq_ids, seq_deltas, seq_mask, u_dense, u_cat)
                if isinstance(output, tuple): output = output[0]
                
                lengths = seq_mask.sum(dim=1)
                last_indices = (lengths - 1).clamp(min=0)
                seq_user = output[torch.arange(len(u_idx)), last_indices]
                
                # 정답 아이템(Pos)에 대한 점수만 계산 (효율성 위해)
                # 실제 학습 땐 Negative Sampling 필요하지만 여기선 간략화
                # 전체 아이템과의 내적 (Batch, Num_Items)
                scores_seq = torch.matmul(seq_user, all_seq_vecs.T)
                
                # (B) GNN Score 계산
                gnn_user = F.normalize(gnn_user_matrix[u_idx], p=2, dim=1)
                scores_gnn = torch.matmul(gnn_user, all_gnn_vecs.T)
            
            # --- 여기부터 Gradient 흐름 ---
            
            # (C) Gate가 Alpha 결정
            # u_dense: (Batch, 3) 가정 -> input_dim = 1 + 3 = 4
            alpha = gate_model(lengths, u_dense) # (Batch, 1)
            
            # (D) 점수 합성
            final_scores = alpha * scores_seq + (1 - alpha) * scores_gnn
            
            # (E) Loss 계산 (Cross Entropy)
            # target_ids의 마지막 아이템(Next Item)을 맞추도록 유도
            # target_ids: (Batch, Seq_Len) -> last item extraction needed
            # 편의상 loader가 last_target을 준다고 가정
            last_target = target_ids[:, -1] - 1 # 0-base index
            
            loss = criterion(final_scores / 0.1, last_target) # Temp=0.1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1), 'avg_alpha': alpha.mean().item()})
            save_gate_model(gate_model, MODEL_DIR, "gate_model_best.pth")
    return gate_model







def main_evaluation_flow():
    # 1. 모델 초기화 (Dummy로 시작)
    train_proc = FeatureProcessor(USER_FEAT_PATH_PQ, ITEM_FEAT_PATH_PQ, SEQ_DATA_PATH_PQ, scaler=None)
    valid_proc = FeatureProcessor(USER_VAL_FEAT_PATH, ITEM_FEAT_PATH_PQ, SEQ_VAL_DATA_PATH, scaler=train_proc.user_scaler)
    valid_proc.item2id, valid_proc.item_ids = train_proc.item2id, train_proc.item_ids

    num_users = len(train_proc.user_ids) + 1
    num_items = len(train_proc.item_ids) + 1
    
    # Dummy Init

    model = HybridUserTower(
        num_users=len(train_proc.user_ids)+1,
        num_items=len(train_proc.item_ids)+1,
        gnn_user_init=torch.zeros((len(train_proc.user_ids)+1, 64)),
        gnn_item_init=torch.zeros((len(train_proc.item_ids)+1, 64)),
        item_content_init=torch.zeros((len(train_proc.item_ids)+1, 128))
    ).to(DEVICE)

    # 2. 임베딩 로드 및 정렬 (기존 함수 재사용) ✅
    # 이 과정에서 Pretrained ID -> Current ID 매핑이 완료됨
    model = load_and_align_embeddings(model, train_proc, MODEL_DIR, DEVICE)     # Content
    model = load_and_align_gnn_items(model, train_proc, BASE_DIR, DEVICE)       # GNN Item
    model = load_and_align_gnn_user_embeddings(model, train_proc, BASE_DIR, DEVICE) # GNN User

    # 3. 모델 가중치 로드 (Sequence 학습된 모델)
    # GNN/Content 임베딩은 위에서 로드했으므로, 학습된 Tower Weight만 덮어씌움 (strict=False 권장)
    # 만약 저장된 pth에 임베딩까지 다 들어있다면 load_state_dict 한방이면 됨
    if os.path.exists(SAVE_PATH_BEST):
        model.load_state_dict(torch.load(SAVE_PATH_BEST), strict=False)
        print("✅ Trained Model Weights Loaded.")

    # 4. GNN 매트릭스 추출 (앙상블용) ⭐
    # 모델 안에 Align 되어 들어있는 가중치를 복사해서 꺼냄
    # .data.clone()을 해야 안전함
    gnn_user_matrix = model.gnn_user_emb.weight.data.clone().detach()
    gnn_item_matrix = model.gnn_item_emb.weight.data.clone().detach()

    # 5. 앙상블 평가 실행
    print("\n🧪 Starting Hybrid Ensemble Evaluation...")
    
    # Alpha 값을 조정해가며 최적점 찾기
# 기존 for loop가 끝난 뒤 호출

# 실행
# main_evaluation_flow()




def evaluate_hybrid_with_trained_gate(
    seq_model,       # 학습된 Sequence Model
    gate_model,      # 학습된 Gate Model (Load 된 것)
    processor, 
    target_df_path, 
    gnn_user_matrix, 
    gnn_item_matrix, 
    k_list=[20, 100, 500], 
    batch_size=256
):
    print("\n🤖 Starting AI-Gated Ensemble Evaluation...")
    seq_model.eval()
    gate_model.eval() # Gate 모델도 평가 모드 필수!
    
    target_df = pd.read_parquet(target_df_path)
    target_dict = target_df.set_index('customer_id')['target_ids'].to_dict()
    
    val_loader = DataLoader(
        UserTowerDataset(processor, is_training=False), 
        batch_size=batch_size, shuffle=False, collate_fn=user_tower_collate_fn
    )
    
    # [Pre-computation: Item Vectors] ------------------------------
    with torch.no_grad():
        all_item_ids = torch.arange(1, len(processor.item_ids)+1).to(DEVICE)
        
        # 1. Seq Item Vecs
        seq_item_vecs_list = []
        for i in range(0, len(all_item_ids), 4096):
            chunk = all_item_ids[i:i+4096]
            c_vec = seq_model.seq_adapter(
                seq_model.item_content_emb(chunk), seq_model.gnn_item_emb(chunk)
            )
            seq_item_vecs_list.append(F.normalize(c_vec, p=2, dim=1))
        all_seq_item_vecs = torch.cat(seq_item_vecs_list, dim=0)

        # 2. GNN Item Vecs
        all_gnn_item_vecs = F.normalize(gnn_item_matrix[1:].to(DEVICE), p=2, dim=1) 
    # --------------------------------------------------------------

    hit_counts = {k: 0 for k in k_list}
    total_users = 0
    total_alpha_sum = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="🤖 AI Gating..."):
            u_idx, u_dense, u_cat, seq_ids, seq_deltas, seq_mask, _, _ = [x.to(DEVICE) for x in batch]
            
            # 유효 유저 필터링
            batch_uids = [processor.user_ids[i-1] for i in u_idx.cpu().numpy()]
            valid_idx_list = [i for i, uid in enumerate(batch_uids) if uid in target_dict]
            if not valid_idx_list: continue
            v_idx = torch.tensor(valid_idx_list).to(DEVICE)

            # ------------------------------------------------------
            # [A] Calculate Scores
            # ------------------------------------------------------
            # 1. Seq Score
            output = seq_model(
                u_idx[v_idx], seq_ids[v_idx], seq_deltas[v_idx], seq_mask[v_idx], u_dense[v_idx], u_cat[v_idx]
            )
            if isinstance(output, tuple): output = output[0]
            
            lengths = seq_mask[v_idx].sum(dim=1)
            last_indices = (lengths - 1).clamp(min=0)
            user_seq_vecs = output[torch.arange(len(v_idx)), last_indices]
            scores_seq = torch.matmul(user_seq_vecs, all_seq_item_vecs.T)

            # 2. GNN Score
            user_gnn_vecs = F.normalize(gnn_user_matrix[u_idx[v_idx]].to(DEVICE), p=2, dim=1)
            scores_gnn = torch.matmul(user_gnn_vecs, all_gnn_item_vecs.T)

            # ------------------------------------------------------
            # [B] Apply Trained Gate Model ⭐
            # ------------------------------------------------------
            # 입력: (시퀀스 길이, 유저 덴스 피처) -> 출력: Alpha (Batch, 1)
            # Gate가 "이 유저는 0.7만큼 Seq를 믿어라"라고 판단함
            alpha_tensor = gate_model(lengths, u_dense[v_idx]) 
            
            total_alpha_sum += alpha_tensor.mean().item() * len(v_idx)

            # ------------------------------------------------------
            # [C] Weighted Fusion
            # ------------------------------------------------------
            # alpha_tensor는 이미 (Batch, 1) 모양이므로 바로 브로드캐스팅 곱셈 가능
            final_scores = (alpha_tensor * scores_seq) + ((1.0 - alpha_tensor) * scores_gnn)
            
            # Top-K Counting
            _, topk_indices = torch.topk(final_scores, k=max(k_list), dim=1)
            pred_ids = (topk_indices + 1).cpu().numpy()
            
            for i, original_idx in enumerate(valid_idx_list):
                u_id = batch_uids[original_idx]
                actual_indices = set(processor.item2id[tid] for tid in target_dict[u_id] if tid in processor.item2id)
                if not actual_indices: continue
                for k in k_list:
                    if not actual_indices.isdisjoint(pred_ids[i, :k]): hit_counts[k] += 1
                total_users += 1

    avg_alpha = total_alpha_sum / total_users if total_users > 0 else 0
    metrics = {f"R@{k}": (hit_counts[k] / total_users if total_users > 0 else 0.0) for k in k_list}
    
    print(f"\n📊 [AI-Gated Result]")
    print(f"   - Avg Alpha Predicted: {avg_alpha:.4f}")
    print(f"   - Metrics: {metrics}")
    return metrics



def main_with_trained_gate():
    # 1. 기본 모델 & 데이터 로드 (기존과 동일)
    # ... (seq_model, gnn_matrix 등 로드 완료 가정) ...
    
    # [가정] 이미 gate_model 학습을 완료하고 저장했다고 가정
    # 예: train_gate_only(...) -> save_gate_model(...) 실행 완료
    
    train_proc = FeatureProcessor(USER_FEAT_PATH_PQ, ITEM_FEAT_PATH_PQ, SEQ_DATA_PATH_PQ, scaler=None)
    valid_proc = FeatureProcessor(USER_VAL_FEAT_PATH, ITEM_FEAT_PATH_PQ, SEQ_VAL_DATA_PATH, scaler=train_proc.user_scaler)
    valid_proc.item2id, valid_proc.item_ids = train_proc.item2id, train_proc.item_ids

    num_users = len(train_proc.user_ids) + 1
    num_items = len(train_proc.item_ids) + 1
    
    # Dummy Init

    model = HybridUserTower(
        num_users=len(train_proc.user_ids)+1,
        num_items=len(train_proc.item_ids)+1,
        gnn_user_init=torch.zeros((len(train_proc.user_ids)+1, 64)),
        gnn_item_init=torch.zeros((len(train_proc.item_ids)+1, 64)),
        item_content_init=torch.zeros((len(train_proc.item_ids)+1, 128))
    ).to(DEVICE)

    # 2. 임베딩 로드 및 정렬 (기존 함수 재사용) ✅
    # 이 과정에서 Pretrained ID -> Current ID 매핑이 완료됨
    model = load_and_align_embeddings(model, train_proc, MODEL_DIR, DEVICE)     # Content
    model = load_and_align_gnn_items(model, train_proc, BASE_DIR, DEVICE)       # GNN Item
    model = load_and_align_gnn_user_embeddings(model, train_proc, BASE_DIR, DEVICE) # GNN User

    # 3. 모델 가중치 로드 (Sequence 학습된 모델)
    # GNN/Content 임베딩은 위에서 로드했으므로, 학습된 Tower Weight만 덮어씌움 (strict=False 권장)
    # 만약 저장된 pth에 임베딩까지 다 들어있다면 load_state_dict 한방이면 됨
    if os.path.exists(SAVE_PATH_BEST):
        model.load_state_dict(torch.load(SAVE_PATH_BEST), strict=False)
        print("✅ Trained Model Weights Loaded.")

    # 4. GNN 매트릭스 추출 (앙상블용) ⭐
    # 모델 안에 Align 되어 들어있는 가중치를 복사해서 꺼냄
    # .data.clone()을 해야 안전함
    gnn_user_matrix = model.gnn_user_emb.weight.data.clone().detach()
    gnn_item_matrix = model.gnn_item_emb.weight.data.clone().detach()

    # 5. 앙상블 평가 실행
    print("\n🧪 Starting Hybrid Ensemble Evaluation...")
       
    
    
    
    
    
    
    # 2. Gate 모델 불러오기 ⭐
    # 저장된 경로에서 학습된 가중치를 로드합니다.
    trained_gate = load_gate_model(
        save_dir=MODEL_DIR, 
        device=DEVICE, 
        filename="gate_model_best.pth"
    )
    
    # 3. 평가 실행
    evaluate_hybrid_with_trained_gate(
        seq_model=model,        # Freeze된 Seq 모델
        gate_model=trained_gate,# Load된 Gate 모델
        processor=valid_proc,
        target_df_path=TARGET_VAL_PATH,
        gnn_user_matrix=gnn_user_matrix,
        gnn_item_matrix=gnn_item_matrix
    )





if __name__ == "__main__":
   main_with_trained_gate()