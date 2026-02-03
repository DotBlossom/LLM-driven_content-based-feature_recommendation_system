import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# 0. Global Configuration
# -------------------------------------------------------------------------
EMBED_DIM = 128
MAX_SEQ_LEN = 50
DROPOUT = 0.1
GNN_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 경로 설정 (preprocess_final.py 결과물 위치)
BASE_DIR = r"D:\trainDataset\localprops"
MODEL_DIR = os.path.join(BASE_DIR, "models")

ITEM_FEAT_PATH_PQ = os.path.join(BASE_DIR, "features_item.parquet")
USER_FEAT_PATH_PQ = os.path.join(BASE_DIR, "features_user.parquet")
SEQ_DATA_PATH_PQ = os.path.join(BASE_DIR, "features_sequence.parquet")

# Pre-trained Weights Paths
GNN_PATH = os.path.join(MODEL_DIR, "1simgcl_trained.pth")
ITEM_MATRIX_PATH = os.path.join(MODEL_DIR, "1pretrained_item_matrix.pt")

# -------------------------------------------------------------------------
# 1. Feature Processor & Dataset (Data Loading)
# -------------------------------------------------------------------------
class FeatureProcessor:
    def __init__(self, user_path, item_path, seq_path):
        print("🔄 Loading & Scaling Features...")
        
        # Load Parquet
        self.users = pd.read_parquet(user_path).set_index('customer_id')
        self.items = pd.read_parquet(item_path).set_index('article_id')
        self.seqs = pd.read_parquet(seq_path).set_index('customer_id')
        
        # User ID Mapping (String -> Int for Embedding Lookup)
        self.user_ids = self.users.index.tolist()
        self.user2id = {uid: i for i, uid in enumerate(self.user_ids)}
        
        # Item ID Mapping (String -> Int)
        self.item_ids = self.items.index.tolist()
        self.item2id = {iid: i for i, iid in enumerate(self.item_ids)}
        
        # Scalers
        self.user_scaler = StandardScaler()
        self.u_dense_cols = ['user_avg_price_log', 'total_cnt_log', 'recency_log']
        
        # Scaling Apply
        self.users_scaled = self.users.copy()
        self.users_scaled[self.u_dense_cols] = self.user_scaler.fit_transform(self.users[self.u_dense_cols])
        
        print("✅ Features processed successfully.")

    def get_user_tensor(self, user_id):
        # Dense Features (3 dims)
        dense_vals = self.users_scaled.loc[user_id, self.u_dense_cols].values
        dense = torch.tensor(dense_vals, dtype=torch.float32)
        
        # Cat Feature (Preferred Channel: 1,2 -> 0,1)
        cat_val = self.users_scaled.loc[user_id, 'preferred_channel']
        cat = torch.tensor(int(cat_val) - 1, dtype=torch.long)
        
        return dense, cat

    def get_logq_probs(self, device):
        """ LogQ Correction을 위한 Log Probability Tensor 생성 """
        # raw_probability 컬럼을 가져와서 item2id 순서대로 정렬
        probs = np.zeros(len(self.item_ids), dtype=np.float32)
        
        # article_id가 인덱스이므로 순회하며 매핑
        # (더 빠른 방법: reindex 사용)
        sorted_probs = self.items['raw_probability'].reindex(self.item_ids).fillna(0).values
        
        return torch.tensor(sorted_probs, dtype=torch.float32).to(device)

class UserTowerDataset(Dataset):
    def __init__(self, processor, max_seq_len=50):
        self.processor = processor
        self.user_ids = processor.user_ids # List of Strings
        self.max_len = max_seq_len
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        u_id_str = self.user_ids[idx]
        u_idx_int = idx # processor.user_ids 순서 그대로 사용
        
        # 1. User Features
        u_dense, u_cat = self.processor.get_user_tensor(u_id_str)
        
        # 2. Sequence Data
        seq_ids = []
        seq_deltas = []
        
        if u_id_str in self.processor.seqs.index:
            seq_row = self.processor.seqs.loc[u_id_str]
            # String Item IDs -> Integer IDs 변환
            raw_seq_ids = seq_row['sequence_ids'][-self.max_len:]
            seq_ids = [self.processor.item2id.get(i, 0) for i in raw_seq_ids]
            seq_deltas = seq_row['sequence_deltas'][-self.max_len:]
        
        return {
            'user_idx': torch.tensor(u_idx_int, dtype=torch.long),
            'user_dense': u_dense,       # (3,)
            'user_cat': u_cat,           # (1,)
            'seq_ids': torch.tensor(seq_ids, dtype=torch.long),
            'seq_deltas': torch.tensor(seq_deltas, dtype=torch.long)
        }

def user_tower_collate_fn(batch):
    user_idx = torch.stack([b['user_idx'] for b in batch])
    user_dense = torch.stack([b['user_dense'] for b in batch])
    user_cat = torch.stack([b['user_cat'] for b in batch])
    
    # Padding (Padding Value = 0)
    seq_ids = pad_sequence([b['seq_ids'] for b in batch], batch_first=True, padding_value=0)
    seq_deltas = pad_sequence([b['seq_deltas'] for b in batch], batch_first=True, padding_value=0)
    
    # Mask (0 is padding)
    seq_mask = (seq_ids != 0).long()
    
    return user_idx, user_dense, user_cat, seq_ids, seq_deltas, seq_mask

# -------------------------------------------------------------------------
# 2. Model Architecture (Updated)
# -------------------------------------------------------------------------
class ContextGatingFusion(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        # 입력: GNN(128) + Seq(128) + Meta(128) = 384
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim * 3), 
            nn.ReLU(),
            nn.Linear(dim * 3, dim * 3), 
            nn.Sigmoid() 
        )
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, v_gnn, v_seq, v_meta):
        combined = torch.cat([v_gnn, v_seq, v_meta], dim=1) 
        all_gates = self.gate_mlp(combined) 
        
        dim = v_gnn.shape[1]
        g_gnn = all_gates[:, :dim]
        g_seq = all_gates[:, dim:2*dim]
        g_meta = all_gates[:, 2*dim:]
        
        v_fused = (v_gnn * g_gnn) + (v_seq * g_seq) + (v_meta * g_meta)
        return self.output_proj(v_fused)

class HybridUserTower(nn.Module):
    def __init__(self, 
                 num_users, 
                 num_items,
                 pretrained_gnn_embeddings=None, 
                 pretrained_item_vectors=None):
        super().__init__()
        
        # A. GNN Part
        if pretrained_gnn_embeddings is not None:
            self.gnn_user_emb = nn.Embedding.from_pretrained(pretrained_gnn_embeddings, freeze=False)
            current_gnn_dim = pretrained_gnn_embeddings.shape[1]
        else:
            current_gnn_dim = GNN_DIM
            self.gnn_user_emb = nn.Embedding(num_users, current_gnn_dim)
            nn.init.xavier_normal_(self.gnn_user_emb.weight)

        self.gnn_projector = nn.Sequential(
            nn.Linear(current_gnn_dim, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
            nn.GELU()
        )

        # B. Sequential Part
        if pretrained_item_vectors is not None:
            # freeze=True for Transfer Learning Stability
            self.item_content_emb = nn.Embedding.from_pretrained(pretrained_item_vectors, freeze=True)
        else:
            self.item_content_emb = nn.Embedding(num_items, EMBED_DIM)

        # Time Embedding (Bucketized)
        # 0~1000일까지의 Time Delta를 임베딩
        self.time_emb = nn.Embedding(1001, EMBED_DIM) 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=4, dim_feedforward=EMBED_DIM*4, dropout=DROPOUT, batch_first=True, norm_first=True
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # C. Meta Part (Updated)
        # Input: Dense(3) + Cat Emb(128)
        self.channel_emb = nn.Embedding(2, 32) # Channel 0 or 1 -> 32 dim
        
        # 3 (Dense) + 32 (Cat) = 35
        self.meta_mlp = nn.Sequential(
            nn.Linear(3 + 32, EMBED_DIM),
            nn.GELU(),
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM)
        )

        # D. Fusion
        self.fusion_layer = ContextGatingFusion(dim=EMBED_DIM)

    def forward(self, user_indices, seq_ids, seq_deltas, seq_mask, user_dense, user_cat):
        # 1. GNN Representation
        v_gnn = self.gnn_projector(self.gnn_user_emb(user_indices))

        # 2. Sequential Representation
        seq_emb = self.item_content_emb(seq_ids)
        
        # Time Embedding (Bucketize: 1000일 넘어가면 1000으로 클리핑)
        deltas = seq_deltas.clamp(max=1000)
        time_emb = self.time_emb(deltas)
        
        seq_input = seq_emb + time_emb
        
        # Transformer
        key_padding_mask = (seq_mask == 0)
        seq_out = self.seq_encoder(seq_input, src_key_padding_mask=key_padding_mask)
        
        # Attention Pooling (Last Valid Token or Mean)
        # 여기서는 간단히 Masked Mean Pooling
        mask_expanded = seq_mask.unsqueeze(-1)
        sum_seq = (seq_out * mask_expanded).sum(dim=1)
        cnt_seq = mask_expanded.sum(dim=1).clamp(min=1e-9)
        v_seq = sum_seq / cnt_seq

        # 3. Meta Representation
        cat_vec = self.channel_emb(user_cat) # (B, 32)
        meta_input = torch.cat([user_dense, cat_vec], dim=1) # (B, 35)
        v_meta = self.meta_mlp(meta_input)

        # 4. Fusion
        output = self.fusion_layer(v_gnn, v_seq, v_meta)
        
        return F.normalize(output, p=2, dim=1)

# -------------------------------------------------------------------------
# 3. LogQ Loss Function
# -------------------------------------------------------------------------
def logq_correction_loss(user_emb, item_emb, pos_item_ids, item_log_probs, temperature=0.07, lambda_logq=0.5):
    """
    LogQ Correction을 적용한 Contrastive Loss (Sampled Softmax)
    """
    # 1. Batch 내의 Positive/Negative Score 계산 (In-batch Negative)
    # user_emb: (B, Dim)
    # item_emb: (B, Dim) -> 여기서 item_emb는 Batch 내 유저들이 다음 시점에 구매한 '정답 아이템'들의 임베딩
    
    # logits: (B, B) -> 대각선이 정답(Positive), 나머지는 Negative
    logits = torch.matmul(user_emb, item_emb.T) 
    logits = logits / temperature
    
    # 2. LogQ Correction
    # 배치에 포함된 아이템들의 인기도 확률 가져오기
    # pos_item_ids: (B,)
    batch_log_probs = torch.log(item_log_probs[pos_item_ids] + 1e-9) # (B,)
    
    # Correction: logits에서 log(P)를 뺌
    # 인기 아이템일수록 P가 크고 log(P)가 큼 -> Logits가 많이 깎임 (페널티)
    # Broadcasting: (1, B) 형태로 빼줌 (각 아이템(Column)에 대해 보정)
    correction = batch_log_probs.unsqueeze(0) 
    
    corrected_logits = logits - (lambda_logq * correction)
    
    # 3. Cross Entropy
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss = F.cross_entropy(corrected_logits, labels)
    
    return loss

# -------------------------------------------------------------------------
# 4. Main Training Routine
# -------------------------------------------------------------------------
def train_user_tower():
    # 1. Load Data & Processor
    processor = FeatureProcessor(USER_FEAT_PATH_PQ, ITEM_FEAT_PATH_PQ, SEQ_DATA_PATH_PQ)
    
    dataset = UserTowerDataset(processor, MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=user_tower_collate_fn, num_workers=0)

    # 2. Load Assets (GNN, Item Matrix)
    # 실제로는 load_pretrained_assets() 함수 사용. 여기서는 가정.
    # item_tensor: (Num_Items, 128)
    item_tensor = torch.load(ITEM_MATRIX_PATH, map_location='cpu') if os.path.exists(ITEM_MATRIX_PATH) else torch.randn(len(processor.item_ids), 128)
    
    # 3. Model Init
    model = HybridUserTower(
        num_users=len(processor.user_ids),
        num_items=len(processor.item_ids),
        pretrained_gnn_embeddings=None, # GNN 파일 있으면 로드해서 넣기
        pretrained_item_vectors=item_tensor
    ).to(DEVICE)

    # 4. Setup Optimization
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # LogQ용 확률 텐서 로드
    item_log_probs = processor.get_logq_probs(DEVICE)
    
    # Target Item Lookup (정답 비교용, Gradient X)
    target_lookup = nn.Embedding.from_pretrained(item_tensor, freeze=True).to(DEVICE)

    print(f"\n🚀 Start Training User Tower (LogQ Corrected)...")
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            # Unpack Batch
            u_idx, u_dense, u_cat, seq_ids, seq_deltas, seq_mask = [x.to(DEVICE) for x in batch]
            
            optimizer.zero_grad()
            
            # Forward
            user_vec = model(u_idx, seq_ids, seq_deltas, seq_mask, u_dense, u_cat)
            
            # Target (Next Item Prediction) - 여기서는 시퀀스의 마지막 다음 아이템을 예측해야 함.
            # 하지만 현재 데이터셋엔 'Target Item'이 명시적으로 없음 (User Seq만 있음).
            # [수정] 학습을 위해선 '시퀀스의 마지막 아이템'을 정답(Target)으로 쓰고, 
            # 입력을 '마지막 제외(t-1)'까지로 하는 Self-Supervised 방식을 쓰거나,
            # 데이터셋에 Target Item 컬럼이 있어야 함.
            
            # 여기서는 편의상 시퀀스의 '마지막 아이템'을 정답으로 간주하고, 입력에서 마스킹하는 방식 사용
            # 실제로는 Dataset __getitem__ 에서 target_item_id를 뱉어주는 게 맞음.
            # 코드가 길어지므로, 현재 배치의 seq_ids의 마지막 값을 Target으로 가정합니다.
            
            # Target: 시퀀스의 실제 마지막 아이템
            # (주의: 패딩이 0이므로 0이 아닌 마지막 값을 찾아야 함. 간단히 seq_ids의 첫번째(최근)가 마지막이라 가정)
            # preprocess 로직상 seq_ids[-1]이 가장 최근임.
            
            target_item_ids = seq_ids[:, -1] # 가장 최근 아이템
            target_item_vec = target_lookup(target_item_ids) # (B, 128)
            target_item_vec = F.normalize(target_item_vec, p=2, dim=1)
            
            # Loss Calculation (LogQ)
            loss = logq_correction_loss(
                user_vec, target_item_vec, target_item_ids, item_log_probs, lambda_logq=0.5
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"✅ Epoch {epoch+1} Done. Avg Loss: {total_loss/len(dataloader):.4f}")
        
    # Save
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "user_tower_logq.pth"))

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_user_tower()








'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# -------------------------------------------------------------------------
# 0. Global Configuration
# -------------------------------------------------------------------------
EMBED_DIM = 128          # Item Tower와 동일하게 맞춤
MAX_SEQ_LEN = 50         # 유저의 구매 이력 최대 길이
NUM_META_FEATURES = 4    # 나이, 성별 등
DROPOUT = 0.1
GNN_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 파일 경로 설정 (현재 파일 기준 relative path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

GNN_PATH = os.path.join(MODEL_DIR, "1simgcl_trained.pth")
ITEM_MATRIX_PATH = os.path.join(MODEL_DIR, "1pretrained_item_matrix.pt")

# -------------------------------------------------------------------------
# 1. Modules (Fusion & User Tower)
# -------------------------------------------------------------------------

class ContextGatingFusion(nn.Module):
    """
    [SE-Block Style Fusion]
    GNN(Global), Seq(Current), Meta(Static) 3가지 신호를 
    상황에 맞게 동적으로 섞는 모듈
    """
    def __init__(self, dim=128):
        super().__init__()
        # 입력: 3개 벡터 연결 (128 * 3) -> 출력: 3개 게이트
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim * 3), 
            nn.ReLU(),
            nn.Linear(dim * 3, dim * 3), 
            nn.Sigmoid() 
        )
        # 최종 정제
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, v_gnn, v_seq, v_meta):
        # 1. Concatenation (B, 384)
        combined = torch.cat([v_gnn, v_seq, v_meta], dim=1) 
        
        # 2. Calculate Channel-wise Gates
        all_gates = self.gate_mlp(combined) 
        
        # 3. Split Gates
        dim = v_gnn.shape[1]
        g_gnn = all_gates[:, :dim]
        g_seq = all_gates[:, dim:2*dim]
        g_meta = all_gates[:, 2*dim:]
        
        # 4. Gated Sum (Element-wise Multiplication)
        v_fused = (v_gnn * g_gnn) + (v_seq * g_seq) + (v_meta * g_meta)
        
        # 5. Final Projection
        return self.output_proj(v_fused)

class HybridUserTower(nn.Module):
    def __init__(self, 
                 num_users: int, 
                 pretrained_gnn_embeddings: torch.Tensor = None, 
                 pretrained_item_vectors: torch.Tensor = None,   
                 freeze_item_emb: bool = True):
        super().__init__()
        
        # A. GNN Part
        if pretrained_gnn_embeddings is not None:
            # GNN 학습 결과 로드 (Num_Users, 64)
            self.gnn_user_emb = nn.Embedding.from_pretrained(pretrained_gnn_embeddings, freeze=False)
            current_gnn_dim = pretrained_gnn_embeddings.shape[1]
        else:
            # Fallback (테스트용)
            print("⚠️ [Warning] Initializing GNN Embedding Randomly.")
            current_gnn_dim = GNN_DIM
            self.gnn_user_emb = nn.Embedding(num_users, current_gnn_dim)
            nn.init.xavier_normal_(self.gnn_user_emb.weight)

        self.gnn_projector = nn.Sequential(
            nn.Linear(current_gnn_dim, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM),
            nn.GELU()
        )

        # B. Sequential Part
        # Item Tower 결과 로드 (Num_Items, 128)
        if pretrained_item_vectors is not None:
            self.item_content_emb = nn.Embedding.from_pretrained(pretrained_item_vectors, freeze=freeze_item_emb)
        else:
            print("⚠️ [Warning] Initializing Item Embedding Randomly.")
            self.item_content_emb = nn.Embedding(10000, EMBED_DIM) # Dummy size

        self.position_emb = nn.Embedding(MAX_SEQ_LEN, EMBED_DIM)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, 
            nhead=4, 
            dim_feedforward=EMBED_DIM*4, 
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # C. Meta Part
        self.meta_mlp = nn.Sequential(
            nn.Linear(NUM_META_FEATURES, EMBED_DIM // 2),
            nn.GELU(),
            nn.Linear(EMBED_DIM // 2, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM)
        )

        # D. Fusion
        self.fusion_layer = ContextGatingFusion(dim=EMBED_DIM)

    def forward(self, user_indices, history_item_ids, history_mask, meta_features):
        # 1. GNN
        v_gnn = self.gnn_projector(self.gnn_user_emb(user_indices))

        # 2. Sequential
        seq_emb = self.item_content_emb(history_item_ids)
        
        # Positional Encoding
        B, L = history_item_ids.shape
        positions = torch.arange(L, device=history_item_ids.device).unsqueeze(0)
        seq_input = seq_emb + self.position_emb(positions)

        # Transformer (Padding Masking)
        # mask가 1이면 Valid, 0이면 Padding -> key_padding_mask는 True가 Masking
        # 따라서 1-mask 혹은 ~mask.bool() 사용
        key_padding_mask = (history_mask == 0)
        seq_out = self.seq_encoder(seq_input, src_key_padding_mask=key_padding_mask)
        
        # Masked Mean Pooling
        mask_expanded = history_mask.unsqueeze(-1)
        sum_seq = (seq_out * mask_expanded).sum(dim=1)
        cnt_seq = mask_expanded.sum(dim=1).clamp(min=1e-9)
        v_seq = sum_seq / cnt_seq

        # 3. Meta
        v_meta = self.meta_mlp(meta_features)

        # 4. Fusion
        output = self.fusion_layer(v_gnn, v_seq, v_meta)
        
        return F.normalize(output, p=2, dim=1)

# -------------------------------------------------------------------------
# 2. Asset Loader Helper
# -------------------------------------------------------------------------
def load_pretrained_assets():
    """ GNN 가중치와 Item Vector 텐서를 파일에서 로드합니다. """
    print("\n📦 Loading Pre-trained Assets...")
    
    gnn_tensor = None
    item_tensor = None
    
    # 1. GNN Load
    if os.path.exists(GNN_PATH):
        print(f"   - Found GNN Checkpoint: {GNN_PATH}")
        ckpt = torch.load(GNN_PATH, map_location='cpu')
        print(f"     ✅ Loaded GNN Tensor: {ckpt.shape}")

    else:
        print(f"   ❌ GNN Checkpoint not found at {GNN_PATH}")

    # 2. Item Matrix Load
    if os.path.exists(ITEM_MATRIX_PATH):
        print(f"   - Found Item Matrix: {ITEM_MATRIX_PATH}")
        item_tensor = torch.load(ITEM_MATRIX_PATH, map_location='cpu')
        print(f"     ✅ Loaded Item Tensor: {item_tensor.shape}")
    else:
        print(f"   ❌ Item Matrix not found at {ITEM_MATRIX_PATH}")
        
    return gnn_tensor, item_tensor

# -------------------------------------------------------------------------
# 3. Dataset
# -------------------------------------------------------------------------
class UserSeqDataset(Dataset):
    def __init__(self, user_ids, history_seqs, target_items, meta_data, max_len=50):
        self.user_ids = user_ids
        self.history_seqs = history_seqs
        self.target_items = target_items
        self.meta_data = meta_data
        self.max_len = max_len

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        seq = self.history_seqs[idx]
        seq_len = len(seq)
        
        if seq_len >= self.max_len:
            seq = seq[-self.max_len:]
            mask = [1] * self.max_len
        else:
            pad_len = self.max_len - seq_len
            seq = seq + [0] * pad_len # 0 is Padding ID
            mask = [1] * seq_len + [0] * pad_len

        return {
            "user_idx": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "history_ids": torch.tensor(seq, dtype=torch.long),
            "history_mask": torch.tensor(mask, dtype=torch.long),
            "meta": torch.tensor(self.meta_data[idx], dtype=torch.float),
            "target_item_id": torch.tensor(self.target_items[idx], dtype=torch.long)
        }

# -------------------------------------------------------------------------
# 4. Training Loop
# -------------------------------------------------------------------------
def train_user_tower():
    # A. 데이터 준비 (Dummy Data for Demo)
    # 실제로는 DB에서 읽어와야 합니다.
    print("\n🛠️ Preparing Data...")
    num_dummy_users = 100
    dummy_user_ids = list(range(num_dummy_users))
    dummy_history = [np.random.randint(1, 1000, size=np.random.randint(5, 30)).tolist() for _ in range(num_dummy_users)]
    dummy_targets = np.random.randint(1, 1000, size=num_dummy_users).tolist()
    dummy_meta = np.random.randn(num_dummy_users, NUM_META_FEATURES).astype(np.float32)

    dataset = UserSeqDataset(dummy_user_ids, dummy_history, dummy_targets, dummy_meta, MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # B. 모델 초기화
    gnn_emb, item_emb = load_pretrained_assets()
    
    # GNN 텐서가 없으면 유저 수라도 맞춰서 더미 생성 (에러 방지)
    real_num_users = gnn_emb.shape[0] if gnn_emb is not None else 1000
    
    model = HybridUserTower(
        num_users=real_num_users,
        pretrained_gnn_embeddings=gnn_emb,
        pretrained_item_vectors=item_emb,
        freeze_item_emb=True
    ).to(DEVICE)

    # C. 학습 설정
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Target Item Lookup을 위한 고정 임베딩 레이어 (학습 X)
    # User Tower 내부의 item_content_emb와 같은 값을 공유하지만, 용도가 다름 (정답 비교용)
    if item_emb is not None:
        target_lookup = nn.Embedding.from_pretrained(item_emb, freeze=True).to(DEVICE)
    else:
        # Fallback
        target_lookup = nn.Embedding(10000, 128).to(DEVICE)

    # D. Training Loop
    EPOCHS = 5
    print(f"\n🚀 Start Training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        steps = 0
        model.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            optimizer.zero_grad()
            
            # Input
            u_idx = batch['user_idx'].to(DEVICE)
            h_ids = batch['history_ids'].to(DEVICE)
            h_mask = batch['history_mask'].to(DEVICE)
            meta = batch['meta'].to(DEVICE)
            t_ids = batch['target_item_id'].to(DEVICE)
            
            # 1. User Representation
            user_vec = model(u_idx, h_ids, h_mask, meta) # (B, 128)
            
            # 2. Target Item Representation
            # "이 유저가 실제로 산 그 아이템"의 미리 계산된 벡터를 가져옴
            target_vec = target_lookup(t_ids)
            target_vec = F.normalize(target_vec, p=2, dim=1) # (B, 128)
            
            # 3. In-batch Contrastive Loss
            # Score: (B, B) -> 대각선이 Positive Pair
            scores = torch.matmul(user_vec, target_vec.T) 
            scores = scores / 0.07 # Temperature
            
            labels = torch.arange(scores.size(0)).to(DEVICE)
            
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"   ✅ Epoch {epoch+1} Avg Loss: {total_loss/steps:.4f}")

    print("\n💾 Training Finished. Saving User Tower...")
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "user_tower_final.pth"))

if __name__ == "__main__":
    # 폴더가 없으면 에러 나므로 생성
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_user_tower()

# -------------------------------------------------------------------------
# 3. Usage Example
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Mock Data
    NUM_USERS = 1000
    NUM_ITEMS = 5000
    EMBED_DIM = 128
    
    # 1. Pre-trained Vectors (가정)
    # Item Tower에서 전체 아이템을 Inference해서 만든 (5000, 128) 행렬
    pretrained_item_vecs = torch.randn(NUM_ITEMS, EMBED_DIM) 
    pretrained_gnn_emb = torch.randn(NUM_USERS, EMBED_DIM)
    
    # 2. Dataset Preparation
    # 유저 0번이 [1, 2, 3]을 샀고, 다음에 4번을 샀다.
    user_ids = [0, 1, 2] * 100
    history_seqs = [[1, 2, 3], [10, 20], [100, 101, 102, 103]] * 100
    target_items = [4, 21, 104] * 100
    meta_data = torch.randn(300, 4) # (N, 4)
    
    dataset = UserSeqDataset(user_ids, history_seqs, target_items, meta_data)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3. Instantiate Model
    # 앞서 정의한 HybridUserTower 클래스
    model = HybridUserTower(
        num_users=NUM_USERS,
        pretrained_gnn_embeddings=pretrained_gnn_emb,
        pretrained_item_vectors=pretrained_item_vecs,
        freeze_item_emb=True # Item Embedding Layer는 고정 (학습 X)
    )
    
    # 4. Train
    trained_model = train_user_tower(
        model, 
        pretrained_item_vecs, 
        loader, 
        epochs=3
    )
'''