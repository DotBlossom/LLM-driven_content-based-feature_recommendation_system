import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# Global Config
# -------------------------------------------------------------------------
EMBED_DIM = 128          # Item Tower와 동일하게 맞춤
MAX_SEQ_LEN = 50         # 유저의 구매 이력 최대 길이
NUM_META_FEATURES = 4    # 예: 나이(1) + 성별(1) + 관심도(1) + 가격민감도(1)(정규화 평균)
DROPOUT = 0.1
GNN_DIM = 64
class HybridUserTower(nn.Module):
    def __init__(self, 
                 num_users: int, 
                 pretrained_gnn_embeddings: torch.Tensor = None, # SimGCL에서 학습된 User Emb
                 pretrained_item_vectors: torch.Tensor = None,   # Item Tower로 미리 뽑아둔 Item Vector Matrix
                 freeze_item_emb: bool = True):
        super().__init__()
        
        # ======================================================
        # 1. GNN Part (Collaborative Signal)
        # ======================================================
        # SimGCL에서 학습된 가중치를 로드 (ID 기반)
        if pretrained_gnn_embeddings is not None:
            # Input: (Num_Users, 64)
            self.gnn_user_emb = nn.Embedding.from_pretrained(pretrained_gnn_embeddings, freeze=False)
            current_gnn_dim = pretrained_gnn_embeddings.shape[1] # Should be 64
        else:
            # Fallback for testing
            current_gnn_dim = GNN_DIM
            self.gnn_user_emb = nn.Embedding(num_users, current_gnn_dim)
            nn.init.xavier_normal_(self.gnn_user_emb.weight)


        self.gnn_projector = nn.Sequential(
            nn.Linear(current_gnn_dim, EMBED_DIM), # 64 -> 128
            nn.LayerNorm(EMBED_DIM),
            nn.GELU()
        )
        # ======================================================
        # 2. Sequential Part (SesRec / Content Signal)
        # ======================================================
        # Item Tower의 128차원 벡터를 Lookup Table로 사용
        # (num_items, 128) 행렬이 들어와야 함
        if pretrained_item_vectors is not None:
            self.item_content_emb = nn.Embedding.from_pretrained(pretrained_item_vectors, freeze=freeze_item_emb)
        else:
            # Fallback (일반 학습용)
            self.item_content_emb = nn.Embedding(10000, EMBED_DIM)

        # Positional Embedding (순서 정보)
        self.position_emb = nn.Embedding(MAX_SEQ_LEN, EMBED_DIM)
        
        # Transformer Encoder (SASRec Style)
        # 유저의 구매 히스토리를 요약
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, 
            nhead=4, 
            dim_feedforward=EMBED_DIM*4, 
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True # Pre-LN 권장
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ======================================================
        # 3. Meta Part (User Demographics)
        # ======================================================
        self.meta_mlp = nn.Sequential(
            nn.Linear(NUM_META_FEATURES, EMBED_DIM // 2),
            nn.GELU(),
            nn.Linear(EMBED_DIM // 2, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM)
        )

        # ======================================================
        # 4. Gating Fusion
        # ======================================================
        # GNN(128) + Seq(128) + Meta(128) -> 가중치(Gate) 생성 / 임시
        self.gate_network = nn.Sequential(
            nn.Linear(EMBED_DIM * 3, 64),
            nn.Tanh(),
            nn.Linear(64, 3), # 3개의 소스에 대한 가중치
            nn.Softmax(dim=1)
        )
        
        # 최종 융합 후 Projection
        self.final_proj = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.LayerNorm(EMBED_DIM) # 최종 출력 정규화
        )

    def forward(self, 
                user_indices: torch.Tensor,      # (B,)
                history_item_ids: torch.Tensor,  # (B, Seq_Len) - Padding 포함
                history_mask: torch.Tensor,      # (B, Seq_Len) - 1:Valid, 0:Pad
                meta_features: torch.Tensor      # (B, Meta_Dim)
                ):
        
        # --- A. GNN Vector (Global Interest) ---
        
        # 1. Retrieve 64-dim vector
        v_gnn_raw = self.gnn_user_emb(user_indices) # (B, 64)
        
        # 2. Project to 128-dim
        v_gnn = self.gnn_projector(v_gnn_raw)

        # --- B. Sequential Vector (Current Interest) ---
        # 1. Item Embedding + Positional Embedding
        batch_size, seq_len = history_item_ids.shape
        
        # 아이템 벡터 (Pre-trained Item Tower의 지식)
        seq_emb = self.item_content_emb(history_item_ids) # (B, Seq, 128)
        
        # 위치 벡터
        positions = torch.arange(seq_len, device=history_item_ids.device).unsqueeze(0)
        pos_emb = self.position_emb(positions) # (1, Seq, 128)
        
        seq_input = seq_emb + pos_emb

        # 2. Transformer Encoding
        # src_key_padding_mask: True가 Masking됨 (PyTorch 표준) -> history_mask가 1(유효)면 False(안가림)여야 함
        # 따라서 ~history_mask.bool() 사용
        seq_out = self.seq_encoder(seq_input, src_key_padding_mask=~history_mask.bool())
        
        # 3. Pooling (Last Valid Item or Mean)
        # 여기서는 가장 최근에 산 물건(Last Valid)이 가장 중요하다고 가정 -> SASRec 방식
        # 혹은 전체 문맥(Mean) 사용.
        
        # [간편 구현] Masked Mean Pooling
        mask_expanded = history_mask.unsqueeze(-1) # (B, Seq, 1)
        sum_seq = (seq_out * mask_expanded).sum(dim=1)
        cnt_seq = mask_expanded.sum(dim=1).clamp(min=1e-9)
        v_seq = sum_seq / cnt_seq # (B, 128)

        # --- C. Meta Vector ---
        v_meta = self.meta_mlp(meta_features) # (B, 128)

        # --- D. Gating & Fusion ---
        # 3가지 벡터를 이어 붙여서 Gate 통과
        combined = torch.cat([v_gnn, v_seq, v_meta], dim=1) # (B, 384)
        gates = self.gate_network(combined) # (B, 3) -> [w_gnn, w_seq, w_meta]
        
        # 가중합 (Weighted Sum)
        v_final = (gates[:, 0:1] * v_gnn) + \
                  (gates[:, 1:2] * v_seq) + \
                  (gates[:, 2:3] * v_meta)
        
        # 최종 Projection (Retrieval을 위해)
        output = self.final_proj(v_final)
        
        # 내적 검색을 위해 L2 Normalize (SimCSE Item Tower와 호환성 유지)
        return F.normalize(output, p=2, dim=1)
    
    import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# -------------------------------------------------------------------------
# 1. Dataset for User Tower Training
# -------------------------------------------------------------------------
class UserSeqDataset(Dataset):
    def __init__(self, 
                 user_ids,         # List[int]
                 history_seqs,     # List[List[int]]: 과거 구매 이력 (Input)
                 target_items,     # List[int]: 다음에 실제로 구매한 아이템 (Label)
                 meta_data,        # Tensor: 유저 메타 정보
                 max_len=50):
        
        self.user_ids = user_ids
        self.history_seqs = history_seqs
        self.target_items = target_items
        self.meta_data = meta_data
        self.max_len = max_len

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        # 1. History Sequence Padding
        seq = self.history_seqs[idx]
        seq_len = len(seq)
        
        if seq_len >= self.max_len:
            seq = seq[-self.max_len:] # 최근거만
            mask = [1] * self.max_len
        else:
            # Pre-padding (앞을 0으로 채움) or Post-padding
            pad_len = self.max_len - seq_len
            seq = seq + [0] * pad_len # Post-padding (0 is PAD ID)
            mask = [1] * seq_len + [0] * pad_len

        return {
            "user_idx": torch.tensor(self.user_ids[idx], dtype=torch.long),
            "history_ids": torch.tensor(seq, dtype=torch.long),
            "history_mask": torch.tensor(mask, dtype=torch.long), # Transformer용 마스크
            "meta": torch.tensor(self.meta_data[idx], dtype=torch.float),
            "target_item_id": torch.tensor(self.target_items[idx], dtype=torch.long)
        }

# -------------------------------------------------------------------------
# 2. Training Loop (In-batch Negatives)
# -------------------------------------------------------------------------
def train_user_tower(
    user_tower_model: nn.Module,
    pretrained_item_matrix: torch.Tensor, # (Num_Items, 128) - Fixed
    dataloader: DataLoader,
    epochs=5,
    lr=1e-4,
    device='cuda'
):
    # 1. Setup
    user_tower_model = user_tower_model.to(device)
    user_tower_model.train()
    
    # Item Vector는 학습되지 않도록 고정 (Lookup Table로 사용)
    # Target Item의 벡터를 가져오기 위함
    item_emb_layer = nn.Embedding.from_pretrained(pretrained_item_matrix, freeze=True).to(device)
    
    optimizer = optim.AdamW(user_tower_model.parameters(), lr=lr)
    
    # Loss: InfoNCE (Contrastive Loss)
    # In-batch Negative Sampling을 활용한 CrossEntropy
    criterion = nn.CrossEntropyLoss()

    print(f"🚀 Start Training User Tower for {epochs} epochs...")

    for epoch in range(epochs):
        total_loss = 0
        step = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            # --- Move to Device ---
            user_idx = batch['user_idx'].to(device)
            hist_ids = batch['history_ids'].to(device)
            hist_mask = batch['history_mask'].to(device)
            meta = batch['meta'].to(device)
            target_ids = batch['target_item_id'].to(device) # 정답 아이템 ID

            # --- Forward Pass ---
            
            # 1. User Vector 생성 (User Tower가 요리함)
            # (Batch, 128)
            user_vector = user_tower_model(user_idx, hist_ids, hist_mask, meta)
            
            # 2. Target Item Vector 가져오기 (이미 학습된 Item Tower 결과값)
            # (Batch, 128)
            target_item_vector = item_emb_layer(target_ids)
            target_item_vector = F.normalize(target_item_vector, p=2, dim=1) # Normalize 필수
            
            # --- Loss Calculation (In-batch Negatives) ---
            # User(B, D) @ Item(B, D).T -> Score Matrix (B, B)
            # 대각선: Positive (내 유저가 산 내 아이템)
            # 나머지: Negative (내 유저가 안 산, 남이 산 아이템)
            scores = torch.matmul(user_vector, target_item_vector.T) # (Batch, Batch)
            
            # Temperature Scaling
            temperature = 0.07
            scores = scores / temperature
            
            # Labels: 0, 1, 2, ... (대각선 인덱스)
            labels = torch.arange(scores.size(0)).to(device)
            
            loss = criterion(scores, labels)
            
            # --- Backward ---
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            
            if step % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{step}] Loss: {loss.item():.4f}")

        print(f"==== Epoch {epoch+1} Avg Loss: {total_loss/step:.4f} ====")

    print("✅ Training Finished.")
    return user_tower_model

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