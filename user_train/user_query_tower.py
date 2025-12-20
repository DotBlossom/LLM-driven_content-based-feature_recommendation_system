import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. Sub-Modules (SE-Block, Residual Head)
# ==========================================

class SEResidualBlock(nn.Module):
    """
    [SE-ResBlock] Channel Attention + Residual Connection
    """
    def __init__(self, dim, dropout=0.2, expansion_factor=4):
        super().__init__()
        # 1. Feature Transformation
        self.block = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.LayerNorm(dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.LayerNorm(dim),
        )
        # 2. SE-Block (Squeeze & Excitation)
        self.se_block = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        self.act = nn.GELU()
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        weight = self.se_block(out)
        out = out * weight  # Channel Attention
        return self.act(residual + out)

class DeepResidualHead(nn.Module):
    """
    [User Tower Head] Expand -> Deep Interaction -> Compress
    """
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        mid_dim = input_dim * 2
        hidden_dim = input_dim * 4
        
        # Progressive Expansion
        self.expand_layer1 = nn.Sequential(
            nn.Linear(input_dim, mid_dim), nn.LayerNorm(mid_dim), nn.GELU(), nn.Dropout(0.1)
        )
        self.expand_layer2 = nn.Sequential(
            nn.Linear(mid_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1)
        )
        
        # Deep Interaction
        self.res_blocks = nn.Sequential(
            SEResidualBlock(hidden_dim, dropout=0.2),
            SEResidualBlock(hidden_dim, dropout=0.2)
        )
        
        # Compression
        self.final_proj = nn.Linear(hidden_dim, output_dim)
        self.input_skip = nn.Linear(input_dim, output_dim) # Global Shortcut

    def forward(self, x):
        m = self.expand_layer1(x)
        h = self.expand_layer2(m)
        h = self.res_blocks(h)
        main_out = self.final_proj(h)
        skip_out = self.input_skip(x)
        return main_out + skip_out

class OptimizedItemTower(nn.Module):
    """
    [Final Projection] Metric Learning용 정규화 레이어
    """
    def __init__(self, input_dim=128, output_dim=128):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(), 
            nn.Linear(input_dim, output_dim),
        )
        
    def forward(self, x):
        x = self.layer(x)
        return F.normalize(x, p=2, dim=1)

# ==========================================
# 2. Main Model: FinalUserTower
# ==========================================

class FinalUserTower(nn.Module):
    def __init__(self, 
                 num_total_products: int,
                 pretrained_item_matrix: torch.Tensor = None,
                 max_seq_len: int = 50,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 output_dim: int = 128):
        super().__init__()
        
        # Embeddings
        self.item_embedding = nn.Embedding(num_total_products + 1, d_model, padding_idx=0)
        if pretrained_item_matrix is not None:
            self.load_pretrained_weights(pretrained_item_matrix)
            
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.season_embedding = nn.Embedding(4, d_model) # 0:Pad, 1~3:Seasons
        self.gender_embedding = nn.Embedding(3, d_model, padding_idx=0)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # User Query Token
        self.user_query_token = nn.Parameter(torch.randn(1, 1, d_model)) 

        # Heads
        self.deep_head = DeepResidualHead(input_dim=d_model, output_dim=output_dim)
        self.final_projector = OptimizedItemTower(input_dim=output_dim, output_dim=output_dim)

    def load_pretrained_weights(self, matrix):
        # matrix shape: (N+1, d_model)
        if self.item_embedding.weight.shape != matrix.shape:
             # 안전장치: shape 다르면 맞춤 (보통은 같아야 함)
            print(f"⚠️ Resizing embedding to match pretrained: {matrix.shape}")
            self.item_embedding = nn.Embedding.from_pretrained(matrix, padding_idx=0, freeze=True)
        else:
            self.item_embedding.weight.data.copy_(matrix)
            self.item_embedding.weight.requires_grad = False # Freeze

    def forward(self, history_ids, season_idx, gender_idx):
        B, L = history_ids.shape
        device = history_ids.device
        
        # 1. Embedding & Context Injection
        seq_emb = self.item_embedding(history_ids)
        pos_emb = self.position_embedding(torch.arange(L, device=device))
        
        # Shape Check & Broadcast
        season_emb = self.season_embedding(season_idx).unsqueeze(1) # (B, 1, D)
        gender_emb = self.gender_embedding(gender_idx).unsqueeze(1) # (B, 1, D)
        
        x = seq_emb + pos_emb + season_emb + gender_emb
        
        # 2. Prepend User Token
        cls_token = self.user_query_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # 3. Masking
        padding_mask = (history_ids == 0)
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)
        full_mask = torch.cat([cls_mask, padding_mask], dim=1)
        
        # 4. Transformer
        out = self.transformer(x, src_key_padding_mask=full_mask)
        raw_user_vector = out[:, 0, :] # Extract CLS token
        
        # 5. Deep Interaction & Projection
        deep_feat = self.deep_head(raw_user_vector)
        final_vector = self.final_projector(deep_feat)
        
        return final_vector

# ==========================================
# 3. Dataset & Loss
# ==========================================

class UserSessionDataset(Dataset):
    def __init__(self, user_sessions: list, max_len: int = 50):
        self.data = user_sessions
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        history = row['history']
        
        # Truncate or Pad
        if len(history) > self.max_len:
            history = history[-self.max_len:]
        else:
            history = history + [0] * (self.max_len - len(history))
            
        return {
            'history': torch.tensor(history, dtype=torch.long),
            'season': torch.tensor(row['season'], dtype=torch.long),
            'gender': torch.tensor(row['gender'], dtype=torch.long),
            'target_item_id': torch.tensor(row['target_item_id'], dtype=torch.long)
        }

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, user_vectors, item_vectors):
        # Cosine Similarity (이미 Normalized 되었다고 가정)
        scores = torch.matmul(user_vectors, item_vectors.T)
        scores = scores / self.temperature
        
        labels = torch.arange(user_vectors.size(0)).to(user_vectors.device)
        return self.criterion(scores, labels)

# ==========================================
# 4. Proxy Item Tower (For Training Loop Compatibility)
# ==========================================
class InferenceItemTower(nn.Module):
    """
    [역할] 학습 루프에서 'item_tower(target_ids)'를 호출할 때
    Pretrained Weight를 Lookup하고 Projection을 수행하여 정답 벡터를 리턴함.
    """
    def __init__(self, pretrained_matrix, output_dim=128):
        super().__init__()
        # Freeze된 임베딩
        self.embedding = nn.Embedding.from_pretrained(pretrained_matrix, padding_idx=0, freeze=True)
        # User Tower와 동일한 위상으로 매핑하기 위한 Projector
        # (실제론 Item Tower 학습 시 사용된 Projector를 로드해야 함)
        self.projector = OptimizedItemTower(input_dim=pretrained_matrix.shape[1], output_dim=output_dim)
        
    def forward(self, item_ids):
        x = self.embedding(item_ids)
        return self.projector(x)

# ==========================================
# 5. Training Loop (Provided by User)
# ==========================================

def train_user_tower(
    user_tower: nn.Module,
    item_tower: nn.Module,      # Pre-trained & Frozen
    train_loader: DataLoader,   # item_feature_db 인자 제거됨
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cuda"
):
    # 1. Setup
    user_tower.to(device)
    item_tower.to(device)
    
    # Item Tower Freezing (학습되지 않도록 고정)
    item_tower.eval()
    for param in item_tower.parameters():
        param.requires_grad = False
        
    optimizer = optim.AdamW(user_tower.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = InfoNCELoss(temperature=0.07)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, 
        steps_per_epoch=len(train_loader), epochs=epochs
    )

    print("🚀 Start Training User Tower...")
    
    for epoch in range(epochs):
        user_tower.train()
        total_loss = 0
        step = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # Data to Device
            history = batch['history'].to(device)
            season = batch['season'].to(device)
            gender = batch['gender'].to(device)
            target_ids = batch['target_item_id'].to(device) 
            
            # -----------------------------------------------------------
            # A. Generate Ground Truth (Teacher)
            # -----------------------------------------------------------
            with torch.no_grad():
                # Item Tower가 ID를 받아서 Pretrained Vector를 리턴한다고 가정
                target_item_vectors = item_tower(target_ids)
                
            # -----------------------------------------------------------
            # B. Generate User Vectors (Student)
            # -----------------------------------------------------------
            user_vectors = user_tower(history, season, gender)
            
            # -----------------------------------------------------------
            # C. Loss Calculation
            # -----------------------------------------------------------
            loss = loss_fn(user_vectors, target_item_vectors)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(user_tower.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            step += 1
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"📊 Epoch {epoch+1} Avg Loss: {total_loss / step:.4f}")
        
    print("✅ Training Finished.")
    return user_tower

# ==========================================
# 6. Execution Example
# ==========================================
if __name__ == "__main__":
    # --- Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_PRODUCTS = 10000
    EMBED_DIM = 128
    BATCH_SIZE = 64
    
    print(f"🔧 Device: {DEVICE}")

    # 1. Pretrained Vectors (Mock)
    # 실제 환경에선 load_pretrained_vectors_from_db() 결과 사용
    pretrained_matrix = torch.randn(NUM_PRODUCTS + 1, EMBED_DIM)
    pretrained_matrix[0] = 0 # Padding
    
    # 2. Initialize Models
    # User Tower
    user_tower = FinalUserTower(
        num_total_products=NUM_PRODUCTS,
        pretrained_item_matrix=pretrained_matrix,
        d_model=EMBED_DIM,
        output_dim=EMBED_DIM
    )
    
    # Item Tower (Proxy for Training)
    item_tower = InferenceItemTower(pretrained_matrix, output_dim=EMBED_DIM)

    # 3. Create Dummy Data (Synthetic)
    print("🎲 Generating Dummy Dataset...")
    dummy_sessions = []
    for _ in range(1000): # 1000 Samples
        dummy_sessions.append({
            'history': list(np.random.randint(1, NUM_PRODUCTS, 20)), # Random History
            'season': np.random.randint(1, 4), # 1~3
            'gender': np.random.randint(1, 3), # 1~2
            'target_item_id': np.random.randint(1, NUM_PRODUCTS) # Ground Truth Item
        })
    
    dataset = UserSessionDataset(dummy_sessions)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 4. Run Training
    trained_model = train_user_tower(
        user_tower=user_tower,
        item_tower=item_tower,
        train_loader=train_loader,
        epochs=3,
        device=DEVICE
    )