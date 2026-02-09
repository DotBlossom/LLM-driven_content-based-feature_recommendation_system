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
import warnings
warnings.filterwarnings("ignore", message="Support for mismatched src_key_padding_mask and mask is deprecated")

# ==========================================
# ⚙️ 설정 & 경로
# ==========================================
TEMPERATURE = 0.15
LAMBDA_LOGQ = 0.0
BATCH_SIZE = 768
EMBED_DIM = 128
MAX_SEQ_LEN = 50
DROPOUT = 0.2
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

SAVE_PATH_BEST = os.path.join(MODEL_DIR, "user_tower_phase2.5_best_ft.pth")

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
        sorted_probs = self.items['raw_probability'].reindex(self.item_ids).fillna(0).values
        return torch.tensor(sorted_probs, dtype=torch.float32).to(device)

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
        model.item_content_emb = nn.Embedding.from_pretrained(new_weight.to(device), freeze=True)
        
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
        model.gnn_item_emb = nn.Embedding.from_pretrained(new_weight.to(device), freeze=True)
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
        model.gnn_user_emb = nn.Embedding.from_pretrained(new_weight.to(device), freeze=True)
        print(f"✅ Injected into 'model.gnn_user_emb'")

    print(f"✅ [GNN User Alignment] Matched: {matched}/{len(processor.user_ids)}")
    return model

def verify_embedding_alignment(model, processor, model_dir):
    # (생략: 기존 코드와 동일, 필요시 추가)
    pass

# ==========================================
# 3. Model Definition (Fixed)
# ==========================================
class RobustFusion(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
    def forward(self, v_gnn, v_seq, v_meta):
        combined = torch.cat([v_gnn, v_seq, v_meta], dim=-1)
        return self.fusion_mlp(combined)

class HybridUserTower(nn.Module):
    def __init__(self, num_users, num_items, gnn_user_init, gnn_item_init, item_content_init):
        super().__init__()
        self.embed_dim = 128
        
        # [핵심 수정] 유저와 아이템 GNN 임베딩 분리
        # 1. GNN User Embedding (96만 개)
        self.gnn_user_emb = nn.Embedding.from_pretrained(gnn_user_init, freeze=True)
        
        # 2. GNN Item Embedding (4.7만 개) - 현재 Forward엔 안쓰여도 정렬 위해 존재해야 함
        self.gnn_item_emb = nn.Embedding.from_pretrained(gnn_item_init, freeze=True)

        # 3. Content Item Embedding (4.7만 개)
        self.item_content_emb = nn.Embedding.from_pretrained(item_content_init, freeze=True)
        
        # Projector
        self.gnn_projector = nn.Sequential(
            nn.Linear(gnn_user_init.shape[1], 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )   
        
        # Other Layers
        self.time_emb = nn.Embedding(1001, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=512, dropout=DROPOUT, batch_first=True, norm_first=True)
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        self.channel_emb = nn.Embedding(2, 32)
        self.meta_mlp = nn.Sequential(nn.Linear(35, 128), nn.GELU(), nn.Linear(128, 128), nn.LayerNorm(128))
        self.fusion_layer = RobustFusion(dim=128)

    def forward(self, u_idx, seq_ids, seq_deltas, seq_mask, u_dense, u_cat):
        B, L = seq_ids.shape
        
        # 1. GNN Features (User Embedding 사용!)
        # u_idx는 0~96만 범위이므로 gnn_user_emb(96만)을 참조해야 안전함
        v_gnn = self.gnn_projector(self.gnn_user_emb(u_idx))
        v_gnn = F.normalize(v_gnn, p=2, dim=1)
        v_gnn_seq = v_gnn.unsqueeze(1).expand(-1, L, -1)
        
        # 2. Meta Features
        cat_vec = self.channel_emb(u_cat)
        v_meta = self.meta_mlp(torch.cat([u_dense, cat_vec], dim=1))
        v_meta = F.normalize(v_meta, p=2, dim=1)
        v_meta_seq = v_meta.unsqueeze(1).expand(-1, L, -1)
        
        # 3. Sequence Features (Content Item Embedding 사용)
        seq_input = self.item_content_emb(seq_ids) * math.sqrt(self.embed_dim) + self.time_emb(seq_deltas.clamp(max=1000))
        
        causal_mask = torch.triu(torch.ones(L, L, device=seq_ids.device) * float('-inf'), diagonal=1)
        key_padding_mask = (seq_mask == 0)

        seq_out = self.seq_encoder(seq_input, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        v_seq = F.normalize(seq_out, p=2, dim=2)

        # 4. Fusion
        output = self.fusion_layer(v_gnn_seq, v_seq, v_meta_seq)
        return F.normalize(output, p=2, dim=2)

# ==========================================
# 4. Loss & Eval
# ==========================================
def logq_correction_loss(user_emb, item_emb, pos_item_ids, item_probs, temperature=0.07, lambda_logq=0.0):
    scores = torch.matmul(user_emb, item_emb.T)
    if lambda_logq > 0.0:
        log_q = torch.log(item_probs[pos_item_ids] + 1e-9).view(1, -1)
        scores = scores - (lambda_logq * log_q)
    logits = scores / temperature
    is_collision = (pos_item_ids.unsqueeze(1) == pos_item_ids.unsqueeze(0))
    mask = is_collision.fill_diagonal_(False)
    logits = logits.masked_fill(mask, -1e4)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)

def evaluate_recall_multi_k(model, processor, target_df_path, k_list=[20, 100, 500], batch_size=256):
    model.eval()
    target_df = pd.read_parquet(target_df_path)
    target_dict = target_df.set_index('customer_id')['target_ids'].to_dict()
    
    val_loader = DataLoader(UserTowerDataset(processor, is_training=False), batch_size=batch_size, shuffle=False, collate_fn=user_tower_collate_fn)
    
    with torch.no_grad():
        # 평가시에는 Content Item Embedding을 후보군으로 사용
        all_item_vecs = F.normalize(model.item_content_emb(torch.arange(1, len(processor.item_ids)+1).to(DEVICE)), p=2, dim=1)

    hit_counts = {k: 0 for k in k_list}
    total_users = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            u_idx, u_dense, u_cat, seq_ids, seq_deltas, seq_mask, _, _ = [x.to(DEVICE) for x in batch]
            batch_uids = [processor.user_ids[i-1] for i in u_idx.cpu().numpy()]
            valid_idx_list = [i for i, uid in enumerate(batch_uids) if uid in target_dict]
            if not valid_idx_list: continue
            
            v_idx = torch.tensor(valid_idx_list).to(DEVICE)
            seq_out = model(u_idx[v_idx], seq_ids[v_idx], seq_deltas[v_idx], seq_mask[v_idx], u_dense[v_idx], u_cat[v_idx])
            
            lengths = seq_mask[v_idx].sum(dim=1)
            last_indices = (lengths - 1).clamp(min=0)
            batch_range = torch.arange(seq_out.size(0), device=DEVICE)
            last_user_vecs = seq_out[batch_range, last_indices]
            
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

# ==========================================
# 5. Main Training Function (Refactored)
# ==========================================
def train_phase_2_5_emergency_fix():
    logger.log(1, "🚀 Phase 2.5: Emergency Fix Running...")
    
    # 1. Load Data
    train_proc = FeatureProcessor(USER_FEAT_PATH_PQ, ITEM_FEAT_PATH_PQ, SEQ_DATA_PATH_PQ, scaler=None)
    valid_proc = FeatureProcessor(USER_VAL_FEAT_PATH, ITEM_FEAT_PATH_PQ, SEQ_VAL_DATA_PATH, scaler=train_proc.user_scaler)
    valid_proc.item2id, valid_proc.item_ids = train_proc.item2id, train_proc.item_ids

    # 2. Prepare Dummy Inits (Correct Shapes!)
    num_users = len(train_proc.user_ids) + 1  # ~960k
    num_items = len(train_proc.item_ids) + 1  # ~47k
    
    logger.log(1, f"Initializing Model with Users: {num_users}, Items: {num_items}")

    # [수정] 3개의 Dummy 텐서 준비
    dummy_gnn_user = torch.zeros((num_users, 64))   # GNN User (960k)
    dummy_gnn_item = torch.zeros((num_items, 64))   # GNN Item (47k)
    dummy_content  = torch.zeros((num_items, 128))  # Content Item (47k)

    # 3. Model Init
    model = HybridUserTower(
        num_users=num_users, 
        num_items=num_items, 
        gnn_user_init=dummy_gnn_user, 
        gnn_item_init=dummy_gnn_item,
        item_content_init=dummy_content
    ).to(DEVICE)
   
    # 4. Alignment (Injection)
    # (A) Content Item
    model = load_and_align_embeddings(model, train_proc, model_dir=MODEL_DIR, device=DEVICE)
    # (B) GNN Item (model.gnn_item_emb 채움)
    model = load_and_align_gnn_items(model, train_proc, base_dir=BASE_DIR, device=DEVICE)
    # (C) GNN User (model.gnn_user_emb 채움 - 중요!)
    model = load_and_align_gnn_user_embeddings(model, train_proc, base_dir=BASE_DIR, device=DEVICE)
    

    # 5. Optimizer & Scheduler
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(model_params, lr=5e-4, betas=(0.9, 0.98), weight_decay=0.01, eps=1e-6)
    
    train_loader = DataLoader(UserTowerDataset(train_proc, is_training=True), batch_size=BATCH_SIZE, shuffle=True, collate_fn=user_tower_collate_fn)
    
    total_steps = len(train_loader) * EPOCHS 
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler('cuda')
    item_probs = train_proc.get_logq_probs(DEVICE)
    best_r100 = 0.0

    # 6. Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            u_idx, u_dense, u_cat, seq_ids, seq_deltas, seq_mask, target_ids, _ = [x.to(DEVICE) for x in batch]
            
            optimizer.zero_grad() 
            with torch.amp.autocast('cuda'):
                user_seq_vecs = model(u_idx, seq_ids, seq_deltas, seq_mask, u_dense, u_cat)
                
                valid_mask = (target_ids != 0) 
                active_user_vecs = user_seq_vecs[valid_mask] 
                active_target_ids = target_ids[valid_mask]
                active_item_vecs = F.normalize(model.item_content_emb(active_target_ids), p=2, dim=1)
                
                loss = logq_correction_loss(active_user_vecs, active_item_vecs, active_target_ids, item_probs, TEMPERATURE, LAMBDA_LOGQ)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step() 
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})

        avg_loss = total_loss / len(train_loader)
        logger.log(1, f"📊 Epoch {epoch+1} Result: Avg Loss {avg_loss:.4f}")

        metrics = evaluate_recall_multi_k(model, valid_proc, TARGET_VAL_PATH, k_list=[20, 100, 500], batch_size=256)
        if metrics['R@100'] > best_r100:
            best_r100 = metrics['R@100']
            torch.save(model.state_dict(), SAVE_PATH_BEST)
            logger.log(1, f"🌟 New Best R@100: {best_r100:.4f} - Model Saved!")

if __name__ == "__main__":
    train_phase_2_5_emergency_fix()
    
    
    
    '''# Optimizer 설정 부분
model_params = []
embedding_params = []

for name, param in model.named_parameters():
    if not param.requires_grad: continue
    # 임베딩 레이어들은 천천히 학습
    if 'emb' in name:
        embedding_params.append(param)
    else:
        model_params.append(param)

optimizer = optim.AdamW([
    {'params': model_params, 'lr': 5e-4},       # 뇌(Transformer)는 빠르게
    {'params': embedding_params, 'lr': 1e-5}    # 몸(Embedding)은 아주 천천히 (5e-6 ~ 1e-5)
], weight_decay=0.01)

    
        # 4. 저장된 가중치(State Dict) 덮어씌우기
    if os.path.exists(SAVE_PATH_BEST_PREV):
        checkpoint = torch.load(SAVE_PATH_BEST_PREV, map_location=DEVICE)
        model.load_state_dict(checkpoint, strict=True) # strict=True: 구조가 완벽히 일치해야 함
        print(f"✅ Successfully loaded model from: {SAVE_PATH_BEST_PREV}")
    else:
        print(f"❌ Model file not found: {SAVE_PATH_BEST_PREV}")
        
    '''