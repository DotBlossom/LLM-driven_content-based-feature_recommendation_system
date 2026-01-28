import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import gc
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# ==========================================
# 0. 데이터 전처리 
# ==========================================
def load_and_process_data(json_file_path, cache_dir="cache"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    cache_path = os.path.join(cache_dir, "processed_graph_train.pt")
    map_path = os.path.join(cache_dir, "id_maps_train.pt")

    if os.path.exists(cache_path) and os.path.exists(map_path):
        print(f"[Cache Hit] Loading graph data from {cache_path}...")
        data_dict = torch.load(cache_path)
        maps = torch.load(map_path)
        return (data_dict['edge_index'], data_dict['num_users'], data_dict['num_items'], 
                maps['user2id'], maps['item2id'])

    print(f"[Cache Miss] Processing Train Sequences from {json_file_path}...")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        train_seq_data = json.load(f)
    
    print("Mapping IDs...")
    user_list = list(train_seq_data.keys())
    user2id = {u: i for i, u in enumerate(user_list)}
    
    all_items = set()
    for items in train_seq_data.values():
        all_items.update(items)
    
    item_list = list(all_items)
    item2id = {item: i for i, item in enumerate(item_list)}
    
    num_users = len(user2id)
    num_items = len(item2id)
    
    print(f" -> Num Users: {num_users}")
    print(f" -> Num Items: {num_items}")

    print("Building Edge Index...")
    src_nodes = []
    dst_nodes = []
    
    for u_str, i_list in tqdm(train_seq_data.items(), desc="Flattening Edges"):
        u_idx = user2id[u_str]
        for i_str in i_list:
            if i_str in item2id:
                i_idx = item2id[i_str]
                src_nodes.append(u_idx)
                dst_nodes.append(i_idx)
    
    src = torch.tensor(src_nodes, dtype=torch.long)
    dst = torch.tensor(dst_nodes, dtype=torch.long)
    
    edge_tensor = torch.stack([src, dst], dim=1)
    edge_tensor = torch.unique(edge_tensor, dim=0)
    edge_index = edge_tensor.t()
    
    print(f" -> Total Interactions (Edges): {edge_index.size(1)}")
    
    del train_seq_data, src_nodes, dst_nodes, src, dst, edge_tensor
    gc.collect()

    print("Saving to cache...")
    torch.save({'edge_index': edge_index, 'num_users': num_users, 'num_items': num_items}, cache_path)
    torch.save({'user2id': user2id, 'item2id': item2id}, map_path)

    return edge_index, num_users, num_items, user2id, item2id

# ==========================================
# 1. Graph Dataset Loader 
# ==========================================
class GraphDataset:
    def __init__(self, num_users, num_items, edge_index, device):
        self.num_users = num_users
        self.num_items = num_items
        self.device = device
        self.Graph = self._get_sparse_graph(edge_index.to(device))

    def _get_sparse_graph(self, edge_index_gpu): 
        print("Generating Sparse Graph Adjacency Matrix...")
        n_nodes = self.num_users + self.num_items
        
        users = edge_index_gpu[0]
        items = edge_index_gpu[1]
        items_offset = items + self.num_users
        
        row = torch.cat([users, items_offset])
        col = torch.cat([items_offset, users])
        
        indices = torch.stack([row, col], dim=0)
        values = torch.ones(indices.size(1), device=self.device)
        
        deg = torch.zeros(n_nodes, device=self.device)
        deg = deg.scatter_add(0, row, values)
        
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        row_idx = indices[0]
        col_idx = indices[1]
        
        norm_values = values * deg_inv_sqrt[row_idx] * deg_inv_sqrt[col_idx]
        
        norm_adj = torch.sparse_coo_tensor(indices, norm_values, size=(n_nodes, n_nodes))
        return norm_adj

# ==========================================
# 2. SimGCL Model 
# ==========================================
class SimGCL(nn.Module):
    def __init__(self, dataset, embed_dim=128, n_layers=2, eps=0.1): 
        super(SimGCL, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.Graph = dataset.Graph
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.eps = eps

        self.embedding_user = nn.Embedding(self.num_users, self.embed_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.embed_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)

    def perturb_embedding(self, embeds):
        noise = torch.rand_like(embeds)
        noise = F.normalize(noise, dim=1)
        return embeds + self.eps * noise

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_user.weight, self.embedding_item.weight], dim=0)
        
        if perturbed:
            ego_embeddings = self.perturb_embedding(ego_embeddings)

        all_embeddings = [ego_embeddings]
        
        for k in range(self.n_layers):
            # Sparse MM은 FP32 연산이 필수, Autocast off
            with torch.amp.autocast(device_type='cuda', enabled=False):
                ego_embeddings = torch.sparse.mm(self.Graph, ego_embeddings.float())
            all_embeddings.append(ego_embeddings)
            
        final_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        users_emb, items_emb = torch.split(final_embeddings, [self.num_users, self.num_items])
        
        return users_emb, items_emb

# ==========================================
# 3. Loss Functions 
# ==========================================
def bpr_loss(users_emb, pos_items_emb, neg_items_emb):
    pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
    neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)
    loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
    return loss

def info_nce_loss(view1_emb, view2_emb, temperature=0.2):
    view1_emb = F.normalize(view1_emb, dim=1)
    view2_emb = F.normalize(view2_emb, dim=1)
    pos_score = torch.sum(view1_emb * view2_emb, dim=1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1_emb, view2_emb.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score).mean()
    return loss

# ==========================================
# 4. Main Execution (Warning Fix 적용)
# ==========================================
if __name__ == "__main__":
    
    BASE_DIR = r"D:\trainDataset\localprops"
    TRAIN_FILE_NAME = "final_train_seq.json" 
    checkpoint_dir="./checkpoints"
    real_data_path = os.path.join(BASE_DIR, TRAIN_FILE_NAME)
    cache_dir = os.path.join(BASE_DIR, "cache")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"📂 Created checkpoint directory: {checkpoint_dir}")
        
        
    # 1. Load Data
    edge_index, NUM_USERS, NUM_ITEMS, u_map, i_map = load_and_process_data(real_data_path, cache_dir)
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Dataset & Loader
    graph_dataset = GraphDataset(NUM_USERS, NUM_ITEMS, edge_index, device)
    train_ds = TensorDataset(edge_index[0], edge_index[1])
    
    BATCH_SIZE = 10240
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)

    # 3. Model Setup
    model = SimGCL(graph_dataset, embed_dim=64, n_layers=2, eps=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    
    scaler = torch.amp.GradScaler('cuda')

    # 4. Training Loop
    EPOCHS = 10
    CL_INTERVAL = 5
    LAMBDA = 0.2
    
    print(f"\n[Training Start] Users: {NUM_USERS}, Items: {NUM_ITEMS}")
    
    # 학습 초반 튀는 것을 막고 후반부 미세 조정을 위해 필수입니다.
    from torch.optim.lr_scheduler import OneCycleLR
    
    # Steps per epoch 계산
    steps_per_epoch = len(train_loader)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.005,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,  # 전체의 10% 기간 동안 Warmup
        anneal_strategy='cos'
    )
    
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (batch_users, batch_pos_items) in enumerate(pbar):
            batch_users = batch_users.to(device)
            batch_pos_items = batch_pos_items.to(device)
            batch_neg_items = torch.randint(0, NUM_ITEMS, (len(batch_users),), device=device)
            
            optimizer.zero_grad()
            
            # autocast
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # BPR Task
                u_emb, i_emb = model(perturbed=False)
                loss_bpr = bpr_loss(u_emb[batch_users], i_emb[batch_pos_items], i_emb[batch_neg_items])
                
                # CL Task (Lazy)
                if batch_idx % CL_INTERVAL == 0:
                    u_view1, i_view1 = model(perturbed=True)
                    u_view2, i_view2 = model(perturbed=True)
                    loss_cl = info_nce_loss(u_view1[batch_users], u_view2[batch_users]) + \
                              info_nce_loss(i_view1[batch_pos_items], i_view2[batch_pos_items])
                    loss = loss_bpr + LAMBDA * loss_cl
                else:
                    loss = loss_bpr

            scaler.scale(loss).backward()
            # [필수x]Gradient Clipping 
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'avg': f"{avg_loss:.4f}"
                })
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),     # 모델 가중치
                'optimizer_state_dict': optimizer.state_dict(),        # 옵티마이저 상태 (Momentum 등)
                'loss': avg_loss,                                      # 기록용 Loss
                'config': {                                            # (선택) 모델 복원 시 참고할 Config
                    'embed_dim': 128,
                    'gnn_dim': 64
                }
            }

            save_path = os.path.join(checkpoint_dir, f"user_tower_epoch_{epoch+1}.pth")
            torch.save(checkpoint, save_path)
            print(f"✅ Checkpoint saved: {save_path}")
    
    # 5. Save Model
    model_save_path = os.path.join(cache_dir, "simgcl_trained.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    
    
    
    
    
    
    
def resume_training(model, optimizer, checkpoint_path):
    print(f"🔄 Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    
    # 1. 모델 가중치 복원
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 2. 옵티마이저 상태 복원 (학습률, 모멘텀 등)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 3. 에포크 정보 복원
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"✅ Resumed from Epoch {start_epoch} (Loss: {loss:.4f})")
    return start_epoch