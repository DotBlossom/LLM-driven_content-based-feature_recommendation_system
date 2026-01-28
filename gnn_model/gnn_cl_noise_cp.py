import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
import json
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

# ==========================================
# 0. Data Processing Utilities
# ==========================================
def load_and_process_data(json_file_path, cache_dir="cache"):
    """
    JSON 데이터를 로드하고 PyTorch Geometric 호환 Edge Index로 변환
    """
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
# 1. SimGCL Model
# ==========================================
class SimGCL(nn.Module):
    def __init__(self, dataset, embed_dim=64, n_layers=2, eps=0.1): 
        super(SimGCL, self).__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.Graph = dataset.Graph
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.eps = eps

        self.embedding_user = nn.Embedding(self.num_users, self.embed_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.embed_dim)
        # Xavier Initialization
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
            # Sparse MM requires FP32 usually, disable autocast for this op if needed
            with torch.amp.autocast(device_type='cuda', enabled=False):
                ego_embeddings = torch.sparse.mm(self.Graph, ego_embeddings.float())
            all_embeddings.append(ego_embeddings)
            
        final_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        users_emb, items_emb = torch.split(final_embeddings, [self.num_users, self.num_items])
        
        return users_emb, items_emb


# ==========================================
# 2. SimGCLLoss (Loss Logic Encapsulated)
# ==========================================
class SimGCLLoss(nn.Module):
    def __init__(self, lambda_val=0.2, temperature=0.2):
        super(SimGCLLoss, self).__init__()
        self.lambda_val = lambda_val
        self.temperature = temperature

    def _bpr_loss(self, users_emb, pos_items_emb, neg_items_emb):
        pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return loss

    def _info_nce_loss(self, view1_emb, view2_emb):
        view1_emb = F.normalize(view1_emb, dim=1)
        view2_emb = F.normalize(view2_emb, dim=1)
        
        pos_score = torch.sum(view1_emb * view2_emb, dim=1)
        pos_score = torch.exp(pos_score / self.temperature)
        
        ttl_score = torch.matmul(view1_emb, view2_emb.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        
        loss = -torch.log(pos_score / ttl_score).mean()
        return loss

    def forward(self, base_out, perturbed_out1=None, perturbed_out2=None, batch_data=None):
        """
        batch_data: (users, pos_items, neg_items)
        """
        users, pos_items, neg_items = batch_data
        u_emb, i_emb = base_out
        
        # 1. Main BPR Loss
        loss_bpr = self._bpr_loss(u_emb[users], i_emb[pos_items], i_emb[neg_items])
        
        # 2. CL Loss (Optional)
        loss_cl = 0.0
        if perturbed_out1 is not None and perturbed_out2 is not None:
            u_view1, i_view1 = perturbed_out1
            u_view2, i_view2 = perturbed_out2
            
            loss_cl = self._info_nce_loss(u_view1[users], u_view2[users]) + \
                      self._info_nce_loss(i_view1[pos_items], i_view2[pos_items])
            
        return loss_bpr + self.lambda_val * loss_cl


# ==========================================
# 3. GNN Trainer
# ==========================================
class GNNTrainer:
    def __init__(self, config: dict, model: nn.Module, train_loader: DataLoader, device: torch.device):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.device = device
        
        # Loss Module
        self.criterion = SimGCLLoss(
            lambda_val=config['lambda'], 
            temperature=0.2
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config['lr'],
            epochs=config['epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        self.scaler = torch.amp.GradScaler('cuda')
        self.checkpoint_dir = config['checkpoint_dir']
        self.final_save_path = os.path.join(config['cache_dir'], "simgcl_trained.pth")
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        total_loss = 0
        steps = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx}/{self.config['epochs']}")
        
        for batch_idx, (batch_users, batch_pos_items) in enumerate(pbar):
            batch_users = batch_users.to(self.device)
            batch_pos_items = batch_pos_items.to(self.device)
            batch_neg_items = torch.randint(0, self.config['num_items'], (len(batch_users),), device=self.device)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # A. Base Forward
                base_out = self.model(perturbed=False)
                
                # B. CL Views (Conditional)
                pert_out1, pert_out2 = None, None
                if batch_idx % self.config['cl_interval'] == 0:
                    pert_out1 = self.model(perturbed=True)
                    pert_out2 = self.model(perturbed=True)
                
                # C. Calculate Loss
                loss = self.criterion(
                    base_out=base_out,
                    perturbed_out1=pert_out1,
                    perturbed_out2=pert_out2,
                    batch_data=(batch_users, batch_pos_items, batch_neg_items)
                )

            # Backward & Step
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'avg': f"{avg_loss:.4f}"})
            
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
                
        return total_loss / steps

    def save_checkpoint(self, epoch_idx, avg_loss):
        checkpoint = {
            'epoch': epoch_idx,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': avg_loss,
            'config': self.config
        }
        save_path = os.path.join(self.checkpoint_dir, f"gnn_epoch_{epoch_idx}.pth")
        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint saved: {save_path}")

    def run(self):
        print(f"\n[Training Start] Users: {self.config['num_users']}, Items: {self.config['num_items']}")
        
        for epoch in range(1, self.config['epochs'] + 1):
            avg_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}")
            self.save_checkpoint(epoch, avg_loss)
            
        torch.save(self.model.state_dict(), self.final_save_path)
        print(f"🎉 Final Model saved to {self.final_save_path}")


# ==========================================
# 4. Main Execution Function
# ==========================================
def train_gnn_cl_user_noise():
    # -----------------------------------------------------------
    # 1. Configuration
    # -----------------------------------------------------------
    BASE_DIR = r"D:\trainDataset\localprops"
    
    config = {
        'json_file_path': os.path.join(BASE_DIR, "final_train_seq.json"),
        'cache_dir': os.path.join(BASE_DIR, "cache"),
        'checkpoint_dir': "./checkpoints",
        'batch_size': 10240,    # VRAM 허용 범위 내 최대
        'epochs': 15,           # 10~15 추천
        'lr': 0.005,
        'weight_decay': 1e-4,
        'embed_dim': 64,        # 64차원 (User Tower에서 Projection 예정)
        'n_layers': 2,
        'eps': 0.1,
        'cl_interval': 5,       # 매 배치마다 CL 수행 (성능 최우선)
        'lambda': 0.2,          # CL 비중
        'num_users': 0,         # 로드 후 업데이트
        'num_items': 0
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")

    # -----------------------------------------------------------
    # 2. Data Loading
    # -----------------------------------------------------------
    edge_index, n_users, n_items, u_map, i_map = load_and_process_data(
        config['json_file_path'], 
        config['cache_dir']
    )
    
    config['num_users'] = n_users
    config['num_items'] = n_items
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # -----------------------------------------------------------
    # 3. Dataset Setup
    # -----------------------------------------------------------
    graph_dataset = GraphDataset(n_users, n_items, edge_index, device)
    train_ds = TensorDataset(edge_index[0], edge_index[1])
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        drop_last=True, 
        pin_memory=True
    )

    # -----------------------------------------------------------
    # 4. Model & Trainer Setup
    # -----------------------------------------------------------
    model = SimGCL(
        graph_dataset, 
        embed_dim=config['embed_dim'], 
        n_layers=config['n_layers'], 
        eps=config['eps']
    ).to(device)
    
    trainer = GNNTrainer(config, model, train_loader, device)
    
    # -----------------------------------------------------------
    # 5. Run Training
    # -----------------------------------------------------------
    trainer.run()






# ==========================================
# 5. Resume Execution Function (추가됨)
# ==========================================
def resume_gnn_cl_user_noise(checkpoint_filename):
    """
    체크포인트 파일에서 모델과 옵티마이저 상태를 로드하여 학습을 재개합니다.
    Args:
        checkpoint_filename: 예) "gnn_epoch_5.pth"
    """
    # -----------------------------------------------------------
    # 1. Base Setup (경로 설정)
    # -----------------------------------------------------------
    BASE_DIR = r"D:\trainDataset\localprops"
    checkpoint_dir = "./checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
        return

    print(f"🔄 Loading checkpoint configuration from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    
    # 저장된 Config 불러오기 (경로는 현재 환경에 맞게 재설정)
    config = checkpoint['config']
    config['json_file_path'] = os.path.join(BASE_DIR, "final_train_seq.json")
    config['cache_dir'] = os.path.join(BASE_DIR, "cache")
    config['checkpoint_dir'] = checkpoint_dir
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")

    # -----------------------------------------------------------
    # 2. Data Loading (모델 초기화를 위해 필수)
    # -----------------------------------------------------------
    # 그래프 구조(Adjacency Matrix)를 다시 만들어야 모델을 올릴 수 있음
    edge_index, n_users, n_items, u_map, i_map = load_and_process_data(
        config['json_file_path'], 
        config['cache_dir']
    )
    
    # Config에 User/Item 수 동기화
    config['num_users'] = n_users
    config['num_items'] = n_items
    
    gc.collect()
    torch.cuda.empty_cache()

    # -----------------------------------------------------------
    # 3. Dataset & Loader Setup
    # -----------------------------------------------------------
    graph_dataset = GraphDataset(n_users, n_items, edge_index, device)
    train_ds = TensorDataset(edge_index[0], edge_index[1])
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        drop_last=True, 
        pin_memory=True
    )

    # -----------------------------------------------------------
    # 4. Model & Trainer Initialization
    # -----------------------------------------------------------
    model = SimGCL(
        graph_dataset, 
        embed_dim=config['embed_dim'], 
        n_layers=config['n_layers'], 
        eps=config['eps']
    ).to(device)
    
    trainer = GNNTrainer(config, model, train_loader, device)

    # -----------------------------------------------------------
    # 5. Load State Dicts (핵심: 상태 복원)
    # -----------------------------------------------------------
    model.load_state_dict(checkpoint['model_state_dict'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 이전 학습 종료 지점 확인
    start_epoch = checkpoint['epoch'] + 1
    prev_loss = checkpoint['loss']
    
    print(f"✅ Successfully loaded checkpoint '{checkpoint_filename}'")
    print(f"   -> Resuming from Epoch {start_epoch} (Previous Loss: {prev_loss:.4f})")
    
    # -----------------------------------------------------------
    # 6. Run Remaining Epochs
    # -----------------------------------------------------------
    if start_epoch > config['epochs']:
        print("⚠️ Training already finished based on config epochs.")
    else:
        trainer.run(start_epoch=start_epoch)


if __name__ == "__main__":
    train_gnn_cl_user_noise()