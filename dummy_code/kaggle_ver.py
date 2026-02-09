import os
import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ==========================================
# 0. Configuration & Paths
# ==========================================
# Kaggle 경로 설정
INPUT_DIR = "/kaggle/input/my-simgcl-data"  # 읽기 전용
OUTPUT_DIR = "/kaggle/working"              # 쓰기 가능
import gc
with torch.no_grad():
    torch.cuda.empty_cache()
gc.collect()

print(f"🧹 Memory Cleared. Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
config = {
    'json_file_path': os.path.join(INPUT_DIR, "final_train_seq.json"),
    'cache_dir': os.path.join(OUTPUT_DIR, "cache"),
    'checkpoint_dir': os.path.join(OUTPUT_DIR, "checkpoints"),
    
    # [최적화 수정] VRAM 안전성 + 학습 효과 동시 확보
    'batch_size': 8192,      # 물리 배치 (OOM 방지)
    'accumulation_steps': 2, # 4096 * 2 = 8192 (논리 배치 유지)
    
    'epochs': 5,            # 배치가 크므로 에폭을 늘려야 충분히 학습됨
    'lr': 0.0005,             # 128차원 안정성
    'weight_decay': 1e-5,
    
    'embed_dim': 128,        # 목표 차원
    'n_layers': 2,
    'eps': 0.2,              # 노이즈 강화
    
    'cl_interval': 2,        # 매 스텝 CL (필수)
    'lambda': 0.7,           # CL 비중 강화
    
    # LogQ & Learnable Temp
    'lambda_logq': 0.5,      # 0.5는 너무 셀 수 있음 -> 0.3 시작 추천
    'init_temp': 0.07,        # 0.1은 너무 Sharp함 -> 0.2로 시작해서 줄어들게 유도
}

# 폴더 생성
if not os.path.exists(config['cache_dir']): os.makedirs(config['cache_dir'])
if not os.path.exists(config['checkpoint_dir']): os.makedirs(config['checkpoint_dir'])

# ==========================================
# 1. Utilities (Data & LogQ)
# ==========================================
def calculate_logq_from_edge_index(edge_index, num_items, cache_dir, device):
    cache_path = os.path.join(cache_dir, "item_logq_pop.pt")
    if os.path.exists(cache_path):
        print(f"[Cache Hit] Loading LogQ from {cache_path}")
        return torch.load(cache_path, map_location=device)
    
    print("⚡ Calculating Item Popularity (LogQ)...")
    items = edge_index[1] # edge_index[1] contains raw item IDs (0 ~ Ni-1)
    item_counts = torch.bincount(items, minlength=num_items).float()
    probs = (item_counts + 1e-6) / item_counts.sum()
    log_q = torch.log(probs)
    torch.save(log_q, cache_path)
    return log_q.to(device)

def load_and_process_data(json_file_path, cache_dir):
    # (기존 코드와 동일하지만 간소화)
    cache_path = os.path.join(cache_dir, "processed_graph_train.pt")
    map_path = os.path.join(cache_dir, "id_maps_train.pt")

    if os.path.exists(cache_path) and os.path.exists(map_path):
        print(f"[Cache Hit] Loading graph data...")
        data = torch.load(cache_path)
        maps = torch.load(map_path)
        return data['edge_index'], data['num_users'], data['num_items'], maps['user2id'], maps['item2id']

    print(f"[Cache Miss] Processing {json_file_path}...")
    with open(json_file_path, 'r') as f: data = json.load(f)
    
    # ID Mapping
    users = list(data.keys())
    user2id = {u: i for i, u in enumerate(users)}
    items = set(i for l in data.values() for i in l)
    item2id = {i: idx for idx, i in enumerate(items)}
    
    src, dst = [], []
    for u, i_list in tqdm(data.items()):
        if u not in user2id: continue
        uid = user2id[u]
        for i in i_list:
            if i in item2id:
                src.append(uid)
                dst.append(item2id[i])
                
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # 중복 제거 등은 생략 가능하나 안전을 위해
    edge_index = torch.unique(edge_index, dim=1) 
    
    num_users, num_items = len(user2id), len(item2id)
    print(f" -> Users: {num_users}, Items: {num_items}, Edges: {edge_index.size(1)}")
    
    torch.save({'edge_index': edge_index, 'num_users': num_users, 'num_items': num_items}, cache_path)
    torch.save({'user2id': user2id, 'item2id': item2id}, map_path)
    return edge_index, num_users, num_items, user2id, item2id

class GraphDataset:
    def __init__(self, num_users, num_items, edge_index, device):
        self.num_users = num_users
        self.num_items = num_items
        self.device = device
        self.Graph = self._get_sparse_graph(edge_index.to(device))

    def _get_sparse_graph(self, edge_index):
        n_nodes = self.num_users + self.num_items
        row = torch.cat([edge_index[0], edge_index[1] + self.num_users])
        col = torch.cat([edge_index[1] + self.num_users, edge_index[0]])
        
        vals = torch.ones(row.size(0), device=self.device)
        indices = torch.stack([row, col], dim=0)
        
        deg = torch.zeros(n_nodes, device=self.device).scatter_add(0, row, vals)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        norm_vals = vals * deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return torch.sparse_coo_tensor(indices, norm_vals, (n_nodes, n_nodes))

# ==========================================
# 2. Model & Loss
# ==========================================
class SimGCL(nn.Module):
    def __init__(self, dataset, embed_dim, n_layers, eps):
        super().__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.Graph = dataset.Graph
        self.n_layers = n_layers
        self.eps = eps
        self.emb_u = nn.Embedding(self.num_users, embed_dim)
        self.emb_i = nn.Embedding(self.num_items, embed_dim)
        nn.init.xavier_uniform_(self.emb_u.weight)
        nn.init.xavier_uniform_(self.emb_i.weight)

    def forward(self, perturbed=False):
        ego = torch.cat([self.emb_u.weight, self.emb_i.weight], dim=0)
        if perturbed:
            noise = F.normalize(torch.rand_like(ego), dim=1)
            ego += self.eps * noise
            
        all_embs = [ego]
        for _ in range(self.n_layers):
            # Sparse MM (FP32 forced)
            with torch.amp.autocast(device_type='cuda', enabled=False):
                ego = torch.sparse.mm(self.Graph, ego.float())
            all_embs.append(ego)
            
        final = torch.stack(all_embs, dim=1).mean(dim=1)
        return torch.split(final, [self.num_users, self.num_items])

class SimGCLLoss(nn.Module):
    def __init__(self, lambda_val, init_temp, lambda_logq, log_q):
        super().__init__()
        self.lambda_val = lambda_val
        self.lambda_logq = lambda_logq
        
        # ⭐ Learnable Temperature (Optimizer에 등록됨)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temp))


        self.min_temp = 0.01  # 하한선 (0에 가까워지면 폭발하므로 방지)
        self.max_temp = 0.10  # ⭐ 상한선 (이 이상 절대 안 올라감)
        # ⭐ LogQ Correction
        if log_q is not None:
            self.register_buffer('log_q', log_q)
        else:
            self.log_q = None

    def get_current_temp(self):
        temp = 1.0 / self.logit_scale.exp()
        
        # 2. ⭐ Clamp 적용: (0.01 <= temp <= 0.15)
        return temp.clamp(min=self.min_temp, max=self.max_temp)

    def forward(self, base, pert1, pert2, batch_data):
        users, pos, neg = batch_data
        u_base, i_base = base
        
        # 1. BPR Loss (with LogQ)
        pos_scores = (u_base[users] * i_base[pos]).sum(1)
        neg_scores = (u_base[users] * i_base[neg]).sum(1)
        
        if self.lambda_logq > 0 and self.log_q is not None:
            pos_scores -= self.lambda_logq * self.log_q[pos]
            neg_scores -= self.lambda_logq * self.log_q[neg]
            
        loss_bpr = F.softplus(-(pos_scores - neg_scores)).mean()
        
        # 2. CL Loss
        loss_cl = 0.0
        if pert1 and pert2:
            curr_temp = self.get_current_temp()
            def info_nce(v1, v2):
                v1, v2 = F.normalize(v1, dim=1), F.normalize(v2, dim=1)
                pos = torch.exp((v1 * v2).sum(1) / curr_temp)
                ttl = torch.exp(torch.matmul(v1, v2.T) / curr_temp).sum(1)
                return -torch.log(pos / ttl).mean()
            
            u1, i1 = pert1
            u2, i2 = pert2
            loss_cl = info_nce(u1[users], u2[users]) + info_nce(i1[pos], i2[pos])
            
        return loss_bpr + self.lambda_val * loss_cl

# ==========================================
# 3. Trainer
# ==========================================
class GNNTrainer:
    def __init__(self, config, model, train_loader, log_q, device):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.device = device
        
        self.criterion = SimGCLLoss(
            config['lambda'], config['init_temp'], 
            config.get('lambda_logq', 0), log_q
        ).to(device)
        
        # ⭐ 파라미터 등록 확인 (Learnable Temp 포함)
        all_params = list(model.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=config['lr'], weight_decay=config['weight_decay'])
        self.scaler = torch.amp.GradScaler('cuda')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        accum_steps = self.config['accumulation_steps']
        self.optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch}")
        for idx, (batch_u, batch_pos) in enumerate(pbar):
            batch_u, batch_pos = batch_u.to(self.device), batch_pos.to(self.device)
            batch_neg = torch.randint(0, self.config['num_items'], (len(batch_u),), device=self.device)
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                base = self.model()
                pert1 = self.model(True) if idx % self.config['cl_interval'] == 0 else None
                pert2 = self.model(True) if idx % self.config['cl_interval'] == 0 else None
                
                loss = self.criterion(base, pert1, pert2, (batch_u, batch_pos, batch_neg))
                loss = loss / accum_steps
            
            self.scaler.scale(loss).backward()
            
            if (idx + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                
            total_loss += loss.item() * accum_steps
            
            if idx % 20 == 0:
                # ⭐ 지표 계산 함수 호출
                # base[0]: 전체 유저 임베딩 / base[1]: 전체 아이템 임베딩
                align, unif = self._calc_batch_metrics(base[0], base[1], batch_u, batch_pos)
                
                # 현재 온도 가져오기
                temp = self.criterion.get_current_temp().item()
                
                # pbar 업데이트 (Ali, Uni 추가)
                pbar.set_postfix({
                    'L': f"{loss.item()*accum_steps:.3f}", 
                    'Ali': f"{align:.2f}", # < 1.0 이면 좋음
                    'Uni': f"{unif:.2f}",  # -2.0 ~ -3.0 이면 좋음
                    'T': f"{temp:.3f}"
                })
        return total_loss / len(self.train_loader)
    def _calc_batch_metrics(self, u_emb, i_emb, users, pos_items):
        """
        현재 배치의 Alignment와 Uniformity를 계산 (모니터링용)
        """
        with torch.no_grad():
            # 1. Normalize (지표 계산 전 필수)
            u_norm = F.normalize(u_emb[users], dim=1)
            i_norm = F.normalize(i_emb[pos_items], dim=1)
            
            # 2. Alignment: (User - PosItem) 거리의 제곱 평균
            # 낮을수록 좋음 (0에 가까울수록 매칭 잘됨)
            align = (u_norm - i_norm).norm(p=2, dim=1).pow(2).mean().item()
            
            # 3. Uniformity: Item 간 거리의 분포
            # 낮을수록 좋음 (음수 값, 보통 -2.0 ~ -3.0)
            # 전체 아이템 다 하면 느리니까, 현재 배치 내의 Positive Item들끼리만 계산 (근사)
            if len(i_norm) > 2048:
                idx = torch.randperm(len(i_norm))[:2048]
                i_sample = i_norm[idx]
            else:
                i_sample = i_norm
                
            dist = torch.cdist(i_sample, i_sample, p=2).pow(2)
            unif = torch.log(torch.exp(-2 * dist).mean()).item()
            
        return align, unif
    def run(self):
        for epoch in range(1, self.config['epochs'] + 1):
            loss = self.train_epoch(epoch)
            print(f"Epoch {epoch}: {loss:.4f}")
            
            # Checkpoint
            torch.save(self.model.state_dict(), os.path.join(self.config['checkpoint_dir'], f"ep_{epoch}.pth"))
        
        final_path = os.path.join(self.config['cache_dir'], "simgcl_final.pth")
        torch.save(self.model.state_dict(), final_path)
        print(f"🎉 Saved to {final_path}")

# ==========================================
# 4. Run
# ==========================================
def train_gnn_cl_user_noise():
    print(f"🔧 Config: Batch {config['batch_size']} x {config['accumulation_steps']} | Dim {config['embed_dim']}")
    device = torch.device("cuda")
    
    # Data Load
    edge_index, nu, ni, um, im = load_and_process_data(config['json_file_path'], config['cache_dir'])
    config['num_users'], config['num_items'] = nu, ni
    
    # LogQ Calculation
    log_q = calculate_logq_from_edge_index(edge_index, ni, config['cache_dir'], device)
    
    # Dataset & Loader (Worker 추가!)
    graph_ds = GraphDataset(nu, ni, edge_index, device)
    train_ds = TensorDataset(edge_index[0], edge_index[1])
    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True, drop_last=True,
        pin_memory=True # ⭐ CPU 병목 해결
    )
    
    # Model & Run
    model = SimGCL(graph_ds, config['embed_dim'], config['n_layers'], config['eps']).to(device)
    trainer = GNNTrainer(config, model, train_loader, log_q, device)
    trainer.run()

if __name__ == "__main__":
    train_gnn_cl_user_noise()