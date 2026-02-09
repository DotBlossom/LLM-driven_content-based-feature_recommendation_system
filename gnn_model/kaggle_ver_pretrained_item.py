import os
import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import random

# ==========================================
# 0. Environment & Configuration
# ==========================================
# VRAM 파편화 방지
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Garbage Collection
with torch.no_grad():
    torch.cuda.empty_cache()
gc.collect()
print(f"🧹 Memory Cleared. Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

INPUT_DIR = "/kaggle/input/my-simgcl-data"
OUTPUT_DIR = "/kaggle/working"

config = {
    'json_file_path': os.path.join(INPUT_DIR, "final_train_seq.json"),
    'cache_dir': os.path.join(OUTPUT_DIR, "cache"),
    'checkpoint_dir': os.path.join(OUTPUT_DIR, "checkpoints"),
    
    'batch_size': 4096,
    'accumulation_steps': 1,
    'epochs': 5,
    'lr': 0.0005,
    'weight_decay': 1e-5,
    
    'embed_dim': 64,    
    'n_layers': 2,
    'eps': 0.035,
    'cl_interval': 2,
    
    # Lambda Scheduling
    'lambda': 0.2,       # Target Lambda
    'lambda_logq': 0.0,
    'init_temp': 0.07,
    'warmup_epochs': 0,  # 1 Epoch 동안 CL 끔
    'ramp_epochs': 2,    # 2 Epoch 동안 서서히 증가
}

# 폴더 생성
import shutil


# ==========================================
# 1. Utilities (Data Processing)
# ==========================================
def load_and_process_data(json_file_path, cache_dir):
    cache_path = os.path.join(cache_dir, "processed_graph_train.pt")
    map_path = os.path.join(cache_dir, "id_maps_train.pt")

    if os.path.exists(cache_path) and os.path.exists(map_path):
        print(f"[Cache Hit] Loading graph data...")
        data = torch.load(cache_path)
        maps = torch.load(map_path)
        return data['edge_index'], data['num_users'], data['num_items'], maps['user2id'], maps['item2id']

    print(f"[Cache Miss] Processing {json_file_path}...")
    with open(json_file_path, 'r') as f: 
        data = json.load(f)
    
    # [Sorted Ensure Determinism]
    users = sorted(list(data.keys()))
    user2id = {u: i for i, u in enumerate(users)}
    
    all_items = set()
    for item_list in data.values():
        all_items.update(item_list)
    items = sorted(list(all_items))
    item2id = {i: idx for idx, i in enumerate(items)}
    
    src, dst = [], []
    for u, i_list in tqdm(data.items(), desc="Building Graph"):
        if u not in user2id: continue
        uid = user2id[u]
        for i in i_list:
            if i in item2id:
                src.append(uid)
                dst.append(item2id[i])
                
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1) 
    
    num_users, num_items = len(user2id), len(item2id)
    print(f" -> Users: {num_users}, Items: {num_items}, Edges: {edge_index.size(1)}")
    
    torch.save({'edge_index': edge_index, 'num_users': num_users, 'num_items': num_items}, cache_path)
    torch.save({'user2id': user2id, 'item2id': item2id}, map_path)
    
    return edge_index, num_users, num_items, user2id, item2id

def calculate_logq_from_edge_index(edge_index, num_items, cache_dir, device):
    cache_path = os.path.join(cache_dir, "item_logq_pop.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location=device)
    
    print("⚡ Calculating Item Popularity (LogQ)...")
    items = edge_index[1]
    item_counts = torch.bincount(items, minlength=num_items).float()
    probs = (item_counts + 1e-6) / item_counts.sum()
    log_q = torch.log(probs)
    torch.save(log_q, cache_path)
    return log_q.to(device)

def load_pretrained_tensor(input_dir, item2id, target_device):
    print("\n📥 Loading Pretrained Embeddings...")
    emb_path = os.path.join(input_dir, "pretrained_item_matrix.pt")
    ids_path = os.path.join(input_dir, "item_ids.pt")

    if not os.path.exists(emb_path):
        print("⚠️ No pretrained file found. Using Random Init.")
        return None

    raw_emb = torch.load(emb_path, map_location='cpu')
    raw_ids = torch.load(ids_path, map_location='cpu')
    
    if isinstance(raw_emb, dict):
        raw_emb = raw_emb.get('weight', list(raw_emb.values())[0])

    input_dim = raw_emb.shape[1] # Usually 128
    num_items = len(item2id)
    
    sorted_features = torch.zeros((num_items, input_dim), dtype=torch.float32)
    
    raw_id_map = {}
    for idx, rid in enumerate(raw_ids):
        k = str(rid.item()) if isinstance(rid, torch.Tensor) else str(rid)
        raw_id_map[k] = idx
        
    matched = 0
    for item_str, model_idx in item2id.items():
        if item_str in raw_id_map:
            src_idx = raw_id_map[item_str]
            sorted_features[model_idx] = raw_emb[src_idx]
            matched += 1
            
    print(f"✅ Loaded Features: {matched}/{num_items} items aligned. (Dim: {input_dim})")
    
    if matched < num_items:
        print("   -> Filling missing items with small random noise.")
        mask = (sorted_features.sum(dim=1) == 0)
        sorted_features[mask] = torch.randn(mask.sum(), input_dim) * 0.01

    return sorted_features.to(target_device)

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
# 2. Model Architecture
# ==========================================
class ResidualAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        
        # [Main Path]
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        )
        
        # [Shortcut Path]
        self.shortcut = nn.Linear(input_dim, output_dim, bias=False)
        
        # Initialization
        nn.init.eye_(self.shortcut.weight) # Identity-like
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def forward(self, x):
        return self.mlp(x) + self.shortcut(x)

class SimGCL(nn.Module):
    def __init__(self, dataset, config, pretrained_features=None):
        super().__init__()
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.Graph = dataset.Graph
        self.n_layers = config['n_layers']
        self.eps = config['eps']
        self.embed_dim = config['embed_dim']
        
        # User Embedding
        self.embedding_user = nn.Embedding(self.num_users, self.embed_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight)

        # Item Embedding
        self.use_pretrained = False
        if pretrained_features is not None:
            self.item_features = nn.Parameter(pretrained_features, requires_grad=False)
            
            # Residual Adapter
            self.item_adapter = ResidualAdapter(pretrained_features.shape[1], self.embed_dim)
            
            # Reconstruction Decoder (64 -> 128)
            self.item_decoder = nn.Linear(self.embed_dim, pretrained_features.shape[1])
            
            self.item_bias = nn.Parameter(torch.zeros(self.num_items, self.embed_dim))
            nn.init.xavier_uniform_(self.item_bias)
            self.use_pretrained = True
        else:
            self.embedding_item = nn.Embedding(self.num_items, self.embed_dim)
            nn.init.xavier_uniform_(self.embedding_item.weight)

    def get_item_embedding(self):
        if self.use_pretrained:
            return self.item_adapter(self.item_features) + self.item_bias
        else:
            return self.embedding_item.weight

    def calculate_recon_loss(self):
        if not self.use_pretrained: return 0.0
        current_emb = self.item_adapter(self.item_features)
        recon_features = self.item_decoder(current_emb)
        loss = F.mse_loss(recon_features, self.item_features)
        return loss

    def forward(self, perturbed=False):
        u_emb = self.embedding_user.weight
        i_emb = self.get_item_embedding()
        
        ego = torch.cat([u_emb, i_emb], dim=0)
        if perturbed:
            noise = F.normalize(torch.rand_like(ego), dim=1)
            ego += self.eps * noise
            
        all_embs = [ego]
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(self.Graph, ego)
            all_embs.append(ego)
            
        final = torch.stack(all_embs, dim=1).mean(dim=1)
        return torch.split(final, [self.num_users, self.num_items])

class SimGCLLoss(nn.Module):
    def __init__(self, init_temp, lambda_logq, log_q):
        super().__init__()
        self.lambda_logq = lambda_logq
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temp))
        if log_q is not None:
            self.register_buffer('log_q', log_q)
        else:
            self.log_q = None

    def get_current_temp(self):
        return (1.0 / self.logit_scale.exp()).clamp(min=0.01, max=0.10)

    def forward(self, base, pert1, pert2, batch_data, current_lambda):
        users, pos, neg = batch_data
        u_base, i_base = base
        
        pos_scores = (u_base[users] * i_base[pos]).sum(1)
        neg_scores = (u_base[users] * i_base[neg]).sum(1)
        
        if self.lambda_logq > 0 and self.log_q is not None:
            pos_scores -= self.lambda_logq * self.log_q[pos]
            neg_scores -= self.lambda_logq * self.log_q[neg]
            
        loss_bpr = F.softplus(-(pos_scores - neg_scores)).mean()
        
        loss_cl = 0.0
        if current_lambda > 0 and pert1 and pert2:
            curr_temp = self.get_current_temp()
            def info_nce(v1, v2):
                v1, v2 = F.normalize(v1, dim=1), F.normalize(v2, dim=1)
                pos = torch.exp((v1 * v2).sum(1) / curr_temp)
                ttl = torch.exp(torch.matmul(v1, v2.T) / curr_temp).sum(1)
                return -torch.log(pos / ttl).mean()
            
            u1, i1 = pert1
            u2, i2 = pert2
            loss_cl = info_nce(u1[users], u2[users]) + info_nce(i1[pos], i2[pos])
            
        return loss_bpr + current_lambda * loss_cl

# ==========================================
# 3. Trainer (Refined with Checks)
# ==========================================
class GNNTrainer:
    def __init__(self, config, model, train_loader, log_q, device, user2id, item2id):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.user2id = user2id
        self.item2id = item2id
        self.history = []
        self.criterion = SimGCLLoss(
            config['init_temp'], config.get('lambda_logq', 0), log_q
        ).to(device)
        
        all_params = list(model.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=config['lr'], weight_decay=config['weight_decay'])

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        accum_steps = self.config['accumulation_steps']
        self.optimizer.zero_grad(set_to_none=True)
        
        # Lambda Ramping Logic
        target_lambda = self.config['lambda']
        warmup_eps = self.config.get('warmup_epochs', 1)
        ramp_eps = self.config.get('ramp_epochs', 2)
        lambda_recon = 0.1
        
        if epoch <= warmup_eps:
            current_lambda = 0.0
            print(f"🔥 [Warm-up] Epoch {epoch}: Lambda = 0.0 (CL Disabled)")
        elif epoch <= (warmup_eps + ramp_eps):
            progress = (epoch - warmup_eps) / ramp_eps
            current_lambda = target_lambda * progress
            print(f"📈 [Ramping] Epoch {epoch}: Lambda = {current_lambda:.4f} ({progress*100:.0f}%)")
        else:
            current_lambda = target_lambda
            print(f"✅ [Stable] Epoch {epoch}: Lambda = {current_lambda:.4f} (Max)")

        pbar = tqdm(self.train_loader, desc=f"Ep {epoch}")
        for idx, (batch_u, batch_pos) in enumerate(pbar):
            batch_u, batch_pos = batch_u.to(self.device), batch_pos.to(self.device)
            batch_neg = torch.randint(0, self.config['num_items'], (len(batch_u),), device=self.device)
            
            base = self.model()
            if idx == 0:
                print(f"\n🔍 [Debug] GNN Output Stats:")
                print(f"   User Emb Mean: {base[0].abs().mean().item():.6f} | Max: {base[0].max().item():.6f}")
                print(f"   Item Emb Mean: {base[1].abs().mean().item():.6f} | Max: {base[1].max().item():.6f}")
                if base[0].abs().mean().item() < 1e-6:
                    print("   ❌ CRITICAL: GNN Output is Zero! Graph propagation failed.")
            if current_lambda > 0 and idx % self.config['cl_interval'] == 0:
                pert1 = self.model(True)
                pert2 = self.model(True)
            else:
                pert1, pert2 = None, None
            
            loss_main = self.criterion(base, pert1, pert2, (batch_u, batch_pos, batch_neg), current_lambda)            
            loss_recon = self.model.calculate_recon_loss()
            loss = (loss_main + lambda_recon * loss_recon) / accum_steps
            
            loss.backward()
            
            if (idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                
            total_loss += loss.item() * accum_steps
            current_loss_val = loss.item() * accum_steps
            if idx % 20 == 0:
                align, unif = self._calc_batch_metrics(base[0], base[1], batch_u, batch_pos)
                temp = self.criterion.get_current_temp().item()
                pbar.set_postfix({
                    'L': f"{loss.item()*accum_steps:.3f}", 
                    'Ali': f"{align:.2f}", 
                    'Uni': f"{unif:.2f}", 
                    'Lam': f"{current_lambda:.2f}"
                })

                # 2. History 리스트에 저장 (나중에 분석용)
                log_entry = {
                    'epoch': epoch,
                    'step': idx,
                    'loss': round(current_loss_val, 4),
                    'align': round(align, 4),
                    'uni': round(unif, 4),
                    'lambda': round(current_lambda, 4),
                    'temp': round(temp, 4)
                }
                self.history.append(log_entry)
                
        return total_loss / len(self.train_loader)
        
    def _calc_batch_metrics(self, u_emb, i_emb, users, pos_items):
        with torch.no_grad():
            u_norm = F.normalize(u_emb[users], dim=1)
            i_norm = F.normalize(i_emb[pos_items], dim=1)
            align = (u_norm - i_norm).norm(p=2, dim=1).pow(2).mean().item()
            if len(i_norm) > 2048:
                idx = torch.randperm(len(i_norm))[:2048]
                i_sample = i_norm[idx]
            else:
                i_sample = i_norm
            dist = torch.cdist(i_sample, i_sample, p=2).pow(2)
            unif = torch.log(torch.exp(-2 * dist).mean()).item()
        return align, unif
        
    def run(self):
        if not os.path.exists(self.config['checkpoint_dir']):
            os.makedirs(self.config['checkpoint_dir'])
            print(f"📂 Created checkpoint directory: {self.config['checkpoint_dir']}")

        for epoch in range(1, self.config['epochs'] + 1):
            loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} Final Loss: {loss:.4f}")
            
            ckpt_name = f"model_ep{epoch}.pth"
            ckpt_path = os.path.join(self.config['checkpoint_dir'], ckpt_name)
            torch.save(self.model.state_dict(), ckpt_path)
            print(f"   └── 💾 Saved Checkpoint: {ckpt_name}")

        self.verify_integrity()


        log_path = os.path.join(self.config['checkpoint_dir'], "training_logs.json")
        with open(log_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"📊 Training Logs saved to {log_path}")
        
        
        final_path = os.path.join(self.config['checkpoint_dir'], "simgcl_final.pth")
        torch.save(self.model.state_dict(), final_path)
        
        map_path = os.path.join(self.config['cache_dir'], "id_maps_train.pt")
        if not os.path.exists(map_path):
            torch.save({'user2id': self.user2id, 'item2id': self.item2id}, map_path)
            
        print("\n🎉 All Training Finished & Saved!")

    def verify_integrity(self):
        print("\n🕵️ Verifying Integrity...")
        if hasattr(self.model, 'use_pretrained') and self.model.use_pretrained:
            print(f"   -> Mode: Adapter (Input {self.model.item_features.shape[1]})")
        try:
            with torch.no_grad():
                test_emb = self.model.get_item_embedding()
                # 차원 확인 (Main에서 64로 설정했으므로 확인)
                expected = self.config['embed_dim']
                if test_emb.shape[1] != expected:
                     raise ValueError(f"Output dim mismatch: {test_emb.shape[1]} != {expected}")
            print(f"✅ Forward Generation Check Passed (Output Dim: {test_emb.shape[1]})")
        except Exception as e:
            print(f"❌ Generation Failed: {e}")

# ==========================================
# 4. Main Execution
# ==========================================
def train_gnn_cl_user_noise():
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"🔧 Config: Batch {config['batch_size']} x {config['accumulation_steps']} | Dim {config['embed_dim']}")
    device = torch.device("cuda")
    
    # 1. Load Data
    edge_index, nu, ni, user2id, item2id = load_and_process_data(config['json_file_path'], config['cache_dir'])
    config['num_users'], config['num_items'] = nu, ni
    
    # 2. LogQ
    log_q = calculate_logq_from_edge_index(edge_index, ni, config['cache_dir'], device)
    
    # 3. Pretrained Tensor
    pretrained_features = load_pretrained_tensor(INPUT_DIR, item2id, device)
    
    # 4. Dataset
    graph_ds = GraphDataset(nu, ni, edge_index, device)
    train_ds = TensorDataset(edge_index[0], edge_index[1])
    train_loader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True, drop_last=True, pin_memory=True
    )
    
    # 5. Model
    model = SimGCL(graph_ds, config, pretrained_features=pretrained_features).to(device)

    # 6. Emergency Check (Before Trainer)
    if pretrained_features is not None:
        print("\n🚑 [Emergency Check] Pretrained Tensor Stats")
        print(f"   -> Max Value: {model.item_features.max().item()}")
        print(f"   -> Min Value: {model.item_features.min().item()}")
        print(f"   -> Mean Value: {model.item_features.mean().item()}")
        print(f"   -> Non-Zero Count: {torch.count_nonzero(model.item_features).item()}")

        with torch.no_grad():
            test_emb = model.get_item_embedding()
            print("\n🚑 [Emergency Check] Initial Item Embedding Stats (After Adapter)")
            print(f"   -> Max Value: {test_emb.max().item()}")
            print(f"   -> Mean Value: {test_emb.mean().item()}")
            print(f"   -> Std Dev: {test_emb.std().item()}")

    # 7. Trainer & Run
    trainer = GNNTrainer(config, model, train_loader, log_q, device, user2id, item2id)
    trainer.run()

    gc.collect()
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    train_gnn_cl_user_noise()