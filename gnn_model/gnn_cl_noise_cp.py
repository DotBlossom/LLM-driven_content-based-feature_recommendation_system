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

def calculate_logq_from_edge_index(edge_index, num_items, cache_dir, device):
    """
    GNN 학습 데이터(Edge Index)에서 아이템 빈도를 계산하여 LogQ 텐서 생성
    """
    cache_path = os.path.join(cache_dir, "item_logq_pop.pt")
    
    if os.path.exists(cache_path):
        print(f"[Cache Hit] Loading LogQ from {cache_path}")
        return torch.load(cache_path, map_location=device)
    
    print("⚡ Calculating Item Popularity (LogQ) for GNN...")
    
    # 1. 아이템 빈도 카운트 (edge_index[1]이 아이템 인덱스라고 가정)
    # GNN에서 보통 item index는 0부터 시작하지만, 
    # GraphDataset에서 user 다음에 item이 오도록 reindex 되었다면 주의 필요.
    # 여기서는 load_and_process_data가 반환한 raw item id (0 ~ num_items-1) 기준입니다.
    
    items = edge_index[1]
    # bincount는 1D 텐서의 각 값의 빈도를 셈
    # minlength를 num_items로 설정하여 안 나온 아이템도 0으로 잡힘
    item_counts = torch.bincount(items, minlength=num_items).float()
    
    # 2. 확률 변환 (Smoothing)
    total_count = item_counts.sum()
    probs = (item_counts + 1e-6) / total_count # Divide by zero 방지
    
    # 3. Log 계산
    log_q = torch.log(probs)
    
    # 4. 저장 및 반환
    torch.save(log_q, cache_path)
    print(f"✅ LogQ Calculated & Saved. Shape: {log_q.shape}")
    
    return log_q.to(device)
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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimGCLLoss(nn.Module):
    def __init__(self, lambda_val=0.2, init_temp=0.2, lambda_logq=0.0, log_q=None):
        super(SimGCLLoss, self).__init__()
        self.lambda_val = lambda_val
        self.lambda_logq = lambda_logq
        
        # Learnable Temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temp))
        
        # LogQ Tensor 등록 (Buffer로 등록하여 업데이트 되지 않게 함)
        if log_q is not None:
            self.register_buffer('log_q', log_q)
        else:
            self.log_q = None

    def get_current_temp(self):
        return (1.0 / self.logit_scale.exp()).clamp(0.01, 1.0)

    def _bpr_loss_with_logq(self, users_emb, pos_items_emb, neg_items_emb, pos_idx, neg_idx):
        # 1. 내적 점수 계산
        pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)
        
        # 2. LogQ Correction 적용 (Popularity Bias 제거)
        if self.lambda_logq > 0.0 and self.log_q is not None:
            # 해당 아이템들의 Log 확률 가져오기
            pos_pop = self.log_q[pos_idx]
            neg_pop = self.log_q[neg_idx]
            
            # 점수 보정: Score_new = Score_old - lambda * log(P(i))
            # 인기 아이템일수록 점수를 깎음 -> Hard Negative 효과
            pos_scores = pos_scores - (self.lambda_logq * pos_pop)
            neg_scores = neg_scores - (self.lambda_logq * neg_pop)

        # 3. BPR Loss: -log(sigmoid(pos - neg)) = softplus(-(pos - neg))
        loss = F.softplus(-(pos_scores - neg_scores))
        return torch.mean(loss)

    def _info_nce_loss(self, view1, view2):
        curr_temp = self.get_current_temp()
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        pos_score = torch.sum(view1 * view2, dim=1)
        pos_score = torch.exp(pos_score / curr_temp)
        
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / curr_temp).sum(dim=1)
        
        return -torch.log(pos_score / ttl_score).mean()

    def forward(self, base_out, perturbed_out1=None, perturbed_out2=None, batch_data=None):
        users, pos_items, neg_items = batch_data
        u_emb, i_emb = base_out
        
        # 1. BPR Loss (LogQ 적용)
        loss_bpr = self._bpr_loss_with_logq(
            u_emb[users], i_emb[pos_items], i_emb[neg_items], 
            pos_items, neg_items # 인덱스 전달 필요
        )
        
        # 2. CL Loss
        loss_cl = 0.0
        if perturbed_out1 is not None:
            u_v1, i_v1 = perturbed_out1
            u_v2, i_v2 = perturbed_out2
            loss_cl = self._info_nce_loss(u_v1[users], u_v2[users]) + \
                      self._info_nce_loss(i_v1[pos_items], i_v2[pos_items])
            
        return loss_bpr + self.lambda_val * loss_cl
# ==========================================
# 3. GNN Trainer
# ==========================================
class GNNTrainer:
    def __init__(self, config, model, train_loader, log_q_tensor, device):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.device = device
        
        # Loss 초기화 (LogQ 전달)
        self.criterion = SimGCLLoss(
            lambda_val=config['lambda'], 
            init_temp=0.2,
            lambda_logq=config.get('lambda_logq', 0.0), # Config에서 받기
            log_q=log_q_tensor
        ).to(device)
        
        # Optimizer (Loss 파라미터 포함)
        all_params = list(model.parameters()) + list(self.criterion.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=config['lr'], weight_decay=config['weight_decay'])
        # ... (Scheduler 등 기존 동일) ...
        self.scaler = torch.amp.GradScaler('cuda')

    def _calc_batch_metrics(self, u_emb, i_emb, users, pos_items):
        """
        현재 배치를 이용하여 Alignment와 Uniformity를 근사 계산 (Fast)
        """
        with torch.no_grad():
            # Normalize
            u_norm = F.normalize(u_emb[users], dim=1)
            i_norm = F.normalize(i_emb[pos_items], dim=1)
            
            # 1. Alignment: (u - i)^2
            align = (u_norm - i_norm).norm(p=2, dim=1).pow(2).mean().item()
            
            # 2. Uniformity: exp(-2 * dist^2)
            # 배치 내 아이템들끼리의 분포만 확인 (전체 근사)
            # pdist 계산 비용이 크므로 2048개까지만 샘플링
            if len(i_norm) > 2048:
                idx = torch.randperm(len(i_norm))[:2048]
                i_sample = i_norm[idx]
            else:
                i_sample = i_norm
                
            dist = torch.cdist(i_sample, i_sample, p=2).pow(2)
            unif = torch.log(torch.exp(-2 * dist).mean()).item()
            
        return align, unif

    def train_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0
        
        # -------------------------------------------------------
        # ⭐ [최적화 1] Gradient Accumulation 설정
        # 물리 배치는 1024지만, 논리 배치는 4096으로 학습 효과를 냄
        # -------------------------------------------------------
        accumulation_steps = 4  # 1024 * 4 = 4096 (권장)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx}")
        
        # Optimizer 초기화 (루프 시작 전)
        self.optimizer.zero_grad(set_to_none=True) # set_to_none=True가 더 빠름
        
        for batch_idx, (batch_users, batch_pos_items) in enumerate(pbar):
            batch_users = batch_users.to(self.device)
            batch_pos_items = batch_pos_items.to(self.device)
            
            # Negatives Sampling
            batch_neg_items = torch.randint(0, self.config['num_items'], (len(batch_users),), device=self.device)
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # 1. Forward
                base_out = self.model(perturbed=False)
                
                # 2. CL Views (매 스텝 수행하되, 메모리 아끼려면 여기서도 조절 가능)
                pert_out1, pert_out2 = None, None
                if batch_idx % self.config['cl_interval'] == 0:
                    pert_out1 = self.model(perturbed=True)
                    pert_out2 = self.model(perturbed=True)
                
                # 3. Loss Calculation
                loss = self.criterion(
                    base_out, pert_out1, pert_out2, 
                    (batch_users, batch_pos_items, batch_neg_items)
                )
                
                # ⭐ Loss 정규화 (Accumulation을 위해 나눠줌)
                loss = loss / accumulation_steps

            # 4. Backward (Gradient 누적됨)
            self.scaler.scale(loss).backward()
            
            # 5. Step (accumulation_steps 마다 실행)
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient Clipping (학습 안정성 확보)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # 5.0 -> 1.0으로 더 빡빡하게
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True) # 메모리 최적화
                
                # Scheduler는 Step 단위 업데이트인 경우 여기서 호출
                # self.scheduler.step() 

            total_loss += loss.item() * accumulation_steps # 로깅용 복원
            
            if batch_idx % 100 == 0:
                # (지표 모니터링 코드는 그대로 유지)
                pass
        
        return total_loss / len(self.train_loader)
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
        'batch_size': 1024,      # 물리적 한계 (유지)
        'accumulation_steps': 4, # ⭐ 추가 (논리적 배치 4096 효과)
        
        'epochs': 15,            # 차원이 커져서 수렴이 느릴 수 있음
        'lr': 0.001,             # ⭐ 0.005 -> 0.001 (안정성 확보)
        'weight_decay': 1e-5,    # ⭐ 1e-4 -> 1e-5 (제약 완화)
        
        'embed_dim': 128,
        'n_layers': 2,           # 3층으로 늘리면 오버스무딩 올 수 있으니 2층 유지
        'eps': 0.2,              # ⭐ 노이즈 0.1 -> 0.2 (더 강한 변형으로 강건성 확보)
        
        'cl_interval': 1,        # 유지
        'lambda': 0.5,           # ⭐ 0.2 -> 0.5 (CL 강화) 배치 작아져서 밀어내는거 일단 좀 초기에 빡
        
        # 아까 구현한 LogQ / Learnable Temp 적용 필수
        'lambda_logq': 0.2,     
        'init_temp': 0.1,        # ⭐ 0.2 -> 0.1 (좀 더 Sharp하게 시작)
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
    
        # --------------------------------------------------------
    # ⭐ [추가] LogQ Tensor 계산
    # --------------------------------------------------------
    log_q_tensor = calculate_logq_from_edge_index(
        edge_index, n_items, config['cache_dir'], device
    )

    # ... (Dataset, Model 초기화) ...

    # --------------------------------------------------------
    # ⭐ [수정] Trainer에 log_q 전달
    # --------------------------------------------------------
    trainer = GNNTrainer(config, model, train_loader, log_q_tensor, device)
    
    # -----------------------------------------------------------
    # 5. Run Training
    # -----------------------------------------------------------
    trainer.run()




import torch
import torch.nn.functional as F
import numpy as np

def calculate_alignment_uniformity(model, edge_index, batch_size=2048):
    """
    SimGCL 모델의 임베딩 품질(Alignment & Uniformity)을 측정합니다.
    """
    model.eval()
    
    # 1. 임베딩 추출 (Normalization 필수)
    with torch.no_grad():
        u_emb, i_emb = model(perturbed=False)
        u_emb = F.normalize(u_emb, dim=1)
        i_emb = F.normalize(i_emb, dim=1)
    
    # ---------------------------------------------------------
    # 1. Alignment Loss (User - Positive Item 거리)
    # : 유저와 그가 상호작용한 아이템은 가까워야 한다.
    # Formula: E[ || f(u) - f(i) ||^2 ]
    # ---------------------------------------------------------
    users = edge_index[0]
    items = edge_index[1]
    
    # 메모리 문제로 배치 단위 계산
    total_align_loss = 0
    num_edges = len(users)
    
    for i in range(0, num_edges, batch_size):
        batch_u = users[i:i+batch_size]
        batch_i = items[i:i+batch_size]
        
        u_vecs = u_emb[batch_u]
        i_vecs = i_emb[batch_i]
        
        # 유클리드 거리 제곱 (x-y)^2
        align_loss = (u_vecs - i_vecs).norm(p=2, dim=1).pow(2).mean()
        total_align_loss += align_loss.item() * len(batch_u)
        
    avg_align = total_align_loss / num_edges
    
    # ---------------------------------------------------------
    # 2. Uniformity Loss (All Items Distribution)
    # : 아이템들은 공간상에 고르게 퍼져 있어야 한다. (붕괴 방지)
    # Formula: log E[ exp( -2 * || f(i) - f(j) ||^2 ) ]
    # 전체 쌍(N*N)은 불가능하므로 랜덤 샘플링으로 근사
    # ---------------------------------------------------------
    num_samples = 5000 # 샘플링 개수
    perm = torch.randperm(len(i_emb))[:num_samples]
    sampled_items = i_emb[perm]
    
    # pdist: pairwise distance between all sampled items
    # (N, D) -> (N, N) distance matrix
    dist_matrix = torch.cdist(sampled_items, sampled_items, p=2).pow(2)
    
    # exp(-2 * dist) 계산 후 평균 -> 로그
    # t=2 (Wang et al. 논문 표준)
    unif_loss = torch.log(torch.exp(-2 * dist_matrix).mean()).item()
    
    return avg_align, unif_loss

# ==========================================
# 사용 예시
# ==========================================
# 학습이 끝난 모델(model)과 edge_index를 넣어주세요.
# align, unif = calculate_alignment_uniformity(model, edge_index.to(device))
# print(f"📊 Alignment: {align:.4f} (Low is Good, < 0.5 is great)")
# print(f"📊 Uniformity: {unif:.4f} (Low is Good, usually -1 ~ -3)")

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