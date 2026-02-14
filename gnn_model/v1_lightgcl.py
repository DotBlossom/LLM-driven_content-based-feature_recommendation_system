import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import scipy.sparse as sp
import time
import os
import json
from tqdm import tqdm

# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================
def load_and_process_data(json_file_path, cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_path = os.path.join(cache_dir, "processed_graph_train.pt")
    map_path = os.path.join(cache_dir, "id_maps_train.pt")

    if os.path.exists(cache_path) and os.path.exists(map_path):
        print(f"[Data] Cache Hit! Loading graph data from {cache_dir}...")
        data_cache = torch.load(cache_path)
        maps_cache = torch.load(map_path)
        return (data_cache['edge_index'], data_cache['num_users'], data_cache['num_items'], 
                maps_cache['user2id'], maps_cache['item2id'])

    print(f"[Data] Cache Miss! Processing {json_file_path}...")
    with open(json_file_path, 'r') as f: 
        raw_data = json.load(f)
    
    users = sorted(list(raw_data.keys()))
    user2id = {u: i for i, u in enumerate(users)}
    
    all_items = set()
    for item_list in raw_data.values():
        all_items.update(item_list)
    items = sorted(list(all_items))
    item2id = {i: idx for idx, i in enumerate(items)}
    
    src, dst = [], []
    for u, i_list in tqdm(raw_data.items(), desc="Building Graph"):
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

# ==========================================
# 2. 데이터셋 클래스 (Optimized)
# ==========================================
class TrainDataset(data.Dataset):
    def __init__(self, edge_index, num_users, num_items):
        self.num_users = num_users
        self.num_items = num_items
        
        # Tensor로 저장
        self.users = edge_index[0]
        self.items = edge_index[1]
        
        print(f"\n[Dataset] Preparing Negative Sampling Sets for {num_users} Users...")
        self.user_pos_set = [set() for _ in range(num_users)]
        
        # CPU 연산 가속을 위해 numpy로 변환하여 순회
        src = self.users.numpy()
        dst = self.items.numpy()
        
        for u, i in tqdm(zip(src, dst), total=len(src), desc="Indexing Interactions"):
            self.user_pos_set[u].add(i)
            
        print("[Dataset] Ready!")

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        # [최적화] torch.tensor() 변환을 제거하고 int/long 타입을 그대로 반환
        # DataLoader의 collate_fn이 배치 단위로 묶을 때 한 번에 Tensor로 변환하므로 훨씬 빠름
        user = self.users[idx].item()
        pos_item = self.items[idx].item()
        
        neg_item = np.random.randint(0, self.num_items)
        while neg_item in self.user_pos_set[user]:
            neg_item = np.random.randint(0, self.num_items)
            
        return user, pos_item, neg_item

# ==========================================
# 3. 그래프 빌더
# ==========================================
def build_graph(edge_index, num_users, num_items, device, q=5):
    print("Building Graph & Calculating SVD...")
    start = time.time()

    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    
    user_nodes_R = src
    item_nodes_R = dst + num_users
    item_nodes_RT = dst + num_users
    user_nodes_RT = src
    
    rows = np.concatenate([user_nodes_R, item_nodes_RT])
    cols = np.concatenate([item_nodes_R, user_nodes_RT])
    data = np.ones(len(rows), dtype=np.float32)
    
    num_nodes = num_users + num_items
    adj_mat = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    
    rowsum = np.array(adj_mat.sum(axis=1)).flatten()
    d_inv = np.power(rowsum, -0.5)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    
    norm_adj = d_mat.dot(adj_mat).dot(d_mat)
    norm_adj = norm_adj.tocoo() 

    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data)
    shape = torch.Size(norm_adj.shape)
    
    adj_tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce().to(device)
    U, S, V = torch.svd_lowrank(adj_tensor, q=q, niter=2)
    
    print(f"Graph Built & SVD Done ({time.time() - start:.2f}s)")
    return adj_tensor, U, S, V

# ==========================================
# 4. 모델
# ==========================================
class LightGCL(nn.Module):
    def __init__(self, num_users, num_items, config, adj_tensor, svd_components):
        super(LightGCL, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = config['emb_dim']
        self.n_layers = config['n_layers']
        self.temp = config['temp']
        self.lambda_ssl = config['lambda_ssl']
        
        self.adj = adj_tensor
        self.U, self.S, self.V = svd_components 
        
        self.embedding_user = nn.Embedding(num_users, self.emb_dim)
        self.embedding_item = nn.Embedding(num_items, self.emb_dim)
        
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)

    def forward(self):
        all_emb = torch.cat([self.embedding_user.weight, self.embedding_item.weight])
        
        local_embs = [all_emb]
        x = all_emb
        for _ in range(self.n_layers):
            with torch.amp.autocast('cuda', enabled=False):
                x = x.float()
                x = torch.sparse.mm(self.adj, x)
            local_embs.append(x)
        local_final = torch.mean(torch.stack(local_embs, dim=1), dim=1)
        
        global_embs = [all_emb]
        x_g = all_emb
        for _ in range(self.n_layers):
            with torch.amp.autocast('cuda', enabled=False):
                x_g = x_g.float()
                temp = torch.matmul(self.V.t(), x_g)
                temp = temp * self.S.unsqueeze(1) 
                x_g = torch.matmul(self.U, temp)
            global_embs.append(x_g)
            
        global_final = torch.mean(torch.stack(global_embs, dim=1), dim=1)
        return local_final, global_final

    def calc_bpr_loss(self, local_emb, users, pos_items, neg_items):
        users_emb = local_emb[users]
        pos_emb = local_emb[pos_items]
        neg_emb = local_emb[neg_items]
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        return loss

    def calc_ssl_loss(self, local_emb, global_emb, users, items):
        local_emb_norm = F.normalize(local_emb, dim=1)
        global_emb_norm = F.normalize(global_emb, dim=1)
        unique_users = torch.unique(users)
        unique_items = torch.unique(items)
        
        def robust_info_nce(view1, view2, indices):
            v1 = view1[indices]
            v2 = view2[indices]
            logits = torch.matmul(v1, v2.t()) / self.temp
            logits = torch.clamp(logits, max=100.0) 
            labels = torch.arange(logits.shape[0]).to(logits.device)
            return F.cross_entropy(logits, labels)

        user_ssl_loss = robust_info_nce(local_emb_norm, global_emb_norm, unique_users)
        item_ssl_loss = robust_info_nce(local_emb_norm, global_emb_norm, unique_items)
        return user_ssl_loss + item_ssl_loss

    def get_l2_reg(self, users, pos_items, neg_items):
        reg_loss = (1/2)*(self.embedding_user.weight[users].norm(2).pow(2) + 
                          self.embedding_item.weight[pos_items].norm(2).pow(2) +
                          self.embedding_item.weight[neg_items].norm(2).pow(2))
        return reg_loss

# ==========================================
# 5. 학습 루프 (Refactored)
# ==========================================
def train(config, dataset, svd_components, edge_index_info):
    """
    이제 train 함수는 외부에서 데이터와 SVD 결과를 받아옵니다. (이중 로딩 방지)
    """
    edge_index, num_users, num_items = edge_index_info
    adj_tensor, U, S, V = svd_components
    
    # DataLoader 생성
    dataloader = data.DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    
    # 모델 초기화
    model = LightGCL(num_users, num_items, config, adj_tensor, (U, S, V)).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # AMP Scaler
    use_amp = (config['device'] == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    print(f"\n🚀 Start Training on {config['device']} | Batch Size: {config['batch_size']}")
    
    # SVD Stats 출력
    print("="*40)
    print(f"📊 [SVD Stats] Top-{config['svd_q']} Singular Values: {S.cpu().numpy()}") 
    print("="*40 + "\n")
    best_loss = float('inf') # 최고 기록 저장용
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        
        for batch_i, (users, pos_items, neg_items) in enumerate(pbar):
            users = users.to(config['device'])
            pos_items = pos_items.to(config['device'])
            neg_items = neg_items.to(config['device'])
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                local_emb, global_emb = model()
                
                bpr_loss = model.calc_bpr_loss(local_emb, users, pos_items + num_users, neg_items + num_users)
                ssl_loss = model.calc_ssl_loss(local_emb, global_emb, users, pos_items + num_users)
                reg_loss = model.get_l2_reg(users, pos_items, neg_items)
                
                loss = bpr_loss + config['lambda_ssl'] * ssl_loss + config['lambda_reg'] * reg_loss
                
                loss_val = loss.item()
                bpr_val = bpr_loss.item()
                ssl_val = ssl_loss.item()
            
            # [Sanity Check] 첫 에포크, 첫 배치일 때만 출력 (tqdm 깨짐 방지 위해 write 사용)
            if epoch == 0 and batch_i == 0:
                tqdm.write("\n" + "="*50)
                tqdm.write(f"🔍 [Sanity Check] First Batch Loss: {loss_val:.4f}")
                tqdm.write(f"   Local Emb Mean: {local_emb.float().mean().item():.4f}")
                tqdm.write("="*50 + "\n")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss_val
            
            # [수정] 매 배치마다 상태바 갱신 (멈춤 확인용)
            # 1.1초마다 갱신되면 정상입니다.
            pbar.set_postfix({
                'Tot': f"{loss_val:.3f}", 
                'BPR': f"{bpr_val:.4f}", 
                'SSL': f"{ssl_val:.3f}"
            })

            # 상세 로그 (500 배치마다)
            if batch_i % 100 == 0 and batch_i > 0:
                with torch.no_grad():
                    user_norm = model.embedding_user.weight.data.norm(2, dim=1).mean().item()
                    l_norm = local_emb.norm(p=2, dim=1).mean().item()
                    g_norm = global_emb.norm(p=2, dim=1).mean().item()
                    alignment = F.cosine_similarity(local_emb, global_emb, dim=1).mean().item()

                tqdm.write(
                    f"   [Step {batch_i}] BPR: {bpr_val:.4f} | SSL: {ssl_val:.3f} | "
                    f"Norm(U): {user_norm:.2f} | SVD(Align): {alignment:.3f}"
                )
        
        avg_loss = total_loss / len(dataloader)
        
        # -------------------------------------------------------
        # [추가] 체크포인트 저장 로직 (Best & Last)
        # -------------------------------------------------------
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(), # AMP 사용 시 필수
            'loss': avg_loss,
            'config': config # 나중에 설정 확인용
        }
        
        # 1. 최신 모델 저장 (덮어쓰기)
        torch.save(checkpoint, os.path.join(config['cache_dir'], "lightgcl_last_checkpoint.pth"))
        
        # 2. 최고 성능 모델 저장 (Loss 갱신 시에만)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, os.path.join(config['cache_dir'], "lightgcl_best_model.pth"))
            tqdm.write(f"💾 New Best Model Saved! (Loss: {best_loss:.4f})")
            
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")
        
    torch.save(model.state_dict(), "lightgcl_model.pth")
    print("Model Saved!")

def resume_training(new_config, checkpoint_path):
    print(f"\n♻️ Resuming training from: {checkpoint_path}")
    
    # 1. 데이터 로드 (Main에서 처리된 것과 동일)
    edge_index, num_users, num_items, _, _ = load_and_process_data(
        new_config['json_file_path'], new_config['cache_dir']
    )
    
    # 2. 그래프 구축 (SVD) - 구조는 변하지 않으므로 다시 계산
    adj_tensor, U, S, V = build_graph(
        edge_index, num_users, num_items, new_config['device'], q=new_config['svd_q']
    )
    
    # 3. 데이터셋 & 로더 준비
    dataset = TrainDataset(edge_index, num_users, num_items)
    dataloader = data.DataLoader(
        dataset, 
        batch_size=new_config['batch_size'], 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    
    # 4. 모델 초기화 & 체크포인트 로드
    model = LightGCL(num_users, num_items, new_config, adj_tensor, (U, S, V)).to(new_config['device'])
    
    # 체크포인트 파일 열기
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model weights loaded.")

    # 5. Optimizer 설정 (중요: 새로운 LR 적용)
    # 방법 A: 아예 새로운 Optimizer를 만듦 (가장 깔끔함, 추천)
    optimizer = torch.optim.Adam(model.parameters(), lr=new_config['lr'])
    print(f"✅ Optimizer reset with NEW LR: {new_config['lr']}")

    # (선택) 방법 B: 이전 Optimizer 상태를 복구하되 LR만 바꿀 경우
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = new_config['lr']

    # 6. AMP Scaler 복구
    use_amp = (new_config['device'] == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    print(f"🚀 Resuming from Epoch {start_epoch + 1}...")

    # 7. 학습 루프 (기존 train 함수와 동일)
    # 목표 에포크만큼 추가로 더 돌리거나, 전체 에포크를 채울 때까지 돌림
    # 여기서는 '추가로 new_config['epochs'] 만큼 더' 돌리는 것으로 설정
    total_epochs = start_epoch + new_config['epochs']
    
    best_loss = checkpoint['loss'] # 이전 기록부터 시작

    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
        
        for batch_i, (users, pos_items, neg_items) in enumerate(pbar):
            users = users.to(new_config['device'])
            pos_items = pos_items.to(new_config['device'])
            neg_items = neg_items.to(new_config['device'])
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                local_emb, global_emb = model()
                
                bpr_loss = model.calc_bpr_loss(local_emb, users, pos_items + num_users, neg_items + num_users)
                ssl_loss = model.calc_ssl_loss(local_emb, global_emb, users, pos_items + num_users)
                reg_loss = model.get_l2_reg(users, pos_items, neg_items)
                
                loss = bpr_loss + new_config['lambda_ssl'] * ssl_loss + new_config['lambda_reg'] * reg_loss
                
                loss_val = loss.item()
                bpr_val = bpr_loss.item()
                ssl_val = ssl_loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss_val
            
            pbar.set_postfix({
                'Tot': f"{loss_val:.3f}", 
                'BPR': f"{bpr_val:.4f}", 
                'SSL': f"{ssl_val:.3f}"
            })

            if batch_i % 100 == 0 and batch_i > 0:
                with torch.no_grad():
                    user_norm = model.embedding_user.weight.data.norm(2, dim=1).mean().item()
                    alignment = F.cosine_similarity(local_emb, global_emb, dim=1).mean().item()
                tqdm.write(f"   [Step {batch_i}] BPR: {bpr_val:.4f} | SSL: {ssl_val:.3f} | Norm(U): {user_norm:.2f} | SVD(Align): {alignment:.3f}")

        avg_loss = total_loss / len(dataloader)
        
        # 체크포인트 저장
        new_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': avg_loss,
            'config': new_config
        }
        
        torch.save(new_checkpoint, os.path.join(new_config['cache_dir'], "lightgcl_last_checkpoint.pth"))
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(new_checkpoint, os.path.join(new_config['cache_dir'], "best_model.pth"))
            tqdm.write(f"💾 New Best Model Saved! (Loss: {best_loss:.4f})")
            
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")
        
        
        
from torch.optim.lr_scheduler import CosineAnnealingLR     
        
        
        
def train_fine_tuning(new_config, checkpoint_path):
    print(f"\n🔥 Starting Fine-tuning with Scheduler & Relaxed Reg...")
    
    # 1. 데이터 및 모델 준비 (기존과 동일)
    edge_index, num_users, num_items, _, _ = load_and_process_data(
        new_config['json_file_path'], new_config['cache_dir']
    )
    adj_tensor, U, S, V = build_graph(
        edge_index, num_users, num_items, new_config['device'], q=new_config['svd_q']
    )
    dataset = TrainDataset(edge_index, num_users, num_items)
    dataloader = data.DataLoader(dataset, batch_size=new_config['batch_size'], shuffle=True, num_workers=0)
    
    model = LightGCL(num_users, num_items, new_config, adj_tensor, (U, S, V)).to(new_config['device'])
    
    # 2. 체크포인트 로드 (Loss 0.40 상태의 모델)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Weights Loaded.")

    # 3. Optimizer & Scheduler 설정 (핵심!)
    # LR을 다시 0.002(약간 높음)으로 시작해서 탈출을 시도합니다.
    start_lr = 0.002 
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
    
    # CosineAnnealingLR: LR을 코사인 곡선처럼 부드럽게 0.00001까지 떨어뜨림
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * new_config['epochs'], eta_min=1e-5)
    
    use_amp = (new_config['device'] == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
  
    best_loss = checkpoint['loss']
    print(f"🚀 Fine-tuning for {new_config['epochs']} epochs...")
    print(f"   Strategy: Reg 1e-4 -> {new_config['lambda_reg']} | LR Schedule: {start_lr} -> 1e-5")

    for epoch in range(new_config['epochs']):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{new_config['epochs']}")
        
        for batch_i, (users, pos_items, neg_items) in enumerate(pbar):
            users = users.to(new_config['device'])
            pos_items = pos_items.to(new_config['device'])
            neg_items = neg_items.to(new_config['device'])
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                local_emb, global_emb = model()
                
                bpr_loss = model.calc_bpr_loss(local_emb, users, pos_items + num_users, neg_items + num_users)
                ssl_loss = model.calc_ssl_loss(local_emb, global_emb, users, pos_items + num_users)
                reg_loss = model.get_l2_reg(users, pos_items, neg_items)
                
                loss = bpr_loss + new_config['lambda_ssl'] * ssl_loss + new_config['lambda_reg'] * reg_loss
                loss_val = loss.item()
                bpr_val = bpr_loss.item()
                ssl_val = ssl_loss.item()
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # [추가] 매 배치마다 LR을 아주 조금씩 깎습니다.
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            total_loss += loss_val
            
            # 진행상황 모니터링
            pbar.set_postfix({
                'Loss': f"{loss_val:.4f}", 
                'LR': f"{current_lr:.6f}",
                'BPR': f"{bpr_val:.4f}", 
                'SSL': f"{ssl_val:.3f}"
            })
            
            # Step 단위 Best 저장 (안전을 위해)
            if batch_i % 100 == 0 and batch_i > 0:
            
                with torch.no_grad():
                    user_norm = model.embedding_user.weight.data.norm(2, dim=1).mean().item()
                    alignment = F.cosine_similarity(local_emb, global_emb, dim=1).mean().item()
                tqdm.write(f"   [Step {batch_i}] BPR: {bpr_val:.4f} | SSL: {ssl_val:.3f} | Norm(U): {user_norm:.2f} | SVD(Align): {alignment:.3f}")
            
            if batch_i % 100 == 0 and loss_val < best_loss:
                best_loss = loss_val
                torch.save(model.state_dict(), os.path.join(new_config['cache_dir'], "lightgcl_best_finetuned.pth"))

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")
# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == '__main__':
    CONFIG = {
        'emb_dim': 64,
        'n_layers': 2,
        'temp': 0.2,
        'lambda_ssl': 0.01,
        'lambda_reg': 1e-5,
        'svd_q': 5,
        'lr': 0.005,
        'batch_size': 8192, # 8192로 늘려도 됩니다
        'epochs': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'json_file_path': r'D:\trainDataset\localprops\cache', # 실제 경로
        'cache_dir': r'D:\trainDataset\localprops\cache'
    }
    NEW_CONFIG = {
        'emb_dim': 64,
        'n_layers': 2,
        'temp': 0.2,
        'lambda_ssl': 0.01,
        'lambda_reg': 1e-4,
        'svd_q': 5,
        
        # 🔥 [핵심] 줄어든 LR 적용
        'lr': 0.001, 
        
        'batch_size': 8192,
        'epochs': 5, # 추가로 10 에포크 더 학습
        'device': 'cuda',
        'json_file_path': r'D:\trainDataset\localprops\cache',
        'cache_dir': r'D:\trainDataset\localprops\cache'
    }
    FINE_TUNE_CONFIG = {
        'emb_dim': 64,
        'n_layers': 2,
        'temp': 0.2,
        
        # 🔥 [핵심 1] 방해꾼 제거 (SSL 거의 끔)
        'lambda_ssl': 0.001, 
        
        # 🔥 [핵심 2] 족쇄 풀기 (Reg를 1/10로 줄임)
        'lambda_reg': 1e-5, 
        
        'svd_q': 5,
        'lr': 0.002, # 스케줄러 시작값 (무시됨, 코드 내부 start_lr 따름)
        'batch_size': 8192,
        'epochs': 5, # 5 에포크면 충분히 수렴합니다.
        'device': 'cuda',
        'json_file_path': r'D:\trainDataset\localprops\cache',
        'cache_dir': r'D:\trainDataset\localprops\cache'
    }

    checkpoint_path = os.path.join(FINE_TUNE_CONFIG['cache_dir'], "lightgcl_last_checkpoint.pth")   
    train_fine_tuning(FINE_TUNE_CONFIG, checkpoint_path)
    '''
    # 1. 메인에서 데이터 로드 (한 번만 수행)
    edge_index, num_users, num_items, _, _ = load_and_process_data(
        CONFIG['json_file_path'], CONFIG['cache_dir']
    )

    # 2. 메인에서 데이터셋 생성
    print("\n--- Initializing Dataset ---")
    dataset = TrainDataset(edge_index, num_users, num_items)
    
    # 3. 메인에서 그래프 구축 (SVD)
    adj_tensor, U, S, V = build_graph(
        edge_index, num_users, num_items, CONFIG['device'], q=CONFIG['svd_q']
    )

    # 4. 준비된 객체들을 train 함수로 전달
    train(
        CONFIG, 
        dataset, 
        (adj_tensor, U, S, V), 
        (edge_index, num_users, num_items)
    )
    
    '''
'''
Epoch 1/20:   4%|▋                   | 51/1375 [00:34<14:39,  1.50it/s, Tot=0.779, BPR=0.6930, SSL=8.345] 






SVD(Align) 0.4 ~ 0.8 사이 유지.
Norm(U) 1.0 ~ 5.0 수준으로 커집니다

'''