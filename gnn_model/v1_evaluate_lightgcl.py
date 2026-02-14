import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
    
from tower_code.inference_utils import ITEM_FEAT_PATH_PQ, SEQ_DATA_PATH_PQ, SEQ_VAL_DATA_PATH, TARGET_VAL_PATH, USER_FEAT_PATH_PQ, USER_VAL_FEAT_PATH, FeatureProcessor

# =========================================================
# 1. Inference용 껍데기 모델
# =========================================================
class LightGCL_InferenceWrapper(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64):
        super().__init__()
        # Padding(0) 포함
        self.embedding_user = nn.Embedding(num_users, emb_dim, padding_idx=0)
        self.embedding_item = nn.Embedding(num_items, emb_dim, padding_idx=0)
        
    def forward(self, u_idx):
        return self.embedding_user(u_idx)

def build_sparse_graph(user_ids, item_ids, train_df, device):
    """
    Train DataFrame(Parquet)을 읽어 LightGCL 학습 때와 동일한 
    Normalized Adjacency Matrix를 생성합니다.
    """
    print("⚡ Building Graph Adjacency Matrix...")
    
    n_users = len(user_ids) + 1 # 0번 padding 고려 (Processor 기준)
    n_items = len(item_ids) + 1
    
    # 1. ID 매핑 준비 (Processor의 ID 체계 사용)
    # user_ids, item_ids는 processor.user_ids, processor.item_ids
    u_mapper = {uid: i+1 for i, uid in enumerate(user_ids)}
    i_mapper = {iid: i+1 for i, iid in enumerate(item_ids)}
    
    # 2. 컬럼명 자동 감지 (sequence_ids 추가)
    u_col = 'customer_id' if 'customer_id' in train_df.columns else train_df.columns[0]
    
    # 'sequence_ids'를 우선순위로 둠
    possible_item_cols = ['sequence_ids', 'article_id', 'item_id', 'product_id', 'article_ids']
    i_col = next((col for col in possible_item_cols if col in train_df.columns), None)
    
    if i_col is None:
        raise KeyError(f"❌ Item column not found! Available: {train_df.columns}")
    
    print(f"   -> Using columns: User='{u_col}', Item='{i_col}'")

    # 3. 엣지(Edge) 추출
    src = []
    dst = []
    
    valid_interactions = 0
    
    # DataFrame 순회
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="   -> Mapping Edges"):
        u_val = row[u_col]
        i_val = row[i_col] # 이게 리스트일 확률 99% (sequence_ids)
        
        # 유저가 존재하지 않으면 건너뜀
        if u_val not in u_mapper:
            continue
            
        u_idx = u_mapper[u_val]
        
        # [핵심] 리스트 형태(sequence_ids) 처리
        if isinstance(i_val, (list, np.ndarray)):
            for item in i_val:
                if item in i_mapper:
                    src.append(u_idx)
                    dst.append(i_mapper[item])
                    valid_interactions += 1
        # 단일 값 처리 (혹시 모를 대비)
        else:
            if i_val in i_mapper:
                src.append(u_idx)
                dst.append(i_mapper[i_val])
                valid_interactions += 1
            
    print(f"   -> Valid Edges: {valid_interactions}")
    
    if valid_interactions == 0:
        raise ValueError("❌ No valid interactions found! Check ID matching.")

    # 4. Sparse Matrix 생성 (Train 코드 로직 준수)
    # User-Item Interaction Matrix R
    # shape: (n_users, n_items)
    # src(user indices), dst(item indices)
    
    # 중복 제거 (User가 같은 아이템을 여러 번 샀을 수 있음 -> Graph Edge는 1개로 취급)
    # coo_matrix 생성 시 중복된 좌표는 값이 더해지므로, 일단 만들고 1로 만듦
    R = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n_users, n_items))
    # 0보다 큰 값은 1로 (Interaction 여부만 중요)
    R.data = np.ones_like(R.data) 

    # 5. Adjacency Matrix A 생성
    # [ 0, R ]
    # [ R.T, 0 ]
    # Training 코드에서는 sp.coo_matrix로 직접 좌표를 합쳐서 만들었지만,
    # 여기서는 R을 기반으로 안전하게 만듭니다.
    
    R = R.tocoo()
    
    # User Node: 0 ~ n_users-1
    # Item Node: n_users ~ n_users + n_items - 1 (Offset 적용)
    user_nodes = R.row
    item_nodes = R.col + n_users
    
    # 상단 우측 (User -> Item)
    row_idx = np.concatenate([user_nodes, item_nodes])
    col_idx = np.concatenate([item_nodes, user_nodes])
    data = np.ones(len(row_idx), dtype=np.float32)
    
    num_nodes = n_users + n_items
    adj_mat = sp.coo_matrix((data, (row_idx, col_idx)), shape=(num_nodes, num_nodes))
    
    # 6. Normalization (Train 코드와 완벽 동일 로직)
    # D^-0.5 * A * D^-0.5
    rowsum = np.array(adj_mat.sum(axis=1)).flatten()
    d_inv = np.power(rowsum, -0.5)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    
    norm_adj = d_mat.dot(adj_mat).dot(d_mat)
    norm_adj = norm_adj.tocoo()
    
    # 7. Torch Sparse Tensor 변환
    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data).float()
    shape = torch.Size(norm_adj.shape)
    
    adj_tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce().to(device)
    
    return adj_tensor

def compute_final_embeddings(model, adj_tensor, n_layers=2):
    """
    저장된 Weight(Layer 0)를 그래프에 통과시켜 Final Embedding을 계산
    """
    print(f"\n🌊 Propagating Embeddings (Layers: {n_layers})...")
    model.eval()
    with torch.no_grad():
        # 1. 초기 임베딩 결합 (User + Item)
        ego_embeddings = torch.cat([
            model.embedding_user.weight, 
            model.embedding_item.weight
        ], dim=0)
        
        all_embeddings = [ego_embeddings]
        
        # 2. 레이어 전파 (Graph Convolution)
        for k in range(n_layers):
            # Sparse Matrix Multiplication (Message Passing)
            ego_embeddings = torch.sparse.mm(adj_tensor, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            print(f"   -> Layer {k+1} done.")
            
        # 3. 레이어 평균 (Mean Aggregation)
        # stack -> (Layers, Nodes, Dim) -> mean(dim=0)
        final_embeddings = torch.stack(all_embeddings, dim=0).mean(dim=0)
        
        # 4. 다시 User/Item으로 분리
        num_users = model.embedding_user.num_embeddings
        num_items = model.embedding_item.num_embeddings
        
        final_user_emb, final_item_emb = torch.split(final_embeddings, [num_users, num_items])
        
        return final_user_emb, final_item_emb
# =========================================================
# 2. 모델 로드 및 정렬 (Alignment) - 가장 중요 ⭐
# =========================================================
def load_and_align_model(model, processor, checkpoint_path, maps_path, device):
    """
    학습된 .pth(과거 ID 순서)를 로드하여,
    현재 processor(검증 ID 순서)에 맞게 임베딩 행렬을 재조립합니다.
    """
    print(f"\n🔄 [Alignment] Loading model weights and aligning IDs...")
    
    # 1. 파일 로드
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # 학습 당시 ID 맵 로드 (이게 없으면 복원 불가능)
    saved_maps = torch.load(maps_path, map_location='cpu')
    train_user2id = saved_maps['user2id']
    train_item2id = saved_maps['item2id']
    
    # 2. User Embedding 정렬
    # 학습된 원본 가중치
    raw_user_emb = state_dict['embedding_user.weight'] 
    # 새로 만들 빈 행렬 (현재 Processor 기준 크기)
    aligned_user_emb = torch.zeros(len(processor.user_ids) + 1, raw_user_emb.shape[1])
    
    u_hit = 0
    # 현재 Processor의 ID 순서대로 순회하며 학습된 가중치를 가져옴
    for i, uid_str in enumerate(processor.user_ids):
        target_idx = i + 1 # 1-based index
        if uid_str in train_user2id:
            src_idx = train_user2id[uid_str]
            if src_idx < len(raw_user_emb):
                aligned_user_emb[target_idx] = raw_user_emb[src_idx]
                u_hit += 1
    
    model.embedding_user = nn.Embedding.from_pretrained(aligned_user_emb, freeze=True, padding_idx=0)
    print(f"   ✅ Users Aligned: {u_hit} / {len(processor.user_ids)} (Coverage: {u_hit/len(processor.user_ids):.2%})")

    # 3. Item Embedding 정렬
    raw_item_emb = state_dict['embedding_item.weight']
    aligned_item_emb = torch.zeros(len(processor.item_ids) + 1, raw_item_emb.shape[1])
    
    i_hit = 0
    for i, iid_str in enumerate(processor.item_ids):
        target_idx = i + 1
        if iid_str in train_item2id:
            src_idx = train_item2id[iid_str]
            if src_idx < len(raw_item_emb):
                aligned_item_emb[target_idx] = raw_item_emb[src_idx]
                i_hit += 1
                
    model.embedding_item = nn.Embedding.from_pretrained(aligned_item_emb, freeze=True, padding_idx=0)
    print(f"   ✅ Items Aligned: {i_hit} / {len(processor.item_ids)} (Coverage: {i_hit/len(processor.item_ids):.2%})")
    
    return model.to(device)

# =========================================================
# 3. 정답 데이터 전처리 (String -> Integer Set)
# =========================================================
def prepare_ground_truth(target_df_path, processor):
    """
    평가 속도를 위해 String ID 정답지를 Integer Index 집합으로 미리 변환합니다.
    Return: {user_idx: {item_idx1, item_idx2, ...}}
    """
    print("\n⚡ Preparing Ground Truth Data...")
    df = pd.read_parquet(target_df_path) # [customer_id, target_ids]
    
    ground_truth = {}
    
    # DataFrame 순회
    for _, row in tqdm(df.iterrows(), total=len(df), desc="   -> Indexing Targets"):
        u_str = row['customer_id']
        t_list = row['target_ids']
        
        # User가 Processor에 없으면 평가 불가 (Skip)
        if u_str not in processor.user2id:
            continue
            
        u_idx = processor.user2id[u_str]
        
        # Target Item들도 Integer ID로 변환
        item_indices = set()
        for i_str in t_list:
            if i_str in processor.item2id:
                item_indices.add(processor.item2id[i_str])
        
        if item_indices: # 정답이 하나라도 있는 경우만
            ground_truth[u_idx] = item_indices
            
    print(f"   ✅ Ready to evaluate {len(ground_truth)} users.")
    return ground_truth

# =========================================================
# 4. 평가 루프 (Clean Logic)
# =========================================================
def evaluate_recall(model, ground_truth_dict, device, k_list=[20, 100], batch_size=4096):
    """
    [수정됨] Cosine Similarity 대신 Dot Product 사용
    """
    max_k = max(k_list)
    model.eval()
    
    eval_user_indices = list(ground_truth_dict.keys())
    
    # DataLoader
    loader = DataLoader(
        eval_user_indices, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda x: torch.tensor(x, dtype=torch.long)
    )
    
    # 1. 아이템 임베딩 준비 (Normalize 제거! ❌)
    with torch.no_grad():
        all_items = model.embedding_item.weight.data # (M, Dim)
        # all_items_norm = F.normalize(all_items, p=2, dim=1) <--- 삭제
    
    hits = {k: 0 for k in k_list}
    total_users = 0
    
    print(f"\n🚀 Starting Recall@{k_list} Evaluation (Metric: Dot Product)...")
    
    with torch.no_grad():
        for batch_u_idx in tqdm(loader, desc="   -> Retrieving"):
            batch_u_idx = batch_u_idx.to(device)
            
            # 2. 유저 임베딩 준비 (Normalize 제거! ❌)
            user_emb = model.embedding_user(batch_u_idx)
            # user_norm = F.normalize(user_emb, p=2, dim=1) <--- 삭제
            
            # 3. Score 계산 (Pure Dot Product)
            # (Batch, Dim) @ (All_Items, Dim).T
            scores = torch.matmul(user_emb, all_items.T)
            
            # Padding 마스킹
            scores[:, 0] = -float('inf')
            
            # Top-K
            _, topk_indices = torch.topk(scores, k=max_k, dim=1)
            topk_cpu = topk_indices.cpu().numpy()
            batch_u_cpu = batch_u_idx.cpu().numpy()
            
            # Metric Check (기존 동일)
            for i, u_idx in enumerate(batch_u_cpu):
                true_item_set = ground_truth_dict[u_idx]
                pred_list = topk_cpu[i]
                
                for k in k_list:
                    if not true_item_set.isdisjoint(pred_list[:k]):
                        hits[k] += 1
                        
            total_users += len(batch_u_cpu)

    # Report
    print(f"\n{'='*40}")
    print(f"📊 LightGCL Final Report (Dot Product)")
    print(f"{'-'*40}")
    for k in sorted(k_list):
        recall = hits[k] / total_users
        print(f"Recall@{k:<3} | {recall:.4f}")
    print(f"{'='*40}\n")

# =========================================================
# 5. 실행부 (Main)
# =========================================================
if __name__ == '__main__':
    # 설정 (경로 수정 필요)
    BASE_DIR = r'D:\trainDataset\localprops'
    CACHE_DIR = os.path.join(BASE_DIR, 'cache')
    
    # 1. 학습된 모델 경로 (Fine-tuning 완료된 모델)
    MODEL_PATH = os.path.join(CACHE_DIR, "lightgcl_best_finetuned_2401_ep3.pth") 
    MAPS_PATH = os.path.join(CACHE_DIR, "id_maps_train.pt") # 학습 시 저장한 ID 매핑
    
    # 2. 검증용 데이터 경로 (Parquet)
    TARGET_DF_PATH = os.path.join(BASE_DIR, "validation_targets.parquet") # [customer_id, target_ids]
    print("1️⃣ Initializing Processors...")
    train_proc = FeatureProcessor(USER_FEAT_PATH_PQ, ITEM_FEAT_PATH_PQ, SEQ_DATA_PATH_PQ)
    valid_proc = FeatureProcessor(
        USER_VAL_FEAT_PATH,  # 검증 유저 피처
        ITEM_FEAT_PATH_PQ,   # 아이템 피처 (공유)
        SEQ_VAL_DATA_PATH,   # ⭐ 핵심: 검증용 시퀀스 (Target 제외)
        scaler=train_proc.user_scaler # Scaler 공유
    )
    
    # [중요] ID 매핑을 Train과 동일하게 강제 일치
    # (새로운 아이템/유저가 있으면 무시하거나 처리하기 위해)
    valid_proc.user2id = train_proc.user2id
    valid_proc.item2id = train_proc.item2id
    valid_proc.user_ids = train_proc.user_ids 
    valid_proc.item_ids = train_proc.item_ids
    num_users = len(train_proc.user_ids) + 1
    num_items = len(train_proc.item_ids) + 1
    

    print("1️⃣ Initializing Processor...")
    # processor = FeatureProcessor(...) # 실제 코드에선 이걸 쓰세요


    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = LightGCL_InferenceWrapper(
        num_users=num_users,
        num_items=num_items
    )
    model = load_and_align_model(model, train_proc, MODEL_PATH, MAPS_PATH, device)
    
    
    
    train_df = pd.read_parquet(SEQ_DATA_PATH_PQ) 



    adj_tensor = build_sparse_graph(
        train_proc.user_ids, 
        train_proc.item_ids, 
        train_df,
        device
    )
    
    
    
    
    print("\n🔍 [Check] Verifying Embedding Propagation...")

    # 1. 전파 전 (Original Layer-0) 상태 저장
    # .clone()을 해야 값이 복사되어 따로 저장됩니다.
    before_user_emb = model.embedding_user.weight.data.clone()
    before_mean = before_user_emb.mean().item()
    before_std = before_user_emb.std().item()

    print(f"   Original Weights | Mean: {before_mean:.6f} | Std: {before_std:.6f}")
    
    
    
    final_user_emb, final_item_emb = compute_final_embeddings(
    model, 
    adj_tensor, 
    n_layers=2 # 학습 때 설정한 레이어 수와 동일해야 함! (보통 2 or 3)
    )


    
    with torch.no_grad():
        model.embedding_user.weight.copy_(final_user_emb)
        model.embedding_item.weight.copy_(final_item_emb)

    print("✅ Model updated with Propagated Embeddings.")
    # 4. 전파 후 (Propagated Final) 상태 확인
    after_user_emb = model.embedding_user.weight.data
    after_mean = after_user_emb.mean().item()
    after_std = after_user_emb.std().item()

    print(f"   Propagated Weights | Mean: {after_mean:.6f} | Std: {after_std:.6f}")

    # 5. 결과 판정
    if before_mean == after_mean:
        print("❌ [FAIL] Embeddings did NOT change! (Something is wrong)")
        # 원인: compute_final_embeddings 함수가 원본을 그대로 반환했거나, copy_가 안 먹힘
    else:
        print("✅ [SUCCESS] Embeddings successfully updated via Graph Propagation!")
        
        # 얼마나 변했는지 차이 계산 (L2 Distance)
        diff = torch.norm(before_user_emb - after_user_emb).item()
        print(f"   -> Difference Magnitude: {diff:.4f}")
        
    
    ground_truth = prepare_ground_truth(TARGET_VAL_PATH, valid_proc)
    # 4. 평가
    evaluate_recall(model, ground_truth, device, k_list=[20, 100])