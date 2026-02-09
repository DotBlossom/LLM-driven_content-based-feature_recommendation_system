import os
import random
import math  # [필수] sqrt 사용을 위해 추가
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="Support for mismatched src_key_padding_mask and mask is deprecated")
# -------------------------------------------------------------------------
# 0. Global Configuration & Logger
# -------------------------------------------------------------------------
# [수정] Temperature를 낮춰야 정규화된 벡터끼리 구분이 가능해짐
TEMPERATURE = 0.15
# [수정] 초기 학습 안정성을 위해 LogQ 가중치를 약간 낮춤 (나중에 올려도 됨)
LAMBDA_LOGQ = 0.0
BATCH_SIZE = 768
EMBED_DIM = 128
MAX_SEQ_LEN = 50
DROPOUT = 0.2
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 경로 설정 (사용자 환경에 맞게 유지)
BASE_DIR = r"D:\trainDataset\localprops"
MODEL_DIR = r"C:\Users\candyform\Desktop\inferenceCode\models"
ITEM_FEAT_PATH_PQ = os.path.join(BASE_DIR, "features_item.parquet")
USER_FEAT_PATH_PQ = os.path.join(BASE_DIR, "features_user.parquet")
SEQ_DATA_PATH_PQ = os.path.join(BASE_DIR, "features_sequence_cleaned.parquet")
GNN_PATH = os.path.join(MODEL_DIR, "simgcl_trained.pth")
GNN_MAP_PATH = os.path.join(BASE_DIR, "cache", "id_maps_train.pt")
ITEM_MATRIX_PATH = os.path.join(MODEL_DIR, "pretrained_item_matrix.pt")
ITEM_ID_PATH = os.path.join(MODEL_DIR, "item_ids.pt")
TARGET_VAL_PATH = os.path.join(BASE_DIR, "features_target_val.parquet")
PHASE2_WEIGHTS = os.path.join(MODEL_DIR, "user_tower_phase2.pth")
SAVE_PATH_BEST_PREV = os.path.join(MODEL_DIR, "user_tower_phase2.5_best.pth")
USER_VAL_FEAT_PATH = os.path.join(BASE_DIR, "features_user_val.parquet")
SEQ_VAL_DATA_PATH = os.path.join(BASE_DIR, "features_sequence_val.parquet")
SAVE_PATH_BEST = os.path.join(MODEL_DIR, "user_tower_phase2.5_best_ft.pth")

class SmartLogger:
    def __init__(self, verbosity=1): self.verbosity = verbosity
    def log(self, level, msg):
        if self.verbosity >= level: print(f"[{'ℹ️' if level==1 else '📊'}] {msg}")

logger = SmartLogger(verbosity=1)
import torch
import torch.nn.functional as F
import pandas as pd

# ==========================================
# 🛠️ 설정 (경로 확인 필수)
# ==========================================
ITEM_MATRIX_PATH = r"C:\Users\candyform\Desktop\inferenceCode\models\pretrained_item_matrix.pt"
ITEM_META_PATH = r"D:\trainDataset\localprops\features_item.parquet" # 아이템 이름 확인할 메타데이터
import torch
import torch.nn as nn
import os
def verify_gnn_alignment(model, processor, base_dir):
    print(f"\n🕵️‍♂️ [GNN Verification] Checking GNN Alignment Integrity...")
    
    cache_dir = os.path.join(base_dir, "cache")
    model_path = os.path.join(MODEL_DIR, "simgcl_trained.pth")
    maps_path = os.path.join(cache_dir, "id_maps_train.pt")
    
    try:
        # 1. 원본 소스 로드 (비교 기준)
        maps = torch.load(maps_path, map_location='cpu')
        gnn_item2id = maps['item2id'] # {'item_str': gnn_idx}
        
        gnn_state_dict = torch.load(model_path, map_location='cpu')
        gnn_source_weight = gnn_state_dict['embedding_item.weight'] # 원본 벡터 뭉치
        
    except Exception as e:
        print(f"⚠️ 원본 파일 로드 실패로 검증 건너뜀: {e}")
        return

    # 2. 현재 모델의 GNN 레이어 가져오기
    # 아까 변수명이 'gnn_user_emb'라고 하셨으므로 그걸 가져옵니다.
    if hasattr(model, 'gnn_user_emb'):
        current_weight = model.gnn_user_emb.weight.detach().cpu()
    elif hasattr(model, 'item_gnn_emb'):
        current_weight = model.item_gnn_emb.weight.detach().cpu()
    else:
        print("❌ [Error] 모델에서 GNN 레이어를 찾을 수 없습니다.")
        return

    # 3. 샘플링 검사
    check_cnt = 0
    success_cnt = 0
    
    print(f"   -------------------------------------------------------------")
    print(f"   {'Status':<10} | {'Item ID':<15} | {'Vector Match?':<15} | {'Diff Sum'}")
    print(f"   -------------------------------------------------------------")

    for item_id_str in processor.item_ids:
        # 5개만 확인
        if check_cnt >= 5: break
        
        # (1) 현재 모델에서의 위치와 값
        if item_id_str not in processor.item2id: continue
        target_idx = processor.item2id[item_id_str]
        model_vec = current_weight[target_idx]
        
        # (2) GNN 원본에서의 위치와 값
        if item_id_str in gnn_item2id:
            gnn_idx = gnn_item2id[item_id_str]
            original_vec = gnn_source_weight[gnn_idx]
            
            # (3) 비교
            is_same = torch.allclose(model_vec, original_vec, atol=1e-5)
            diff = (model_vec - original_vec).abs().sum().item()
            
            status = "✅ Matched" if is_same else "❌ Broken"
            print(f"   {status:<10} | {item_id_str:<15} | {str(is_same):<15} | {diff:.6f}")
            
            if is_same: success_cnt += 1
            check_cnt += 1
            
    print(f"   -------------------------------------------------------------")
    
    if success_cnt == check_cnt:
        print(f"🎉 [Success] GNN Vectors are perfectly aligned!")
    else:
        print(f"🔥 [Fail] Some GNN vectors do not match. Check Alignment Logic!")
def load_and_align_gnn_items(model, processor, base_dir, device):
    """
    GNN 학습 결과(simgcl_trained.pth)와 ID맵(id_maps_train.pt)을 로드하여
    현재 User Tower의 아이템 순서에 맞게 재정렬 후 주입합니다.
    """
    print(f"\n🔄 [GNN Alignment] Starting GNN Item Embedding Alignment...")
    
    # GNN 학습 코드에 설정된 경로 기준
    cache_dir = os.path.join(base_dir, "cache")
    
    # 1. 파일 경로 설정
    # GNN 모델 가중치 (simgcl_trained.pth)
    model_path = os.path.join(MODEL_DIR , "simgcl_trained.pth")
    # GNN ID 매핑 파일 (id_maps_train.pt)
    maps_path = os.path.join(cache_dir, "id_maps_train.pt")

    # 2. 파일 로드
    try:
        # (A) ID 매핑 로드
        maps = torch.load(maps_path, map_location='cpu')
        gnn_item2id = maps['item2id'] # {'item_str': index} 형태
        
        # (B) GNN 모델 가중치 로드
        gnn_state_dict = torch.load(model_path, map_location='cpu')
        # GNN 코드상 변수명: embedding_item.weight
        gnn_emb_weight = gnn_state_dict['embedding_item.weight']
        
        print(f"   - GNN Source: {gnn_emb_weight.shape} vectors")
        print(f"   - GNN Map Size: {len(gnn_item2id)} items")
        
    except Exception as e:
        print(f"❌ [Error] Failed to load GNN files: {e}")
        print("   👉 경로가 맞는지, GNN 학습이 완료되었는지 확인하세요.")
        return model

    # 3. 타겟(User Tower)에 맞는 새로운 매트릭스 생성
    # User Tower 아이템 개수 + 1 (Padding)
    num_embeddings = len(processor.item_ids) + 1 
    emb_dim = gnn_emb_weight.shape[1]
    
    # 초기화: 매칭 안 되는 건 랜덤 (또는 0)
    new_weight = torch.randn(num_embeddings, emb_dim) * 0.01
    new_weight[0] = 0.0 # Padding

    # 4. 매핑 수행 (Alignment)
    matched_count = 0
    missing_count = 0
    
    # processor.item_ids: User Tower가 사용하는 아이템 리스트 (순서 중요!)
    for i, current_id_str in enumerate(processor.item_ids):
        # User Tower의 인덱스 (1부터 시작)
        target_idx = i + 1 
        
        # GNN 족보(gnn_item2id)에 이 아이템이 있는가?
        if current_id_str in gnn_item2id:
            # GNN에서의 인덱스 찾기
            gnn_idx = gnn_item2id[current_id_str]
            
            # 벡터 복사: GNN[gnn_idx] -> UserTower[target_idx]
            new_weight[target_idx] = gnn_emb_weight[gnn_idx]
            matched_count += 1
        else:
            missing_count += 1
            
    # 5. 모델 주입
    # HybridUserTower 내부의 GNN 임베딩 레이어 변수명 확인 필요!
    # (여기서는 'item_gnn_emb'라고 가정합니다. 다르면 수정하세요!)
    target_layer_name = 'gnn_user_emb' 
    
    with torch.no_grad():
        if hasattr(model, target_layer_name):
            # [중요] freeze=False (미세조정 허용)
            setattr(model, target_layer_name, nn.Embedding.from_pretrained(new_weight.to(device), freeze=False))
            print(f"   ✅ Injected aligned vectors into 'model.{target_layer_name}'")
        else:
            # 혹시 변수명이 gnn_item_emb 일 수도 있음
            fallback_name = 'gnn_item_emb'
            if hasattr(model, fallback_name):
                setattr(model, fallback_name, nn.Embedding.from_pretrained(new_weight.to(device), freeze=False))
                print(f"   ✅ Injected aligned vectors into 'model.{fallback_name}'")
            else:
                print(f"❌ [Critical] Could not find GNN layer in User Tower. Check variable names!")
                return model

    print(f"✅ [GNN Alignment] Complete!")
    print(f"   - Matched: {matched_count}")
    print(f"   - Missing: {missing_count}")
    
    return model
def verify_embedding_alignment(model, processor, model_dir):
    print(f"\n🕵️‍♂️ [Verification] Checking Alignment Integrity...")
    
    # 1. 비교를 위해 원본(Source) 다시 로드 (메모리 부담되면 생략 가능하지만, 확실한 검증을 위해 권장)
    emb_path = os.path.join(model_dir, "pretrained_item_matrix.pt")
    ids_path = os.path.join(model_dir, "item_ids.pt")
    
    try:
        source_emb = torch.load(emb_path, map_location='cpu')
        if isinstance(source_emb, dict):
            source_emb = source_emb.get('weight', source_emb.get('item_content_emb.weight'))
        source_ids = torch.load(ids_path, map_location='cpu')
    except:
        print("⚠️ 원본 파일 로드 실패로 검증 건너뜀")
        return

    # 원본 맵 생성
    source_map = {}
    for idx, item_id in enumerate(source_ids):
        key = str(item_id.item()) if isinstance(item_id, torch.Tensor) else str(item_id)
        source_map[key] = source_emb[idx]

    # 2. 샘플링 검사 (매칭된 것 5개, 없는 것 1개 확인)
    check_cnt = 0
    success_cnt = 0
    
    model_weight = model.item_content_emb.weight.detach().cpu()
    
    print(f"   -------------------------------------------------------------")
    print(f"   {'Status':<10} | {'Item ID':<15} | {'Vector Match?':<15} | {'Diff Sum'}")
    print(f"   -------------------------------------------------------------")

    for item_id_str in processor.item_ids:
        # 5개만 확인하고 종료
        if check_cnt >= 5: break
        
        # 모델 내 인덱스 찾기
        if item_id_str not in processor.item2id: continue
        target_idx = processor.item2id[item_id_str]
        
        # 모델에 있는 벡터
        current_vec = model_weight[target_idx]
        
        if item_id_str in source_map:
            # Case 1: 매칭된 아이템 (Pretrained와 값이 같아야 함)
            original_vec = source_map[item_id_str]
            
            # 값이 같은지 확인 (오차 1e-5 이내)
            is_same = torch.allclose(current_vec, original_vec, atol=1e-5)
            diff = (current_vec - original_vec).abs().sum().item()
            
            status = "✅ Matched" if is_same else "❌ Broken"
            print(f"   {status:<10} | {item_id_str:<15} | {str(is_same):<15} | {diff:.6f}")
            if is_same: success_cnt += 1
            check_cnt += 1
            
        else:
            # Case 2: 매칭 안 된 아이템 (검증 대상 아님, 로그만 확인)
            pass

    print(f"   -------------------------------------------------------------")
    
    if success_cnt == check_cnt:
        print(f"🎉 [Success] Vectors are perfectly aligned!")
    else:
        print(f"🔥 [Fail] Some vectors do not match source. Check Logic!")
def check_embedding_sanity():
    print("🕵️‍♂️ [Sanity Check] ID Mapping Verification Starting...")

    # 1. Pretrained Matrix 로드
    try:
        vectors = torch.load(ITEM_MATRIX_PATH, map_location='cpu')
        # 만약 vectors가 dict 형태라면 가중치 키를 찾아야 함
        if isinstance(vectors, dict):
            vectors = vectors['weight'] if 'weight' in vectors else list(vectors.values())[0]
        
        print(f"✅ Loaded Vectors: {vectors.shape}")
    except Exception as e:
        print(f"❌ Failed to load vectors: {e}")
        return

    # 2. 아이템 메타데이터 로드 (이름 확인용)
    items_df = pd.read_parquet(ITEM_META_PATH)
    # article_id가 String인지 확인
    if 'article_id' in items_df.columns:
        items_df['article_id'] = items_df['article_id'].astype(str)
        items_df = items_df.set_index('article_id')
    
    # 3. 테스트할 아이템 선정 (유명한거나 랜덤으로)
    # 예: 데이터프레임의 첫 번째 아이템
    test_ids = items_df.index[:3].tolist() 
    
    # 학습 때 사용한 item2id가 있다면 그 순서대로 조회해야 함.
    # 여기서는 "Pretrained Matrix의 n번째 줄이, items_df의 n번째 아이템과 일치하는지" 가정하고 테스트
    
    vectors = F.normalize(vectors, p=2, dim=1)

    for i, target_id in enumerate(test_ids):
        if i >= len(vectors): break
        
        query_vec = vectors[i].unsqueeze(0) # i번째 벡터 (라고 가정되는 것)
        
        # 전체와의 유사도 계산
        sims = torch.matmul(query_vec, vectors.T).squeeze()
        topk_val, topk_idx = torch.topk(sims, k=5) # 자기 자신 포함 상위 5개
        
        target_name = items_df.loc[target_id]['prod_name'] if 'prod_name' in items_df.columns else "Unknown"
        print(f"\n🎯 Query [{i}]: ID={target_id} ({target_name})")
        print("-" * 40)
        
        for rank, idx in enumerate(topk_idx.tolist()):
            idx = int(idx)
            # 순서가 맞다면 items_df의 idx번째 아이템 정보를 가져와야 함
            if idx < len(items_df):
                neighbor_id = items_df.index[idx]
                neighbor_name = items_df.iloc[idx]['prod_name'] if 'prod_name' in items_df.columns else "Unknown"
                score = topk_val[rank].item()
                print(f"   Rank {rank}: {neighbor_name} (Sim: {score:.4f})")
            else:
                print(f"   Rank {rank}: Index {idx} (Out of DF bounds)")
        
        print("-" * 40)

    print("\n🤔 판단 기준:")
    print("1. Rank 0은 무조건 자기 자신이 나와야 함 (Sim 1.0)")
    print("2. Rank 1~4에 의미적으로 비슷한 상품(예: 같은 카테고리)이 나와야 함")
    print("👉 만약 Rank 0에 엉뚱한 이름이 나오거나, 유사 상품이 전혀 쌩뚱맞다면 'ID 순서 꼬임' 확정!")


# -------------------------------------------------------------------------
# 1. Feature Processor (Scaler 로직 수정)
# -------------------------------------------------------------------------
class FeatureProcessor:
    def __init__(self, user_path, item_path, seq_path, scaler=None):
        self.users = pd.read_parquet(user_path).set_index('customer_id')
        self.items = pd.read_parquet(item_path).set_index('article_id')
        self.seqs = pd.read_parquet(seq_path).set_index('customer_id')
        self.user_ids = self.users.index.tolist()
        self.user2id = {uid: i + 1 for i, uid in enumerate(self.user_ids)}
        self.item_ids = self.items.index.tolist()
        self.item2id = {iid: i + 1 for i, iid in enumerate(self.item_ids)}
        
        self.u_dense_cols = ['user_avg_price_log', 'total_cnt_log', 'recency_log']
        self.users_scaled = self.users.copy()
        self.user_scaler = StandardScaler()

        # [수정] Scaler 공유 로직
        if scaler is None: # 학습용 (Fit 수행)
            self.users_scaled[self.u_dense_cols] = self.user_scaler.fit_transform(self.users[self.u_dense_cols])
        else: # 검증용 (Train의 분포를 그대로 사용)
            self.user_scaler = scaler
            self.users_scaled[self.u_dense_cols] = self.user_scaler.transform(self.users[self.u_dense_cols])

    def get_user_tensor(self, user_id):
        dense = torch.tensor(self.users_scaled.loc[user_id, self.u_dense_cols].values, dtype=torch.float32)
        cat = torch.tensor(int(self.users_scaled.loc[user_id, 'preferred_channel']) - 1, dtype=torch.long)
        return dense, cat

    def get_logq_probs(self, device):
        sorted_probs = self.items['raw_probability'].reindex(self.item_ids).fillna(0).values
   
        return torch.tensor(sorted_probs, dtype=torch.float32).to(device)

def load_and_align_embeddings(model, processor, model_dir, device):
    """
    Pretrained 임베딩(66k)을 현재 데이터셋(47k) 순서에 맞춰 재정렬하여 모델에 주입하는 함수
    """
    print(f"\n🔄 [Alignment] Starting Item Embedding Alignment...")
    
    emb_path = os.path.join(model_dir, "pretrained_item_matrix.pt")
    ids_path = os.path.join(model_dir, "item_ids.pt")

    # 1. 파일 로드
    try:
        # 임베딩 로드
        pretrained_emb = torch.load(emb_path, map_location='cpu')
        if isinstance(pretrained_emb, dict):
             # state_dict 형태로 저장된 경우 'weight' 키를 찾음
            pretrained_emb = pretrained_emb.get('weight', pretrained_emb.get('item_content_emb.weight'))
        
        # ID 리스트 로드
        pretrained_ids = torch.load(ids_path, map_location='cpu')
        
        print(f"   - Pretrained Source: {pretrained_emb.shape} vectors, {len(pretrained_ids)} IDs")
        
    except Exception as e:
        print(f"❌ [Error] Failed to load pretrained files: {e}")
        return model

    # 2. Dictionary로 변환 (검색 속도 향상: O(1))
    # { '아이템ID_스트링': 벡터_텐서 }
    pretrained_map = {}
    for idx, item_id in enumerate(pretrained_ids):
        # item_id가 Tensor면 값을 꺼내고, 아니면 그대로 문자열 변환
        key = str(item_id.item()) if isinstance(item_id, torch.Tensor) else str(item_id)
        pretrained_map[key] = pretrained_emb[idx]

    # 3. 타겟(현재 모델)에 맞는 새로운 임베딩 매트릭스 생성
    # processor.item_ids 개수 + 1 (Padding용 0번 인덱스)
    num_embeddings = len(processor.item_ids) + 1 
    emb_dim = pretrained_emb.shape[1]
    
    # 초기화: 랜덤 값으로 시작 (매칭 안 되는 신규 아이템을 위해)
    new_weight = torch.randn(num_embeddings, emb_dim) * 0.01 
    # Padding(0번)은 0으로 고정
    new_weight[0] = 0.0 

    # 4. 매핑 수행 (Alignment)
    matched_count = 0
    missing_count = 0
    
    # processor.item_ids는 실제 아이템 리스트 (1번 인덱스부터 시작)
    for i, current_id_str in enumerate(processor.item_ids):
        target_idx = i + 1  # 모델 내 인덱스 (0은 패딩이므로 +1)
        
        if current_id_str in pretrained_map:
            # 매칭 성공: 벡터 복사
            new_weight[target_idx] = pretrained_map[current_id_str]
            matched_count += 1
        else:
            # 매칭 실패: 랜덤 초기화 유지 (신규 아이템 등)
            missing_count += 1
            
    # 5. 모델에 주입 (수술)
    with torch.no_grad():
        # 모델의 임베딩 레이어 교체
        # [중요] freeze=False로 설정하여 Missing 아이템도 학습되게 함
        model.item_content_emb = nn.Embedding.from_pretrained(new_weight.to(device), freeze=False)
        
    print(f"✅ [Alignment] Complete!")
    print(f"   - Matched: {matched_count} (Recovered from Pretrained)")
    print(f"   - Missing: {missing_count} (Initialized Randomly)")
    print(f"   - Total: {num_embeddings} rows injected into Model.")
    
    return model

def load_and_align_gnn_user_embeddings(gnn_model_path, gnn_map_path, target_user_ids, device='cpu'):
    logger.log(1, "🔄 Aligning GNN User Embeddings...")

    # 1. 파일 존재 확인
    if not os.path.exists(gnn_model_path) or not os.path.exists(gnn_map_path):
        logger.log(1, f"⚠️ GNN files missing ({gnn_model_path} or {gnn_map_path}). Using Random Init.")
        return None

    try:
        # 2. GNN 모델 가중치 로드
        state_dict = torch.load(gnn_model_path, map_location=device)
        
        # SimGCL 코드 기준 변수명: 'embedding_user.weight'
        # 혹시 모르니 검색 로직 유지
        gnn_matrix = None
        for key, tensor in state_dict.items():
            if 'embedding_user' in key and tensor.ndim == 2:
                gnn_matrix = tensor
                break
        
        if gnn_matrix is None:
            logger.log(1, "❌ Could not find 'embedding_user' in GNN state_dict.")
            return None

        # 3. GNN ID 매핑 로드
        # SimGCL 저장 코드: torch.save({'user2id': user2id, ...}, map_path)
        maps = torch.load(gnn_map_path)
        if 'user2id' not in maps:
            logger.log(1, "❌ 'user2id' key missing in GNN map file.")
            return None
            
        gnn_user2id = maps['user2id'] # {user_str: int_idx}

        # 4. 정렬 (Alignment)
        num_target = len(target_user_ids) + 1
        dim = gnn_matrix.shape[1]
    
        # 기본적으로 0.0으로 채워짐 (Padding Vector 역할)
        aligned_matrix = torch.zeros((num_target, dim), dtype=torch.float32)
        
        hit_count = 0
        
        for i, u_id in enumerate(target_user_ids):
                # FeatureProcessor 순서상 i번째 유저는 -> 모델 내부에서 i+1번 인덱스를 씀
            current_idx = i + 1 
                
            if u_id in gnn_user2id:
                    # GNN에서는 0부터 시작했으므로 그대로 가져옴
                origin_idx = gnn_user2id[u_id] 
                    
                    # 인덱스 범위 안전장치
                if origin_idx < gnn_matrix.shape[0]:
                        # ★ 여기가 핵심: GNN(origin) -> Tower(current=i+1)
                    aligned_matrix[current_idx] = gnn_matrix[origin_idx]
                    hit_count += 1
                else:
                    torch.nn.init.xavier_normal_(aligned_matrix[current_idx].unsqueeze(0))
            else:
                    # GNN에 없던 유저는 랜덤 초기화
                torch.nn.init.xavier_normal_(aligned_matrix[current_idx].unsqueeze(0))

        logger.log(1, f"✅ GNN Alignment: {hit_count}/{len(target_user_ids)} users aligned.")
        if aligned_matrix is not None:
            aligned_matrix = F.normalize(aligned_matrix, p=2, dim=1)
            
        logger.log(1, f"✅ GNN Alignment: {hit_count}/{len(target_user_ids)} users aligned.")
        return aligned_matrix

    except Exception as e:
        logger.log(1, f"❌ Error during GNN alignment: {e}")
        return None
    
   
def load_and_align_item_vectors(pretrained_path, id_path, target_item_ids, embed_dim=128):
    logger.log(1, "Aligning Item Vectors (Pretrained -> Current Model)...")
    
    if not os.path.exists(pretrained_path) or not os.path.exists(id_path):
        logger.log(1, "⚠️ Pretrained item files missing. Returning Random.")
        return None
        
    master_matrix = torch.load(pretrained_path, map_location='cpu') 
    master_ids = torch.load(id_path)
    
    logger.log(2, f"Loaded Master Matrix: {master_matrix.shape}, IDs: {len(master_ids)}")
    
    master_id2idx = {pid: i for i, pid in enumerate(master_ids)}
    num_target = len(target_item_ids) + 1
    aligned_matrix = torch.zeros((num_target, embed_dim), dtype=torch.float32)
    
    for i, target_id in enumerate(target_item_ids):
        # [중요] Tower 인덱스는 1부터
        current_idx = i + 1
        
        if target_id in master_id2idx:
            origin_idx = master_id2idx[target_id]
            # Master(origin) -> Tower(current=i+1)
            aligned_matrix[current_idx] = master_matrix[origin_idx]
        else:
            torch.nn.init.xavier_normal_(aligned_matrix[current_idx].unsqueeze(0))
            
        if aligned_matrix is not None:
            aligned_matrix = F.normalize(aligned_matrix, p=2, dim=1)
        
        return aligned_matrix
# -------------------------------------------------------------------------
# 2. Final Dataset (Slicing 로직 수정)
# -------------------------------------------------------------------------
class UserTowerDataset(Dataset):
    def __init__(self, processor, max_seq_len=50, is_training=True):
        self.processor = processor
        self.user_ids = processor.user_ids 
        self.max_len = max_seq_len
        self.is_training = is_training
        self.min_cut_len = 3      
        self.last_item_prob = 0.2

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        u_id_str = self.user_ids[idx]
        u_dense, u_cat = self.processor.get_user_tensor(u_id_str)
        
        processed_tokens = []
        processed_deltas = []
        
        # ... (시퀀스 로드 로직은 기존과 동일) ...
        if u_id_str in self.processor.seqs.index:
            seq_row = self.processor.seqs.loc[u_id_str]
            # (중략: 토큰 파싱 로직 동일)
            for i, d in zip(seq_row['sequence_ids'], seq_row['sequence_deltas']):
                 token = self.processor.item2id.get(i, 0)
                 if token == 0: continue
                 processed_tokens.append(token)
                 processed_deltas.append(d)

        seq_len = len(processed_tokens)

        # ------------------------------------------------------------------
        # [수정 1] All-Action을 위한 Slicing 로직 변경
        # ------------------------------------------------------------------
        input_seq = []
        target_seq = [] 

        if seq_len > 0:
            if self.is_training:
                # 1. 랜덤 컷이 가능한지 확인 (최소 2개는 있어야 Input/Target 나눔)
                can_sample = seq_len > self.min_cut_len

                # 2. 로직 분기
                # (자를 수 없거나 OR 80% 확률) -> 전체 사용
                if not can_sample or random.random() < 0.8:
                    # [수정 포인트] 전체를 다 쓸 때도 Shift는 필수입니다!
                    # 원본: [A, B, C, D]
                    # Input: [A, B, C]  (마지막 D는 정답으로 써야 하니까 Input에서 제외)
                    # Target: [B, C, D] (A 다음은 B니까 1번부터 시작)
                    input_seq = processed_tokens[:-1]
                    target_seq = processed_tokens[1:]
                
                else:
                    # 20% 확률 -> Random Cut
                    # randint 범위 에러 방지
                    max_cut = seq_len - 1
                    if max_cut < self.min_cut_len:
                        cut_idx = seq_len # 예외처리
                    else:
                        cut_idx = random.randint(self.min_cut_len, max_cut)
                    
                    # cut_idx+1 까지 가져와서 Input/Target 분리
                    full_slice = processed_tokens[:cut_idx+1]
                    input_seq = full_slice[:-1]
                    target_seq = full_slice[1:]
            
            else:
                # 평가 시
                input_seq = processed_tokens[:]
                target_seq = [0] * len(input_seq)
        # ------------------------------------------------------------------
        # Padding & Truncation (Window Sliding)
        # ------------------------------------------------------------------
        # Max Len에 맞춰 뒤에서부터 자름
        input_ids = input_seq[-self.max_len:]
        target_ids = target_seq[-self.max_len:]
        
        # Delta는 Input 길이에 맞춤
        input_deltas = processed_deltas[:len(input_seq)][-self.max_len:]

        return {
            'user_idx': torch.tensor(idx + 1, dtype=torch.long),
            'user_dense': u_dense, 'user_cat': u_cat,
            # 리스트 -> 텐서 변환
            'seq_ids': torch.tensor(input_ids, dtype=torch.long),
            'seq_deltas': torch.tensor(input_deltas, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long) # [변경] 시퀀스 형태
        }

def user_tower_collate_fn(batch):
    u_idx = torch.stack([b['user_idx'] for b in batch])
    u_dense = torch.stack([b['user_dense'] for b in batch])
    u_cat = torch.stack([b['user_cat'] for b in batch])
    
    # Pad Sequence (Batch First)
    seq_ids = pad_sequence([b['seq_ids'] for b in batch], batch_first=True, padding_value=0)
    seq_deltas = pad_sequence([b['seq_deltas'] for b in batch], batch_first=True, padding_value=0)
    target_ids = pad_sequence([b['target_ids'] for b in batch], batch_first=True, padding_value=0) # [변경] Padding
    
    seq_mask = (seq_ids != 0).long()
    
    # 평가용(Validation)을 위해 마지막 타겟 아이템 하나는 별도로 뽑아둘 수 있음
    # 학습시는 target_ids 전체를 씀
    last_target = torch.tensor([b['target_ids'][-1] if len(b['target_ids']) > 0 else 0 for b in batch], dtype=torch.long)

    return u_idx, u_dense, u_cat, seq_ids, seq_deltas, seq_mask, target_ids, last_target

# -------------------------------------------------------------------------
# 3. Model Components (Simplified for Convergence)
# -------------------------------------------------------------------------
# [수정] 초기 수렴을 위해 복잡한 Gating 대신 Robust한 MLP Fusion 사용
class RobustFusion(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        # 3개의 128차원 벡터를 Concat -> 384
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim) # 최종 출력 정규화 도움
        )

    def forward(self, v_gnn, v_seq, v_meta):
        combined = torch.cat([v_gnn, v_seq, v_meta], dim=-1)
        return self.fusion_mlp(combined)

class HybridUserTower(nn.Module):
    def __init__(self, num_users, num_items, gnn_emb_init, item_emb_init):
        super().__init__()
        self.embed_dim = 128
        
        # A. Pretrained Layers
        self.gnn_user_emb = nn.Embedding.from_pretrained(gnn_emb_init, freeze=True)
        self.gnn_projector = nn.Sequential(
            nn.Linear(gnn_emb_init.shape[1], 256), # 한번 확 넓혔다가
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 128), # 다시 줄임 (정보 압축 및 정렬)
            nn.LayerNorm(128)    # 마지막에 정규화 필수
        )   
        self.item_content_emb = nn.Embedding.from_pretrained(item_emb_init, freeze=True)
        
        # B. Sequence Layers
        self.time_emb = nn.Embedding(1001, 128)
        encoder_layer = nn.TransformerEncoderLayer(
        d_model=128, 
        nhead=4, # 헤드 수도 4 -> 8로 늘리면 더 좋습니다 (선택사항)
        dim_feedforward=512, 
        dropout=DROPOUT, 
        batch_first=True,
        norm_first=True
        )
        self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # C. Meta Layers
        self.channel_emb = nn.Embedding(2, 32)
        self.meta_mlp = nn.Sequential(nn.Linear(35, 128), nn.GELU(), nn.Linear(128, 128), nn.LayerNorm(128))
        
        # D. Fusion Layer (Simplified)
        self.fusion_layer = RobustFusion(dim=128)

    def forward(self, u_idx, seq_ids, seq_deltas, seq_mask, u_dense, u_cat):
        B, L = seq_ids.shape
        
        # 1. GNN Features (Static -> Broadcast)
        v_gnn = self.gnn_projector(self.gnn_user_emb(u_idx))
        v_gnn = F.normalize(v_gnn, p=2, dim=1)
        # [수정] (B, D) -> (B, L, D) 로 확장
        v_gnn_seq = v_gnn.unsqueeze(1).expand(-1, L, -1)
        
        # 2. Meta Features (Static -> Broadcast)
        cat_vec = self.channel_emb(u_cat)
        v_meta = self.meta_mlp(torch.cat([u_dense, cat_vec], dim=1))
        v_meta = F.normalize(v_meta, p=2, dim=1)
        # [수정] (B, D) -> (B, L, D) 로 확장
        v_meta_seq = v_meta.unsqueeze(1).expand(-1, L, -1)
        
        # 3. Sequence Features (Transformer with Causal Mask)
        seq_input = self.item_content_emb(seq_ids) * math.sqrt(self.embed_dim) + self.time_emb(seq_deltas.clamp(max=1000))
        
        # [핵심] Causal Mask 생성 (미래 참조 방지)
        # 상삼각행렬(Upper Triangular)을 -inf로 마스킹
        causal_mask = torch.triu(torch.ones(L, L, device=seq_ids.device) * float('-inf'), diagonal=1)
        
        # Padding Mask (Key Padding Mask)
        # PyTorch Transformer는 (B, L) 형태의 True/False 마스크를 받음 (True가 무시됨)
        key_padding_mask = (seq_mask == 0)

        # Transformer Forward
        # is_causal=True (PyTorch 2.0+) 혹은 mask=causal_mask 사용
        seq_out = self.seq_encoder(seq_input, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        
        # [수정] Attention Pooling 대신, 일단 모든 스텝의 출력을 사용 (All-Action)
        v_seq = F.normalize(seq_out, p=2, dim=2) # (B, L, D)

        # 4. Final Fusion (All Steps)
        # (B, L, D) + (B, L, D) + (B, L, D) -> (B, L, D)
        # RobustFusion(MLP)은 마지막 차원만 맞으면 3D 텐서도 처리 가능
        output = self.fusion_layer(v_gnn_seq, v_seq, v_meta_seq)
        
        return F.normalize(output, p=2, dim=2) # (B, L, D) 리턴

# -------------------------------------------------------------------------
# 4. Improved Loss Function (Mathematical Fix)
# -------------------------------------------------------------------------
def logq_correction_loss(user_emb, item_emb, pos_item_ids, item_probs, temperature=0.07, lambda_logq=0.0):
    # 1. 내적 (Cosine Similarity)
    scores = torch.matmul(user_emb, item_emb.T)
    
    # 2. LogQ Correction (먼저 수행)
    if lambda_logq > 0.0:
        log_q = torch.log(item_probs[pos_item_ids] + 1e-9).view(1, -1) # [1, Batch]
        scores = scores - (lambda_logq * log_q)

    # 3. Temperature Scaling (나중에 수행)
    logits = scores / temperature

    # 4. In-batch Masking
    is_collision = (pos_item_ids.unsqueeze(1) == pos_item_ids.unsqueeze(0))
    mask = is_collision.fill_diagonal_(False)
    logits = logits.masked_fill(mask, -1e4) # FP16 Safe value

    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)
def evaluate_recall_multi_k(model, processor, target_df_path, k_list=[20, 100, 500], batch_size=1024):
    model.eval()
    
    # Target Dictionary 로드
    target_df = pd.read_parquet(target_df_path)
    # customer_id가 index인 dict 생성
    target_dict = target_df.set_index('customer_id')['target_ids'].to_dict()
    
    # Valid Dataset 로더
    val_loader = DataLoader(
        UserTowerDataset(processor, is_training=False), 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=user_tower_collate_fn
    )
    
    # [핵심] 전체 아이템 벡터 캐싱
    # processor.item_ids는 train_proc과 동기화되어 있으므로, 모델의 임베딩 순서와 일치함.
    with torch.no_grad():
        all_item_vecs = F.normalize(
            model.item_content_emb(torch.arange(1, len(processor.item_ids)+1).to(DEVICE)), 
            p=2, dim=1
        )

    hit_counts = {k: 0 for k in k_list}
    total_users = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Unpacking (Valid 모드일 때 반환값 개수 주의)
            u_idx, u_dense, u_cat, seq_ids, seq_deltas, seq_mask, _, _ = [x.to(DEVICE) for x in batch]
            
            # User ID String 복원
            batch_uids = [processor.user_ids[i-1] for i in u_idx.cpu().numpy()]
            
            # Target이 있는 유저만 필터링
            valid_idx_list = [i for i, uid in enumerate(batch_uids) if uid in target_dict]
            if not valid_idx_list: continue
            
            v_idx = torch.tensor(valid_idx_list).to(DEVICE)
            
            # 1. Forward (Last Hidden State)
            seq_out = model(
                u_idx[v_idx], seq_ids[v_idx], seq_deltas[v_idx], 
                seq_mask[v_idx], u_dense[v_idx], u_cat[v_idx]
            )
            
            # Last Valid Step 추출
            lengths = seq_mask[v_idx].sum(dim=1)
            last_indices = (lengths - 1).clamp(min=0)
            
            batch_range = torch.arange(seq_out.size(0), device=DEVICE)
            last_user_vecs = seq_out[batch_range, last_indices]
            
            # 2. Similarity Search (Dot Product)
            # (Batch, Dim) @ (Dim, Num_Items) -> (Batch, Num_Items)
            scores = torch.matmul(last_user_vecs, all_item_vecs.T)
            
            # Top-K 추출
            _, topk_indices = torch.topk(scores, k=max(k_list), dim=1)
            pred_ids = (topk_indices + 1).cpu().numpy() # Index -> ItemID(1~)
            
            # 3. Hit Calculation
            for i, original_idx in enumerate(valid_idx_list):
                u_id = batch_uids[original_idx]
                actual_item_ids = target_dict[u_id] # 정답 아이템들 (String List)
                
                # String ID -> Integer Index 변환 (processor.item2id 사용)
                # 만약 Valid Set에 Train에 없던 신규 아이템이 있다면 여기서 걸러짐 (안전!)
                actual_indices = set(
                    processor.item2id[tid] for tid in actual_item_ids if tid in processor.item2id
                )
                
                if not actual_indices: continue

                # 성능 최적화된 Hit Check
                # pred_ids 상위 k개 중에 actual이 하나라도 있는지 확인
                for k in k_list:
                    # numpy array slicing은 빠름
                    preds_k = pred_ids[i, :k]
                    # 교집합이 있으면 Hit
                    if not actual_indices.isdisjoint(preds_k):
                        hit_counts[k] += 1
                        
                total_users += 1
    
    # Metric 집계
    metrics = {f"R@{k}": (hit_counts[k] / total_users if total_users > 0 else 0.0) for k in k_list}
    logger.log(1, f"📊 Eval Result: {metrics}")
    
    model.train() # 다시 Train 모드로 복귀
    return metrics
# -------------------------------------------------------------------------
# 5. Training Loop
# -------------------------------------------------------------------------
def train_phase_2_5_emergency_fix():
    logger.log(1, "🚀 Phase 2.5: Emergency Fix Running...")
    
    # [수정] Scaler를 Train에서 Valid로 넘겨줌
    train_proc = FeatureProcessor(USER_FEAT_PATH_PQ, ITEM_FEAT_PATH_PQ, SEQ_DATA_PATH_PQ, scaler=None)
    valid_proc = FeatureProcessor(USER_VAL_FEAT_PATH, ITEM_FEAT_PATH_PQ, SEQ_VAL_DATA_PATH, scaler=train_proc.user_scaler)
    valid_proc.item2id, valid_proc.item_ids = train_proc.item2id, train_proc.item_ids

    '''

    
        # 4. 저장된 가중치(State Dict) 덮어씌우기
    if os.path.exists(SAVE_PATH_BEST_PREV):
        checkpoint = torch.load(SAVE_PATH_BEST_PREV, map_location=DEVICE)
        model.load_state_dict(checkpoint, strict=True) # strict=True: 구조가 완벽히 일치해야 함
        print(f"✅ Successfully loaded model from: {SAVE_PATH_BEST_PREV}")
    else:
        print(f"❌ Model file not found: {SAVE_PATH_BEST_PREV}")
        
    
    # Optimizer
    optimizer = optim.AdamW([
    # 상위 레이어 (유저 타워)
        {'params': model.seq_encoder.parameters(), 'lr': 1e-5},
        {'params': model.fusion_layer.parameters(), 'lr': 1e-5},
        {'params': model.meta_mlp.parameters(), 'lr': 1e-5},
        {'params': model.gnn_projector.parameters(), 'lr': 1e-5},
        
        # 하위 레이어 (거대 임베딩) - 여기가 핵심!
        {'params': model.gnn_user_emb.parameters(), 'lr': 5e-6},
        {'params': model.item_content_emb.parameters(), 'lr': 5e-6},
    ], weight_decay=1e-4)
    '''
    
    
    
    num_users = len(train_proc.user_ids) + 1
    num_items = len(train_proc.item_ids) + 1
    dummy_gnn_tensor = torch.zeros((num_users, 64))
    dummy_item_tensor = torch.zeros((num_items, 128))

    model = HybridUserTower(
        num_users, 
        num_items, 
        gnn_emb_init=dummy_gnn_tensor, 
        item_emb_init=dummy_item_tensor
    ).to(DEVICE)
   
    model = load_and_align_embeddings(model, train_proc, model_dir=MODEL_DIR, device=DEVICE)
    verify_embedding_alignment(model, train_proc, model_dir=MODEL_DIR)

    model = load_and_align_gnn_items(model, train_proc, base_dir=BASE_DIR, device=DEVICE)
    verify_gnn_alignment(model, train_proc, base_dir=BASE_DIR)
    
    
    model_params = filter(lambda p: p.requires_grad, model.parameters())

# [추천 설정]
    optimizer = optim.AdamW(
        model_params, 
        lr=5e-4,           # [변경] 1e-4는 너무 작습니다. 3e-4 ~ 5e-4 추천 (Effective Batch가 1.5만 이므로)
        betas=(0.9, 0.98), # [고급] 대규모 배치에서는 beta2를 0.999 -> 0.98로 낮추면 안정성이 올라갑니다.
        weight_decay=0.01, # [변경] 1e-4는 너무 작습니다. AdamW의 기본값(0.01)이 일반화 성능에 더 좋습니다.
        eps=1e-6           # FP16(AMP) 사용 시 수치 안정성 확보
    )
    train_loader = DataLoader(UserTowerDataset(train_proc, is_training=True), 
                              batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=user_tower_collate_fn)

    

    total_steps = len(train_loader) * EPOCHS 
    warmup_steps = int(total_steps * 0.1) # 전체의 10%를 웜업

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler('cuda')
    item_probs = train_proc.get_logq_probs(DEVICE)
    best_r100 = 0.0


    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            u_idx, u_dense, u_cat, seq_ids, seq_deltas, seq_mask, target_ids, _ = [x.to(DEVICE) for x in batch]
            
            # [추가 1] 그라디언트 초기화 (필수!)
            optimizer.zero_grad() 

            # 1. Forward
            with torch.amp.autocast('cuda'):
                user_seq_vecs = model(u_idx, seq_ids, seq_deltas, seq_mask, u_dense, u_cat)
                
                valid_mask = (target_ids != 0) 
                active_user_vecs = user_seq_vecs[valid_mask] 
                active_target_ids = target_ids[valid_mask]
                active_item_vecs = F.normalize(model.item_content_emb(active_target_ids), p=2, dim=1)
                
                loss = logq_correction_loss(
                    active_user_vecs, 
                    active_item_vecs, 
                    active_target_ids, 
                    item_probs, 
                    TEMPERATURE, 
                    LAMBDA_LOGQ 
                )

            scaler.scale(loss).backward()
            
            # [기존] Unscale은 step 전에 명시적으로 해주는 게 안전함 (Gradient Clipping 등을 위해)
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # (선택) 안전장치

            scaler.step(optimizer)
            scaler.update()
            
            # [추가 2] 스케줄러 업데이트 (이게 있어야 LR이 오름!)
            scheduler.step() 
            
            total_loss += loss.item()
            
            # 이제 lr이 정상적으로 0.00e+00 -> 1.25e-05 ... 식으로 오를 겁니다.
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})

        avg_loss = total_loss / len(train_loader)
        logger.log(1, f"📊 Epoch {epoch+1} Result: Avg Loss {avg_loss:.4f}")

        # 정기 평가
        metrics = evaluate_recall_multi_k(
            model, 
            valid_proc, 
            TARGET_VAL_PATH, 
            k_list=[20, 100, 500], 
            batch_size=256
        )
        
        if metrics['R@100'] > best_r100:
            best_r100 = metrics['R@100']
            torch.save(model.state_dict(), SAVE_PATH_BEST)
            logger.log(1, f"🌟 New Best R@100: {best_r100:.4f} - Model Saved!")




def test_dataset_train():
    logger.log(1, "🚀 Phase 2.5: Emergency Fix Running...")
    
    # [수정] Scaler를 Train에서 Valid로 넘겨줌
    train_proc = FeatureProcessor(USER_FEAT_PATH_PQ, ITEM_FEAT_PATH_PQ, SEQ_DATA_PATH_PQ, scaler=None)
    valid_proc = FeatureProcessor(USER_VAL_FEAT_PATH, ITEM_FEAT_PATH_PQ, SEQ_VAL_DATA_PATH, scaler=train_proc.user_scaler)
    valid_proc.item2id, valid_proc.item_ids = train_proc.item2id, train_proc.item_ids
    
    num_users = len(train_proc.user_ids) + 1
    num_items = len(train_proc.item_ids) + 1
    dummy_gnn_tensor = torch.zeros((num_users, 64))
    dummy_item_tensor = torch.zeros((num_items, 128))

    model = HybridUserTower(
        num_users, 
        num_items, 
        gnn_emb_init=dummy_gnn_tensor, 
        item_emb_init=dummy_item_tensor
    ).to(DEVICE)
   
    model = load_and_align_embeddings(model, train_proc, model_dir=MODEL_DIR, device=DEVICE)
    verify_embedding_alignment(model, train_proc, model_dir=MODEL_DIR)

    model = load_and_align_gnn_items(model, train_proc, base_dir=BASE_DIR, device=DEVICE)
    verify_gnn_alignment(model, train_proc, base_dir=BASE_DIR)
    
    OVERFIT_BATCH_SIZE = 128  # 작게 설정 (확실한 암기 유도)
    TEST_EPOCHS = 50          # 충분히 반복
    TEST_LR = 1e-3            # 학습률을 높게 설정 (빠른 수렴)
    TEMP_TEST = 0.2           # 온도를 높여서 난이도 하향
        
    full_dataset = UserTowerDataset(train_proc, is_training=True)
    mini_dataset = torch.utils.data.Subset(full_dataset, range(OVERFIT_BATCH_SIZE))

    # 2. Mini Loader 생성 (Shuffle 끔 -> 순서 고정해서 암기 돕기)
    mini_loader = DataLoader(
        mini_dataset, 
        batch_size=OVERFIT_BATCH_SIZE, 
        shuffle=False, 
        collate_fn=user_tower_collate_fn
    )
    trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Total Trainable Params: {len(trainable_params)}")

    # User Tower의 핵심 부품이 포함되어 있는지 확인
    if any('seq_encoder' in n for n in trainable_params):
        print("✅ User Tower (SeqEncoder) is Trainable")
    else:
        print("❌ User Tower is FROZEN! (This is the bug)")

    if any('fusion_layer' in n for n in trainable_params):
        print("✅ User Tower (Fusion) is Trainable")
    else:
        print("❌ User Tower Fusion is FROZEN!")
    # 3. Optimizer 새로 정의 (Scheduler 없이 단순하게)
    optimizer_test = optim.AdamW(model.parameters(), lr=TEST_LR, weight_decay=0.0) # Decay 0으로 설정 (과적합 유도)
    scaler = torch.amp.GradScaler('cuda')
    item_probs = train_proc.get_logq_probs(DEVICE)
    # 4. Test Loop
    for epoch in range(TEST_EPOCHS):
        model.train()
        total_loss = 0
        
        # 배치는 딱 1번만 돔
        for batch in mini_loader:
            u_idx, u_dense, u_cat, seq_ids, seq_deltas, seq_mask, target_ids, _ = [x.to(DEVICE) for x in batch]
            
            optimizer_test.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # Forward
                user_seq_vecs = model(u_idx, seq_ids, seq_deltas, seq_mask, u_dense, u_cat)
                
                # Masking & Flattening
                valid_mask = (target_ids != 0)
                active_user_vecs = user_seq_vecs[valid_mask]
                active_target_ids = target_ids[valid_mask]
                active_item_vecs = F.normalize(model.item_content_emb(active_target_ids), p=2, dim=1)
                
                # Loss Calculation (LogQ 끄고, 온도 높임)
                loss = logq_correction_loss(
                    active_user_vecs, 
                    active_item_vecs, 
                    active_target_ids, 
                    item_probs, 
                    temperature=TEMP_TEST, 
                    lambda_logq=0.0 
                )
            
            # Backward
            scaler.scale(loss).backward()
            scaler.step(optimizer_test)
            scaler.update()
            
            total_loss = loss.item()

        # 로그 출력 (10 에포크마다)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{TEST_EPOCHS} | Loss: {total_loss:.6f}")

    print("✅ Test Finished.")
def check_shape_mismatch():
    print("🚑 [Emergency Check] Shape & Alignment Analysis")
    
    # 1. 임베딩 로드
    try:
        vectors = torch.load(ITEM_MATRIX_PATH, map_location='cpu')
        if isinstance(vectors, dict):
            # 만약 state_dict라면 가중치 추출
            vectors = vectors.get('weight', vectors.get('item_content_emb.weight'))
        
        print(f"📊 Embedding Matrix Shape: {vectors.shape}")
        # 예: torch.Size([105542, 128]) -> 10만 5천개
    except Exception as e:
        print(f"❌ Matrix Load Error: {e}")
        return

    # 2. 메타데이터 로드
    df = pd.read_parquet(ITEM_META_PATH)
    print(f"📄 Metadata DataFrame Shape: {df.shape}")
    # 예: (50000, 25) -> 5만개
    
    print(f"📋 Metadata Columns: {df.columns.tolist()}")

    # 3. 비교 분석
    n_vec = vectors.shape[0]
    n_meta = df.shape[0]
    
    print("\n⚖️ [Conclusion]")
    if n_vec != n_meta:
        print(f"❌ MISMATCH DETECTED! (Diff: {abs(n_vec - n_meta)})")
        print("👉 임베딩은 {}개인데, 데이터는 {}개입니다.".format(n_vec, n_meta))
        print("👉 순서가 보장되지 않으므로, 'ID Mapping 파일'이 없으면 이 임베딩은 쓸 수 없습니다.")
    else:
        print("✅ Counts match. (But order might still be wrong)")




# 생략된 함수들(align, evaluate 등)은 기존 코드와 동일하다고 가정하거나 위에 정의된 것 사용
if __name__ == "__main__":
    #train_phase_2_5_warmup_finetune()
    #train_phase_2_5_fresh_start_v2()
    train_phase_2_5_emergency_fix()
     #test_dataset_train()
    #check_embedding_sanity()
     #check_shape_mismatch()