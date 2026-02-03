import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import os
BASE_DIR = r"D:\trainDataset\localprops"
RAW_FILE_PATH = os.path.join(BASE_DIR, "transactions_train_filtered.json")

# 결과 저장 경로 (Parquet + JSON)
USER_FEAT_PATH_PQ = os.path.join(BASE_DIR, "features_user.parquet")
USER_FEAT_PATH_JS = os.path.join(BASE_DIR, "features_user.json")

ITEM_FEAT_PATH_PQ = os.path.join(BASE_DIR, "features_item.parquet")
ITEM_FEAT_PATH_JS = os.path.join(BASE_DIR, "features_item.json")

SEQ_DATA_PATH_PQ = os.path.join(BASE_DIR, "features_sequence.parquet")
SEQ_DATA_PATH_JS = os.path.join(BASE_DIR, "features_sequence.json")

# 전체 히스토리 저장 경로
WEEKLY_HISTORY_PATH = os.path.join(BASE_DIR, "history_weekly_sales.parquet")
MONTHLY_HISTORY_PATH = os.path.join(BASE_DIR, "history_monthly_sales.parquet")

class FeatureProcessor:
    def __init__(self, user_path, item_path, seq_path):
        print("🔄 Loading & Scaling Features...")
        
        # 1. Load Parquet
        self.users = pd.read_parquet(user_path).set_index('customer_id')
        self.items = pd.read_parquet(item_path).set_index('article_id')
        self.seqs = pd.read_parquet(seq_path).set_index('customer_id')
        
        # -------------------------------------------------------
        # 2. Prepare Scalers (전처리된 Log값들을 정규화)
        # -------------------------------------------------------
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
        
        # User Dense Columns
        self.u_dense_cols = ['user_avg_price_log', 'total_cnt_log', 'recency_log']
        # Item Dense Columns (Advanced Features 포함)
        self.i_dense_cols = [
            'pop_1w_log', 'pop_1m_log', 
            'velocity_1w', 'velocity_1m', 
            'days_since_release_log', 'avg_item_price_log'
        ]
        
        # -------------------------------------------------------
        # 3. Apply Scaling & Store
        # -------------------------------------------------------
        # (중요) Cross Feature 계산을 위해 Raw(Log) 값은 따로 보관해야 함
        # Python Dictionary는 느리므로 DataFrame 상태로 유지하되, 
        # get_item_tensor 호출 시 Scaled 값을 반환하도록 미리 계산해둠.
        
        # User Scaling
        self.users_scaled = self.users.copy()
        self.users_scaled[self.u_dense_cols] = self.user_scaler.fit_transform(self.users[self.u_dense_cols])
        
        # Item Scaling (raw_probability는 스케일링 제외!)
        self.items_scaled = self.items.copy()
        self.items_scaled[self.i_dense_cols] = self.item_scaler.fit_transform(self.items[self.i_dense_cols])
        
        print(f"✅ User Features: {len(self.users)}, Item Features: {len(self.items)}")

    def get_user_tensor(self, user_ids):
        """User Tower Input: Scaled Dense Features"""
        # loc으로 가져오기
        batch_data = self.users_scaled.loc[user_ids]
        
        # Dense Features
        dense = torch.tensor(batch_data[self.u_dense_cols].values, dtype=torch.float32)
        
        # Categorical (Preferred Channel: 1,2 -> 0,1)
        cat = torch.tensor(batch_data['preferred_channel'].values - 1, dtype=torch.long)
        
        return dense, cat

    def get_item_tensor(self, item_ids):
        """GDCN Input: Scaled Dense Features (1w, 1m, velocity, release...)"""
        batch_data = self.items_scaled.loc[item_ids]
        return torch.tensor(batch_data[self.i_dense_cols].values, dtype=torch.float32)

    def get_raw_probability(self, item_ids):
        """User Tower Loss용 (LogQ Correction)"""
        return torch.tensor(self.items.loc[item_ids]['raw_probability'].values, dtype=torch.float32)

    def get_cross_features(self, user_ids, item_ids):
        """
        Cross Features 계산
        (중요) 스케일링 된 값이 아니라, 원본 Log 값을 써야 물리적 의미가 맞음!
        """
        # self.users, self.items는 스케일링 전 원본(Log applied)
        u_raw = self.users.loc[user_ids]
        i_raw = self.items.loc[item_ids]
        
        # 1. Price Gap: Item Price - User Avg Price (둘 다 Log 상태)
        price_gap = i_raw['avg_item_price_log'].values - u_raw['user_avg_price_log'].values
        
        # 2. Trend Interaction: Item Velocity * User Activity
        # 활동적인 유저(cnt high)가 가속도(velocity) 높은 아이템에 반응
        # velocity는 스케일링 전에도 -1~5 범위이므로 그대로 사용
        trend_interaction_1w = i_raw['velocity_1w'].values * u_raw['total_cnt_log'].values
        trend_interaction_1m = i_raw['velocity_1m'].values * u_raw['total_cnt_log'].values
        
        # (B, 3)
        cross_feats = np.stack([price_gap, trend_interaction_1w, trend_interaction_1m], axis=1)
        return torch.tensor(cross_feats, dtype=torch.float32)

# ==========================================
# 3. Dataset Classes
# ==========================================
class UserTowerDataset(Dataset):
    def __init__(self, user_ids, processor, max_seq_len=50):
        self.user_ids = user_ids
        self.processor = processor
        self.max_len = max_seq_len
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        u_id = self.user_ids[idx]
        
        # 1. User Features
        u_dense, u_cat = self.processor.get_user_tensor([u_id])
        
        # 2. Sequence
        try:
            seq_row = self.processor.seqs.loc[u_id]
            seq_ids = seq_row['sequence_ids'][-self.max_len:]
            seq_deltas = seq_row['sequence_deltas'][-self.max_len:]
        except KeyError: # 시퀀스 없는 유저 예외처리
            seq_ids, seq_deltas = [], []
            
        return {
            'user_dense': u_dense.squeeze(0),
            'user_cat': u_cat.squeeze(0),
            'seq_ids': torch.tensor(seq_ids, dtype=torch.long),
            'seq_deltas': torch.tensor(seq_deltas, dtype=torch.long)
        }

class RerankerDataset(Dataset):
    def __init__(self, interactions_df, processor, max_seq_len=50):
        self.data = interactions_df
        self.processor = processor
        self.max_len = max_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        u_id = str(row['user_id'])
        i_id = str(row['item_id'])
        label = row['label']
        
        # 1. User & Item Tensors (Scaled)
        u_dense, u_cat = self.processor.get_user_tensor([u_id])
        i_dense = self.processor.get_item_tensor([i_id])
        
        # 2. Cross Features (Calculated from Raw Log)
        cross_feats = self.processor.get_cross_features([u_id], [i_id])
        
        # 3. GDCN Input Concatenation
        # User(3) + Item(6) + Cross(3) = 12 Dense Features
        gdcn_dense = torch.cat([u_dense.squeeze(0), i_dense.squeeze(0), cross_feats.squeeze(0)], dim=0)
        
        # 4. Sequence (Attention용)
        try:
            seq_row = self.processor.seqs.loc[u_id]
            seq_ids = torch.tensor(seq_row['sequence_ids'][-self.max_len:], dtype=torch.long)
        except KeyError:
            seq_ids = torch.tensor([], dtype=torch.long)

        return {
            'gdcn_dense': gdcn_dense,
            'user_cat': u_cat.squeeze(0),
            'seq_ids': seq_ids,
            'target_item_id': torch.tensor(int(i_id) if i_id.isdigit() else 0), # ID Mapping 필요
            'label': torch.tensor(label, dtype=torch.float32)
        }

# Collate Function
def reranker_collate_fn(batch):
    dense = torch.stack([b['gdcn_dense'] for b in batch])
    cat = torch.stack([b['user_cat'] for b in batch])
    label = torch.stack([b['label'] for b in batch])
    target_item = torch.stack([b['target_item_id'] for b in batch])
    
    seq_ids = pad_sequence([b['seq_ids'] for b in batch], batch_first=True, padding_value=0)
    seq_mask = (seq_ids != 0).long()
    
    return dense, cat, seq_ids, seq_mask, target_item, label

# ==========================================
# Main Execution Check
# ==========================================
if __name__ == "__main__":
    # Test Loading
    try:
        processor = FeatureProcessor(
            USER_FEAT_PATH_PQ, ITEM_FEAT_PATH_PQ, SEQ_DATA_PATH_PQ
        )
        
        # Mock Data
        mock_interactions = pd.DataFrame({
            'user_id': processor.users.index[:5],
            'item_id': processor.items.index[:5],
            'label': [1, 0, 1, 0, 1]
        })
        
        ds = RerankerDataset(mock_interactions, processor)
        loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=reranker_collate_fn)
        
        for batch in loader:
            dense, cat, seq, mask, target, lbl = batch
            print("\n✅ Reranker Batch Check:")
            print(f" - Dense Input Shape: {dense.shape} (Batch, Features)")
            print(f" - Sequence Shape: {seq.shape}")
            break
            
    except Exception as e:
        print(f"⚠️ Error during test: {e}")
    
    
    
    '''
    
    
    from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class UserTowerDataset(Dataset):
    def __init__(self, user_ids, processor, max_seq_len=50):
        self.user_ids = user_ids # 학습할 유저 리스트
        self.processor = processor
        self.max_len = max_seq_len
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        u_id = self.user_ids[idx]
        
        # 1. Dense & Cat Features (Pre-computed)
        # processor 내부적으로 loc을 쓰지만, 실제론 array indexing이 더 빠름
        # 여기서는 예시로 processor 메소드 호출
        user_dense, user_cat = self.processor.get_user_tensor([u_id])
        
        # 2. Sequence Data
        seq_row = self.processor.seqs.loc[u_id]
        seq_ids = seq_row['sequence_ids'][-self.max_len:]
        seq_deltas = seq_row['sequence_deltas'][-self.max_len:]
        
        # Tensor 변환
        seq_ids_tensor = torch.tensor(seq_ids, dtype=torch.long)
        seq_deltas_tensor = torch.tensor(seq_deltas, dtype=torch.long)
        
        return {
            'user_dense': user_dense.squeeze(0), # (2,)
            'user_cat': user_cat.squeeze(0),     # (1,)
            'seq_ids': seq_ids_tensor,           # (L,)
            'seq_deltas': seq_deltas_tensor      # (L,)
        }

# Collate Fn: 배치 단위 패딩 처리
def user_tower_collate_fn(batch):
    user_dense = torch.stack([b['user_dense'] for b in batch])
    user_cat = torch.stack([b['user_cat'] for b in batch])
    
    # Sequence Padding (뒤에 0 채움)
    seq_ids = pad_sequence([b['seq_ids'] for b in batch], batch_first=True, padding_value=0)
    seq_deltas = pad_sequence([b['seq_deltas'] for b in batch], batch_first=True, padding_value=0)
    
    # Mask 생성 (Padding 부분은 0, 실제 데이터는 1)
    seq_mask = (seq_ids != 0).long()
    
    return user_dense, user_cat, seq_ids, seq_deltas, seq_mask
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    '''
    
    '''
    class RerankerDataset(Dataset):
    def __init__(self, interactions_df, processor, max_seq_len=50):
        """
        interactions_df: [user_id, item_id, label, retrieval_score(Optional)]
        """
        self.data = interactions_df
        self.processor = processor
        self.max_len = max_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        u_id = row['user_id']
        i_id = row['item_id']
        label = row['label']
        
        # 1. User Features (Dense, Cat, Sequence) - Tower와 동일
        u_dense, u_cat = self.processor.get_user_tensor([u_id])
        
        seq_row = self.processor.seqs.loc[u_id]
        seq_ids = torch.tensor(seq_row['sequence_ids'][-self.max_len:], dtype=torch.long)
        # Re-ranker에서는 보통 Delta까지는 안 쓰거나, 쓰더라도 Attention Mask 용도로 씀
        
        # 2. Item Features (Dense)
        i_dense = self.processor.get_item_tensor([i_id]) # Velocity, Steady, Price
        
        # 3. Cross Features (Price Gap, Trend Interaction) - ★ 핵심
        cross_feats = self.processor.get_cross_features([u_id], [i_id])
        
        # 4. Retrieval Score (Two-Tower에서 나온 점수, 있다면)
        # ret_score = torch.tensor([row['retrieval_score']], dtype=torch.float32)
        
        # 5. GDCN용 Dense Vector 통합 (User Dense + Item Dense + Cross)
        # (B, 2) + (B, 4) + (B, 2) -> (B, 8)
        gdcn_dense_input = torch.cat([u_dense, i_dense, cross_feats], dim=1)
        
        return {
            'gdcn_dense': gdcn_dense_input.squeeze(0), # MLP/CrossNet 입력
            'user_cat': u_cat.squeeze(0),              # Embedding 입력
            'seq_ids': seq_ids,                        # DIN Attention 입력
            'target_item_id': torch.tensor(int(i_id)), # DIN Attention Query
            'label': torch.tensor(label, dtype=torch.float32)
        }

def reranker_collate_fn(batch):
    dense = torch.stack([b['gdcn_dense'] for b in batch])
    cat = torch.stack([b['user_cat'] for b in batch])
    label = torch.stack([b['label'] for b in batch])
    target_item = torch.stack([b['target_item_id'] for b in batch])
    
    seq_ids = pad_sequence([b['seq_ids'] for b in batch], batch_first=True, padding_value=0)
    seq_mask = (seq_ids != 0).long()
    
    return dense, cat, seq_ids, seq_mask, target_item, label
    
    
    '''
    
    '''
    if __name__ == "__main__":
    # 1. Processor 초기화 (한 번만 로딩)
    processor = FeatureProcessor(
        user_path="features_user.parquet", 
        item_path="features_item.parquet", 
        seq_path="features_sequence.parquet"
    )
    
    # 2. User Tower 학습용
    train_users = ["u1", "u2", "u3"] # 실제 ID 리스트
    tower_ds = UserTowerDataset(train_users, processor)
    tower_loader = torch.utils.data.DataLoader(tower_ds, batch_size=32, collate_fn=user_tower_collate_fn)
    
    # 3. Re-ranker 학습용 (Positive + Negative Samples)
    # 실제로는 Retrieval 결과나 Random Negative로 생성된 DF 필요
    interaction_data = pd.DataFrame({
        'user_id': ['u1', 'u1', 'u2'],
        'item_id': ['i100', 'i200', 'i100'], # i200은 Negative 가정
        'label': [1, 0, 1]
    })
    
    rerank_ds = RerankerDataset(interaction_data, processor)
    rerank_loader = torch.utils.data.DataLoader(rerank_ds, batch_size=32, collate_fn=reranker_collate_fn)
    
    # Test Output
    for batch in rerank_loader:
        dense, cat, seq, mask, target, lbl = batch
        print("GDCN Input Dense Shape:", dense.shape) # (32, 8) -> 8개 피처가 합쳐짐
        print("Cross Features included (Price Gap etc.)")
        break
    
    '''