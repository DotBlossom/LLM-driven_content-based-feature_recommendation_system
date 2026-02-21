import os
import pandas as pd
import numpy as np
import gc
import json
import ijson
from datetime import timedelta
from pandarallel import pandarallel
from sklearn.preprocessing import StandardScaler

import torch

# ==========================================
# 0. Global Settings
# ==========================================
# 16GB RAM 기준: Worker 2~3개 추천 (안전하게 2)
WORKER_COUNT = 2
pandarallel.initialize(progress_bar=True, nb_workers=WORKER_COUNT, verbose=1)

BASE_DIR = r"D:\trainDataset\localprops"
RAW_FILE_PATH = os.path.join(BASE_DIR, "transactions_train_filtered.json")
CACHE_FILE_PATH = os.path.join(BASE_DIR, "cached_transactions_1yr.parquet")

path_case = {
    "train" : ["features_user","features_item", "features_sequence", "history_weekly_sales", "history_monthly_sales" ] ,
    "valid" :  ["features_user_val","features_item_val", "features_sequence_val", "history_weekly_sales_val", "history_monthly_sales_val" ]        
}
# f'path_case["valid"][0].parquet'
# Output Paths
USER_FEAT_PATH_PQ = os.path.join(BASE_DIR, "features_user_w_meta.parquet")
USER_FEAT_PATH_JS = os.path.join(BASE_DIR, "features_user_w_meta.json")
USER_FEAT_VAL_PATH_PQ = os.path.join(BASE_DIR, "features_user_w_meta_val.parquet")
USER_FEAT_VAL_PATH_JS = os.path.join(BASE_DIR, "features_user_w_meta_val.json")
ITEM_FEAT_PATH_PQ = os.path.join(BASE_DIR, "features_item.parquet")
ITEM_FEAT_PATH_JS = os.path.join(BASE_DIR, "features_item.json")
SEQ_DATA_PATH_PQ = os.path.join(BASE_DIR, "features_sequence.parquet")
SEQ_DATA_PATH_JS = os.path.join(BASE_DIR, "features_sequence.json")
WEEKLY_HISTORY_PATH = os.path.join(BASE_DIR, "history_weekly_sales.parquet")
MONTHLY_HISTORY_PATH = os.path.join(BASE_DIR, "history_monthly_sales.parquet")
USER_META_PATH = os.path.join(BASE_DIR, "customers.csv")

# Date Config
TRAIN_START_DATE = pd.to_datetime("2019-09-23")
DATASET_MAX_DATE = pd.to_datetime("2020-09-22")
VALID_START_DATE = pd.to_datetime("2020-09-16") 





def make_validation_target_file(full_df, valid_start_date, max_date, save_path):
    """
    검증 기간 동안의 실제 구매 내역을 유저별 리스트로 정렬하여 저장합니다.
    """
    print(f"🎯 [Valid Target] Extracting ground truth ({valid_start_date.date()} ~ {max_date.date()})...")
    
    # 1. 검증 기간(마지막 1주일) 데이터만 필터링
    valid_mask = (full_df['t_dat'] >= valid_start_date) & (full_df['t_dat'] <= max_date)
    valid_target_df = full_df.loc[valid_mask].copy()
    
    if valid_target_df.empty:
        print("⚠️ Warning: 해당 기간에 구매 데이터가 없습니다. 날짜 설정을 확인하세요.")
        return None

    # 2. 유저별 구매 리스트 생성
    # - 한 유저가 일주일 동안 여러 아이템을 샀을 수 있으므로 list로 묶습니다.
    # - 결과 형태: customer_id | target_ids (list)
    ground_truth = valid_target_df.groupby('customer_id')['article_id'].apply(list).reset_index()
    ground_truth.columns = ['customer_id', 'target_ids']

    # 3. 저장 (TARGET_VAL_PATH)
    print(f" 💾 Saving Ground Truth to: {save_path}")
    ground_truth.to_parquet(save_path, index=False)
    
    print(f" ✅ Extraction Complete! Total Users in Target: {len(ground_truth)}")
    return ground_truth




# ==========================================
# 1. Utility Functions
# ==========================================
def save_dataframe(df, parquet_path, json_path):
    print(f"   💾 Saving to {parquet_path} ...")
    df.to_parquet(parquet_path, index=False)
    # df.to_json(json_path, orient='records', force_ascii=False) # 필요 시 주석 해제

def load_data():
    """
    ijson을 사용하여 메모리 폭발 없이 데이터를 로드하고,
    reduce_mem_usage 함수 없이 명시적 타입 변환으로 최적화합니다.
    """
    # 1. Cache Hit
    if os.path.exists(CACHE_FILE_PATH):
        print(f"\n🚀 [Cache Hit] {CACHE_FILE_PATH}")
        df = pd.read_parquet(CACHE_FILE_PATH)
        
    # 2. Cache Miss (Streaming Load)
    else:
        print(f"\n🐢 [Cache Miss] Streaming load with ijson...")
        
        chunk_list = []
        chunk_size = 100000
        buffer = []
        
        with open(RAW_FILE_PATH, 'rb') as f:
            parser = ijson.items(f, 'item')
            
            for obj in parser:
                buffer.append(obj)
                
                if len(buffer) >= chunk_size:
                    temp_df = pd.DataFrame(buffer)
                    
                    # [Clean Optimization] 명시적 타입 변환 (함수 대신 직접 지정)
                    temp_df['t_dat'] = pd.to_datetime(temp_df['t_dat'])
                    temp_df['article_id'] = temp_df['article_id'].astype(str)
                    temp_df['customer_id'] = temp_df['customer_id'].astype(str)
                    # 숫자형은 float32/int8로 즉시 변환하여 메모리 절약
                    temp_df['price'] = temp_df['price'].astype(np.float32)
                    temp_df['sales_channel_id'] = temp_df['sales_channel_id'].astype(np.int8)
                    
                    # 필터링
                    mask = (temp_df['t_dat'] >= TRAIN_START_DATE) & (temp_df['t_dat'] <= DATASET_MAX_DATE)
                    temp_df = temp_df.loc[mask]
                    
                    if not temp_df.empty:
                        chunk_list.append(temp_df)
                    buffer = []

            # 남은 버퍼 처리
            if buffer:
                temp_df = pd.DataFrame(buffer)
                temp_df['t_dat'] = pd.to_datetime(temp_df['t_dat'])
                temp_df['article_id'] = temp_df['article_id'].astype(str)
                temp_df['customer_id'] = temp_df['customer_id'].astype(str)
                temp_df['price'] = temp_df['price'].astype(np.float32)
                temp_df['sales_channel_id'] = temp_df['sales_channel_id'].astype(np.int8)
                
                mask = (temp_df['t_dat'] >= TRAIN_START_DATE) & (temp_df['t_dat'] <= DATASET_MAX_DATE)
                temp_df = temp_df.loc[mask]
                
                if not temp_df.empty:
                    chunk_list.append(temp_df)

        if not chunk_list:
             raise ValueError("데이터 로드 실패: JSON 파일 내용이나 날짜 범위를 확인하세요.")
             
        print("Merging chunks...")
        df = pd.concat(chunk_list, ignore_index=True)
        del chunk_list, buffer
        gc.collect()
        
        print("Sorting...")
        df = df.sort_values(by=['customer_id', 't_dat']).reset_index(drop=True)
            
        print(f"Saving cache to {CACHE_FILE_PATH}...")
        df.to_parquet(CACHE_FILE_PATH, index=False)

    train_df = df[df['t_dat'] < VALID_START_DATE].copy()
    print(f" -> Loaded: {len(df)} rows, Train: {len(train_df)} rows")
    return df, train_df

# ==========================================
# 2. Item Features (Cleaned)
# ==========================================
def make_item_features(train_df):
    print("\n📦 [Item Stats] Calculating...")

    # A. Raw Probability
    total_tx = len(train_df)
    item_counts = train_df['article_id'].value_counts()
    # float32 명시
    item_feats = pd.DataFrame({'raw_probability': (item_counts / total_tx).astype(np.float32)})
    item_feats.index.name = 'article_id'

    # B. Pivot Sales
    train_df['week_start'] = train_df['t_dat'] - pd.to_timedelta(train_df['t_dat'].dt.dayofweek, unit='D')
    weekly_sales = train_df.groupby(['article_id', 'week_start']).size().unstack(fill_value=0).sort_index(axis=1)
    
    # Archive
    print("   💾 Archiving Weekly History...")
    weekly_save = weekly_sales.copy()
    weekly_save.columns = weekly_save.columns.astype(str)
    weekly_save.reset_index().to_parquet(WEEKLY_HISTORY_PATH)
    del weekly_save; gc.collect()

    # C. Popularity & Velocity (Vectorized)
    last_4w = weekly_sales.iloc[:, -4:]
    prev_4w = weekly_sales.iloc[:, -8:-4]
    
    # Log1p 적용 (float32 변환 불필요, pivot 결과가 이미 숫자임)
    item_feats['pop_1w_log'] = np.log1p(weekly_sales.iloc[:, -1].astype(np.float32))
    item_feats['pop_1m_log'] = np.log1p(last_4w.sum(axis=1).astype(np.float32))

    # Velocity
    s_curr_w = weekly_sales.iloc[:, -1]
    s_prev_w = weekly_sales.iloc[:, -2] if weekly_sales.shape[1] > 1 else 0
    item_feats['velocity_1w'] = ((s_curr_w - s_prev_w) / (s_prev_w + 1)).clip(-1, 5).astype(np.float32)

    s_curr_m = last_4w.sum(axis=1)
    s_prev_m = prev_4w.sum(axis=1) if len(prev_4w) > 0 else 0
    item_feats['velocity_1m'] = ((s_curr_m - s_prev_m) / (s_prev_m + 1)).clip(-1, 5).astype(np.float32)

    # Steady Score
    recent_12w = weekly_sales.iloc[:, -12:]
    mean_12w = recent_12w.mean(axis=1)
    std_12w = recent_12w.std(axis=1)
    item_feats['steady_score_log'] = np.log1p(mean_12w / (std_12w + 1e-9)).astype(np.float32)

    del weekly_sales; gc.collect()

    # Price
    item_feats['avg_item_price_log'] = np.log1p(train_df.groupby('article_id')['price'].mean().astype(np.float32))

    # D. Cold-Start Imputation
    first_sale = train_df.groupby('article_id')['t_dat'].min()
    max_date = train_df['t_dat'].max()
    
    # Days Calculation
    days_since = (max_date - first_sale).dt.days
    item_feats['days_since_release_log'] = np.log1p(days_since).astype(np.float32)
    
    # Imputation
    is_new = days_since < 14
    cols_to_impute = ['pop_1w_log', 'pop_1m_log', 'velocity_1w', 'velocity_1m']
    
    for col in cols_to_impute:
        if col in item_feats.columns:
            avg_val = item_feats[col].mean()
            item_feats.loc[is_new, col] = avg_val

    # E. Save
    item_feats = item_feats.reset_index()
    final_cols = ['article_id', 'raw_probability'] + cols_to_impute + ['steady_score_log', 'avg_item_price_log', 'days_since_release_log']
    final_df = item_feats[final_cols].fillna(0)
    
    save_dataframe(final_df, ITEM_FEAT_PATH_PQ, ITEM_FEAT_PATH_JS)
    return final_df
'''
# ==========================================
# 3. User Features (Cleaned)
# ==========================================
def make_user_features(train_df):
    print("\n👤 [User Stats] Calculating...")
    user_stats = train_df.groupby('customer_id').agg({
        'price': ['mean', 'count'],
        'article_id': 'nunique',
        't_dat': 'max',
        'sales_channel_id': 'mean'
    })
    user_stats.columns = ['user_avg_price', 'total_cnt', 'unique_item_cnt', 'last_purchase_date', 'channel_avg']
    user_stats = user_stats.reset_index()

    # 파생 변수 (명시적 float32 변환으로 에러 방지)
    user_stats['user_avg_price_log'] = np.log1p(user_stats['user_avg_price'].astype(np.float32))
    user_stats['total_cnt_log'] = np.log1p(user_stats['total_cnt'].astype(np.float32))
    user_stats['repurchase_ratio'] = (1 - (user_stats['unique_item_cnt'] / user_stats['total_cnt'])).astype(np.float32)
    
    max_date = train_df['t_dat'].max()
    days_diff = (max_date - user_stats['last_purchase_date']).dt.days
    user_stats['recency_log'] = np.log1p(days_diff.astype(np.float32))
    
    user_stats['preferred_channel'] = np.where(user_stats['channel_avg'] > 1.5, 2, 1).astype(np.int8)
    
    final_cols = ['customer_id', 'user_avg_price_log', 'total_cnt_log', 'repurchase_ratio', 'recency_log', 'preferred_channel']
    final_df = user_stats[final_cols].fillna(0)
    
    save_dataframe(final_df, USER_FEAT_PATH_PQ, USER_FEAT_PATH_JS)
    return final_df

'''
import os
import gc
import numpy as np
import pandas as pd

def make_user_features(train_df, target_val_path):
    print("\n👤 [User Stats] Calculating Enhanced Features (with Bucketing & Scaling)...")




    print("\n🎯 [Validation User Stats] Preparing point-in-time features...")

    # 1. 평가 대상 유저 ID 추출 (정답지가 있는 유저들)
    target_val = pd.read_parquet(target_val_path)
    val_user_set = set(target_val['customer_id'].unique())
    print(f" -> Found {len(val_user_set):,} target users for validation.")

    # 2. 9/15 이전 거래 중 '평가 대상 유저'의 기록만 추출
    # (이미 full_df가 9/15 이전 데이터라면 날짜 필터는 생략 가능)
    val_train_df = train_df[train_df['customer_id'].isin(val_user_set)].copy()
    print(f" -> Using {len(val_train_df):,} transaction records for feature calculation.")



    # ==========================================
    # 1. Basic Interaction Stats
    # ==========================================
    train_df['day_of_week'] = train_df['t_dat'].dt.dayofweek
    train_df['is_weekend'] = (train_df['day_of_week'] >= 5).astype(np.int8)
    train_df['month_id'] = train_df['t_dat'].dt.to_period('M')

    user_agg = train_df.groupby('customer_id').agg({
        'price': ['mean', 'std', 'last'],  
        'article_id': ['count', 'nunique'], 
        't_dat': 'max',                     
        'sales_channel_id': 'mean',         
        'is_weekend': 'mean',               
        'month_id': 'nunique'               
    })
    
    user_agg.columns = [
        'user_avg_price', 'price_std', 'last_price',
        'total_cnt', 'unique_item_cnt',
        'last_purchase_date', 'channel_avg', 
        'weekend_ratio', 'active_months'
    ]
    user_agg = user_agg.reset_index()

    # ==========================================
    # 2. Derived Features & Bucketing/Scaling
    # ==========================================
    print("   ⚙️ Generating Derived Features & Bucketing...")
    
    # (1) 결측치 및 파생 변수 처리
    user_agg['price_std'] = user_agg['price_std'].fillna(0).astype(np.float32)
    user_agg['last_price_diff'] = (user_agg['last_price'] - user_agg['user_avg_price']).astype(np.float32)
    user_agg['repurchase_ratio'] = (1.0 - (user_agg['unique_item_cnt'] / user_agg['total_cnt'])).astype(np.float32)
    
    max_date = train_df['t_dat'].max()
    user_agg['recency_days'] = (max_date - user_agg['last_purchase_date']).dt.days.astype(np.float32)
    
    user_agg['preferred_channel'] = np.where(user_agg['channel_avg'] > 1.5, 2, 1).astype(np.int8)
    user_agg['active_months'] = user_agg['active_months'].astype(np.int16)

    # (2) Bucketing (Quantile 기반 10구간 분할 -> Categorical ID로 변환)
    # 중복값이 많을 수 있으므로 duplicates='drop' 적용
    user_agg['user_avg_price_bucket'] = pd.qcut(user_agg['user_avg_price'], q=10, labels=False, duplicates='drop').astype(np.int8) + 1
    user_agg['total_cnt_bucket'] = pd.qcut(user_agg['total_cnt'], q=10, labels=False, duplicates='drop').astype(np.int8) + 1
    user_agg['recency_bucket'] = pd.qcut(user_agg['recency_days'], q=10, labels=False, duplicates='drop').astype(np.int8) + 1

    # (3) Continuous Features Scaling (Standardization: 평균 0, 표준편차 1)
    # 모델의 Continuous MLP에 들어갈 변수들
    cont_cols = ['price_std', 'last_price_diff', 'repurchase_ratio', 'weekend_ratio']
    for col in cont_cols:
        col_mean = user_agg[col].mean()
        col_std = user_agg[col].std() + 1e-9 # 0으로 나누기 방지
        user_agg[f'{col}_scaled'] = ((user_agg[col] - col_mean) / col_std).astype(np.float32)

    # ==========================================
    # 3. Customer Metadata Integration (고객 정보 병합)
    # ==========================================
    print("   👥 Loading & Merging Customer Metadata...")
    customers_df = pd.read_csv(USER_META_PATH)
    
    if 'postal_code' in customers_df.columns:
        customers_df = customers_df.drop(columns=['postal_code'])
        
    age_median = customers_df['age'].median()
    customers_df['age'] = customers_df['age'].fillna(age_median)
    # Age Bucketing (10구간)
    customers_df['age_bucket'] = pd.qcut(customers_df['age'], q=10, labels=False, duplicates='drop').astype(np.int8) + 1
    
    customers_df['FN'] = customers_df['FN'].fillna(0).astype(np.int8)
    customers_df['Active'] = customers_df['Active'].fillna(0).astype(np.int8)
    
    customers_df['club_member_status'] = customers_df['club_member_status'].fillna('OTHER').astype(str).str.upper()
    status_map = {'ACTIVE': 1, 'PRE-CREATE': 2}
    customers_df['club_member_status_idx'] = customers_df['club_member_status'].map(status_map).fillna(0).astype(np.int8)
    
    customers_df['fashion_news_frequency'] = customers_df['fashion_news_frequency'].fillna('NONE').astype(str).str.upper()
    news_map = {'REGULARLY': 1}
    customers_df['fashion_news_frequency_idx'] = customers_df['fashion_news_frequency'].map(news_map).fillna(0).astype(np.int8)

    meta_cols = ['customer_id', 'age_bucket', 'FN', 'Active', 'club_member_status_idx', 'fashion_news_frequency_idx']
    customers_meta = customers_df[meta_cols]

    final_df = pd.merge(user_agg, customers_meta, on='customer_id', how='left')

    # 병합 후 결측치 방어
    final_df['age_bucket'] = final_df['age_bucket'].fillna(0).astype(np.int8)
    for col in ['FN', 'Active', 'club_member_status_idx', 'fashion_news_frequency_idx']:
        final_df[col] = final_df[col].fillna(0).astype(np.int8)

    # ==========================================
    # 4. Final Selection & Save
    # ==========================================
    # 저장할 최종 컬럼 (Bucket IDs 4개, Scaled Cont 4개, Categorical IDs 5개)
    final_cols = [
        'customer_id', 
        'user_avg_price_bucket', 'total_cnt_bucket', 'recency_bucket', 'age_bucket', # Bucket IDs
        'price_std_scaled', 'last_price_diff_scaled', 'repurchase_ratio_scaled', 'weekend_ratio_scaled', # Scaled Cont
        'preferred_channel', 'active_months', 'FN', 'Active', 'club_member_status_idx', 'fashion_news_frequency_idx' # Categoricals
    ]
    
    final_df = final_df[final_cols].fillna(0)
    print("\n🔍 [Check] Generated User Features (Top 5):")
    print(final_df.head(5).T.to_string())
    save_dataframe(final_df, USER_FEAT_VAL_PATH_PQ, USER_FEAT_VAL_PATH_JS)
    del user_agg, customers_df, customers_meta; gc.collect()
    print("   ✅ User features successfully calculated and saved!")
    
    return final_df
# ==========================================
# 4. Sequences (Cleaned)
# ==========================================
def process_sequence_row(group):
    import pandas as pd # ★ 여기에 import 추가 (Windows 멀티프로세싱 필수)
    
    # 정렬 (sort_values는 얕은 복사이므로 메모리 효율적)
    group = group.sort_values('days_int')
    
    # Values만 추출하여 Numpy Array로 처리 (빠름)
    article_ids = group['article_id'].values
    days_ints = group['days_int'].values
    
    if len(article_ids) > 50:
        article_ids = article_ids[-50:]
        days_ints = days_ints[-50:]
        
    # Vectorized subtraction
    time_deltas = days_ints[-1] - days_ints
    
    # Parquet 저장을 위해 list 변환
    return pd.Series({
        'sequence_ids': list(article_ids),
        'sequence_deltas': list(time_deltas)
    })
def make_cleaned_sequences(full_df, processor, save_path):
    """
    정제된 full_df를 바탕으로 0번 노이즈가 없는 시퀀스 파일을 생성합니다.
    """
    print("\n" + "="*50)
    print("🧹 [Step 1] Filtering Transactions with Valid Items Only...")
    
    # 🌟 [핵심] FeatureProcessor에 등록된 7만 개 아이템만 남깁니다.
    valid_item_set = set(processor.item_ids)
    initial_rows = len(full_df)
    
    # 리스트에 없는 아이템 거래를 여기서 삭제 (0번 원천 봉쇄)
    full_df = full_df[full_df['article_id'].isin(valid_item_set)].copy()
    
    print(f" -> Removed {initial_rows - len(full_df):,} noise records.")
    print(f" -> Remaining Records: {len(full_df):,}")

    print("\n🔗 [Step 2] Building Sequences with Parallel Processing...")
    
    # 날짜 정수 변환
    full_df['days_int'] = ((full_df['t_dat'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')).astype(np.int32)
    
    # 필요한 컬럼만 추출하여 메모리 절약
    mini_df = full_df[['customer_id', 'article_id', 'days_int']].copy()
    del full_df
    gc.collect()
    
    # 병렬 처리 적용
    grouped = mini_df.groupby('customer_id')
    seq_df = grouped.parallel_apply(process_sequence_row)
    
    # 결과 저장
    seq_df = seq_df.reset_index()
    seq_df.to_parquet(save_path, index=False)
    
    print(f" ✅ [Success] Cleaned sequence file saved to: {save_path}")
    print("="*50 + "\n")
    return seq_df
class FeatureProcessor:
    def __init__(self, user_path, item_path, seq_path):
        self.users = pd.read_parquet(user_path).set_index('customer_id')
        self.items = pd.read_parquet(item_path).set_index('article_id')
        self.seqs = pd.read_parquet(seq_path).set_index('customer_id')
        self.user_ids = self.users.index.tolist()
        self.user2id = {uid: i + 1 for i, uid in enumerate(self.user_ids)}
        self.item_ids = self.items.index.tolist()
        self.item2id = {iid: i + 1 for i, iid in enumerate(self.item_ids)}
        self.user_scaler = StandardScaler()
        self.u_dense_cols = ['user_avg_price_log', 'total_cnt_log', 'recency_log']
        self.users_scaled = self.users.copy()
        self.users_scaled[self.u_dense_cols] = self.user_scaler.fit_transform(self.users[self.u_dense_cols])

    def get_user_tensor(self, user_id):
        dense = torch.tensor(self.users_scaled.loc[user_id, self.u_dense_cols].values, dtype=torch.float32)
        cat = torch.tensor(int(self.users_scaled.loc[user_id, 'preferred_channel']) - 1, dtype=torch.long)
        return dense, cat

    def get_logq_probs(self, device):
        sorted_probs = self.items['raw_probability'].reindex(self.item_ids).fillna(0).values
        return torch.tensor(sorted_probs, dtype=torch.float32).to(device)




def deep_inspect_missing_items(full_df, processor):
    print("\n🔍 [Deep Inspection] Identifying the source of 107k Zeros...")
    
    # 1. FeatureProcessor에 등록된 유효 아이템 ID 셋
    valid_items = set(processor.item_ids)
    
    # 2. 거래 데이터(full_df)에서 등록되지 않은 아이템 찾기
    is_invalid = ~full_df['article_id'].isin(valid_items)
    invalid_transactions = full_df[is_invalid]
    
    missing_count = len(invalid_transactions)
    unique_missing_items = invalid_transactions['article_id'].nunique()
    
    print(f" - Total Transactions with Missing Items: {missing_count:,}건")
    print(f" - Unique Missing Item IDs: {unique_missing_items:,}종류")
    
    if missing_count > 0:
        print("\n📊 [Top 10 Missing Items] 이 아이템들이 0번의 주범입니다:")
        print(invalid_transactions['article_id'].value_counts().head(10))
        
        # 3. 조치 제안
        print("\n💡 [Recommendation]")
        print(f" - 이 {missing_count}건의 데이터는 학습 시 target_id=0을 만듭니다.")
        print(f" - Recall 향상을 위해 full_df에서 위 아이템들을 제거(drop)하고 학습하세요.")
    else:
        print("✅ All items in full_df are correctly mapped to Processor!")
        
        
        
import pandas as pd
import numpy as np
import os

def make_validation_user_features(full_df, target_val_path, save_path):
    """
    full_df: 9/15 이전까지의 모든 거래 기록 (정제된 것)
    target_val_path: features_target_val.parquet 경로
    save_path: 저장할 경로 (USER_VAL_FEAT_PATH)
    """
    print("\n🎯 [Validation User Stats] Preparing point-in-time features...")

    # 1. 평가 대상 유저 ID 추출 (정답지가 있는 유저들)
    target_val = pd.read_parquet(target_val_path)
    val_user_set = set(target_val['customer_id'].unique())
    print(f" -> Found {len(val_user_set):,} target users for validation.")

    # 2. 9/15 이전 거래 중 '평가 대상 유저'의 기록만 추출
    # (이미 full_df가 9/15 이전 데이터라면 날짜 필터는 생략 가능)
    val_train_df = full_df[full_df['customer_id'].isin(val_user_set)].copy()
    print(f" -> Using {len(val_train_df):,} transaction records for feature calculation.")

    # 3. 기존 로직 그대로 실행 (make_user_features의 내부 로직)
    print("👤 Calculating user stats for validation...")
    user_stats = val_train_df.groupby('customer_id').agg({
        'price': ['mean', 'count'],
        'article_id': 'nunique',
        't_dat': 'max',
        'sales_channel_id': 'mean'
    })
    user_stats.columns = ['user_avg_price', 'total_cnt', 'unique_item_cnt', 'last_purchase_date', 'channel_avg']
    user_stats = user_stats.reset_index()

    # 파생 변수 계산
    user_stats['user_avg_price_log'] = np.log1p(user_stats['user_avg_price'].astype(np.float32))
    user_stats['total_cnt_log'] = np.log1p(user_stats['total_cnt'].astype(np.float32))
    user_stats['repurchase_ratio'] = (1 - (user_stats['unique_item_cnt'] / user_stats['total_cnt'])).astype(np.float32)
    
    max_date = val_train_df['t_dat'].max()
    days_diff = (max_date - user_stats['last_purchase_date']).dt.days
    user_stats['recency_log'] = np.log1p(days_diff.astype(np.float32))
    
    user_stats['preferred_channel'] = np.where(user_stats['channel_avg'] > 1.5, 2, 1).astype(np.int8)
    
    final_cols = ['customer_id', 'user_avg_price_log', 'total_cnt_log', 'repurchase_ratio', 'recency_log', 'preferred_channel']
    final_df = user_stats[final_cols].fillna(0)
    
    # 4. 저장 (평가 전용 경로로 저장)
    final_df.to_parquet(save_path, index=False)
    print(f" ✨ [Success] Validation user features saved to: {save_path}")
    return final_df

        
def make_validation_sequences(full_df, target_val_path, save_path, processor):
    """
    full_df: 9/15 이전의 거래 기록 (train_df를 넣으시면 됩니다)
    target_val_path: features_target_val.parquet (정답지)
    save_path: features_sequence_val.parquet
    """
    print("\n🔗 [Validation Sequences] Creating point-in-time sequences for target users...")
    
    # 1. 7만 개 명단에 있는 아이템만 필터링 (0번 노이즈 원천 봉쇄)
    valid_item_set = set(processor.item_ids)
    initial_len = len(full_df)
    
    # 변수명을 full_df로 통일하거나, 깔끔하게 여기서부터 df로 정의합니다.
    df = full_df[full_df['article_id'].isin(valid_item_set)].copy() 
    
    deleted = initial_len - len(df)
    if deleted > 0:
        print(f"🧹 시퀀스 생성 전 {deleted:,}건의 미등록 아이템을 제거했습니다.")

    # 2. 평가 대상 6.5만 명 유저 ID 추출
    target_val = pd.read_parquet(target_val_path)
    val_user_set = set(target_val['customer_id'].unique())
    print(f" -> Found {len(val_user_set):,} target users for validation.")

    # 3. 정제된 데이터에서 '평가 대상 유저'의 기록만 추출
    val_train_df = df[df['customer_id'].isin(val_user_set)].copy()
    
    if val_train_df.empty:
        print("⚠️ Warning: 필터링 결과 데이터가 없습니다. ID 매핑을 확인하세요.")
        return None

    # 4. 시퀀스 생성 로직 실행
    print(f" -> Processing sequences for {len(val_user_set):,} users...")
    
    # 날짜 정수 변환
    val_train_df['days_int'] = ((val_train_df['t_dat'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')).astype(np.int32)
    
    grouped = val_train_df.groupby('customer_id')
    # 병렬 처리 적용 (process_sequence_row 호출)
    seq_df = grouped.parallel_apply(process_sequence_row) 
    
    seq_df = seq_df.reset_index()
    seq_df.to_parquet(save_path, index=False)
    
    print(f" ✨ [Success] Validation sequences saved to: {save_path}")
    return seq_df

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np




def check_sequence_distribution(train_seq_path, valid_seq_path):
    print("📈 [Data Audit] Comparing Sequence Length Distributions...")
    
    # 1. 데이터 로드
    train_seq = pd.read_parquet(train_seq_path)
    valid_seq = pd.read_parquet(valid_seq_path)
    
    # 2. 시퀀스 길이 계산
    train_lens = train_seq['sequence_ids'].apply(len)
    valid_lens = valid_seq['sequence_ids'].apply(len)
    
    # 3. 기초 통계량 비교 테이블 생성
    stats = pd.DataFrame({
        'Dataset': ['Train (All)', 'Valid (Target Users)'],
        'Count': [len(train_lens), len(valid_lens)],
        'Mean': [train_lens.mean(), valid_lens.mean()],
        'Median': [train_lens.median(), valid_lens.median()],
        'Std': [train_lens.std(), valid_lens.std()],
        'Min': [train_lens.min(), valid_lens.min()],
        'Max': [train_lens.max(), valid_lens.max()]
    })
    
    print("\n[Check 1] Descriptive Statistics:")
    print(stats.to_string(index=False))

    # 4. 시각화 (Distribution Plot)
    plt.figure(figsize=(12, 5))
    
    sns.histplot(train_lens, color='skyblue', label='Train', kde=True, stat="probability", bins=30)
    sns.histplot(valid_lens, color='orange', label='Valid', kde=True, stat="probability", bins=30)
    
    plt.title('Sequence Length Distribution Comparison')
    plt.xlabel('Sequence Length (Number of Items)')
    plt.ylabel('Density (Probability)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    # 5. 분석 의견 출력
    diff = abs(train_lens.mean() - valid_lens.mean())
    if diff < 5:
        print(f"\n✅ SUCCESS: 분포가 매우 유사합니다. (평균 차이: {diff:.2f})")
    else:
        print(f"\n⚠️ WARNING: 분포 차이가 큽니다. (평균 차이: {diff:.2f})")
        print(" -> Valid 유저들이 상대적으로 헤비 유저이거나 라이트 유저일 수 있습니다.")

# 실행
# check_sequence_distribution(SEQ_DATA_PATH_PQ, SEQ_VAL_DATA_PATH)
 
import pandas as pd
import numpy as np

def final_sanity_check(seq_val_path, target_val_path):
    print("🔍 [Final Guardrail] Verifying Validation Data Integrity...")
    
    # 1. 데이터 로드
    seq_val = pd.read_parquet(seq_val_path)
    target_val = pd.read_parquet(target_val_path)
    
    # 2. 유저 수 일치 확인
    target_users = set(target_val['customer_id'].unique())
    seq_users = set(seq_val['customer_id'].unique())
    
    missing_in_seq = target_users - seq_users
    
    print(f"\n[Check 1] User Count Consistency")
    print(f" - Target Ground Truth Users: {len(target_users):,}명")
    print(f" - Sequence Data Users: {len(seq_users):,}명")
    
    if len(missing_in_seq) == 0:
        print(" ✅ SUCCESS: 모든 타겟 유저의 시퀀스가 존재합니다.")
    else:
        print(f" ⚠️ WARNING: {len(missing_in_seq):,}명의 유저 시퀀스가 누락되었습니다.")
        print(" (사유: 해당 유저들이 9/15 이전에 구매한 기록이 전혀 없을 수 있습니다.)")

    # 3. 0번(Unknown Item) 포함 여부 확인
    # sequence_ids 리스트 안에 0이 하나라도 있는지 전수조사
    contains_zero = seq_val['sequence_ids'].apply(lambda x: 0 in x).sum()
    
    print(f"\n[Check 2] Zero-ID (Noise) Check")
    if contains_zero == 0:
        print(" ✅ SUCCESS: 시퀀스 내에 '0'번(알 수 없는 아이템)이 전혀 없습니다.")
    else:
        print(f" ❌ ERROR: {contains_zero:,}개의 행에서 '0'번이 발견되었습니다! 필터링 로직을 재점검하세요.")

    # 4. 시퀀스 길이 적절성 확인
    avg_len = seq_val['sequence_ids'].apply(len).mean()
    print(f"\n[Check 3] Sequence Quality")
    print(f" - Average Sequence Length: {avg_len:.2f}")
    if avg_len >= 15:
        print(" ✅ SUCCESS: 풍부한 맥락 정보가 확보되었습니다.")
    else:
        print(" ℹ️ INFO: 평균 시퀀스가 다소 짧습니다.")

    print("\n" + "="*50)
    if len(missing_in_seq) == 0 and contains_zero == 0:
        print("🚀 ALL SYSTEMS GO! 이제 Phase 2.5 학습을 시작하셔도 좋습니다.")
    else:
        print("🛠️ 위 경고/에러 사항을 확인 후 진행 여부를 결정하세요.")
    print("="*50)

# 실행
# final_sanity_check(SEQ_VAL_DATA_PATH, TARGET_VAL_PATH)   

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_visualize_user_stats(parquet_path):
    """
    저장된 user_features parquet 파일을 로드하여 
    기초 통계량 출력 및 분포 시각화를 수행합니다.
    """
    print(f"📂 Loading data from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # 1. 기초 통계량 (최대, 최소, 평균, 4분위수) 출력
    print("\n" + "="*50)
    print("📊 [1. Summary Statistics]")
    print("="*50)
    # 가독성을 위해 소수점 4자리까지 표시
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    stats_df = df.describe().T[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    print(stats_df)
    
    # 분석할 컬럼 정의
    # customer_id 제외, preferred_channel은 범주형으로 취급
    numeric_cols = ['user_avg_price_log', 'total_cnt_log', 'repurchase_ratio', 'recency_log']
    cat_col = 'preferred_channel'
    
    # 시각화 스타일 설정
    sns.set(style="whitegrid")
    
    # 2. 수치형 변수 분포 시각화 (Histogram + Boxplot)
    print("\n" + "="*50)
    print("📈 [2. Numeric Features Distribution]")
    print("="*50)
    
    fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(16, 5 * len(numeric_cols)))
    
    for i, col in enumerate(numeric_cols):
        # (1) 히스토그램 & KDE (분포 모양 확인)
        sns.histplot(df[col], kde=True, ax=axes[i, 0], color='skyblue', bins=50)
        axes[i, 0].set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
        axes[i, 0].set_xlabel('')
        
        # 평균과 중앙값 선 표시
        axes[i, 0].axvline(df[col].mean(), color='red', linestyle='--', label='Mean')
        axes[i, 0].axvline(df[col].median(), color='green', linestyle='-', label='Median')
        axes[i, 0].legend()
        
        # (2) 박스플롯 (이상치 확인)
        sns.boxplot(x=df[col], ax=axes[i, 1], color='lightgreen')
        axes[i, 1].set_title(f'Boxplot of {col}', fontsize=14, fontweight='bold')
        axes[i, 1].set_xlabel('')
        
    plt.tight_layout()
    plt.show()
    
    # 3. 범주형 변수 분포 (Channel)
    print("\n" + "="*50)
    print("📊 [3. Categorical Feature Balance]")
    print("="*50)
    
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x=cat_col, data=df, palette='viridis')
    plt.title(f'Count of {cat_col} (1: Offline/Mixed, 2: Online)', fontsize=14, fontweight='bold')
    
    # 바 위에 카운트 숫자 표시
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    plt.show()
    
    # 4. 상관관계 히트맵 (변수 간 다중공선성 체크)
    print("\n" + "="*50)
    print("🔥 [4. Correlation Heatmap]")
    print("="*50)
    
    plt.figure(figsize=(10, 8))
    # 수치형 변수 + 채널만 포함
    corr_matrix = df[numeric_cols + [cat_col]].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # 상단 삼각형 가리기
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.show()

# ==========================================
# 실행 예시
# ==========================================
# 저장했던 경로를 그대로 넣어주세요
# USER_VAL_FEAT_PATH = "D:/trainDataset/localprops/features_user_val.parquet" (예시)
# analyze_and_visualize_user_stats(USER_VAL_FEAT_PATH)
# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    full_df, train_df = load_data()
    gc.collect()
    
    
    # 2. 설정된 경로 및 날짜 사용
    VALID_START_DATE = pd.to_datetime("2020-09-16")
    DATASET_MAX_DATE = pd.to_datetime("2020-09-22")
    TARGET_VAL_PATH = os.path.join(BASE_DIR, "features_target_val.parquet")


    #processor = FeatureProcessor(USER_FEAT_PATH_PQ, ITEM_FEAT_PATH_PQ, SEQ_DATA_PATH_PQ)

    #CLEANED_SEQ_PATH = os.path.join(BASE_DIR, "features_sequence_cleaned.parquet")
    #new_seq_df = make_cleaned_sequences(full_df, processor, CLEANED_SEQ_PATH)
    

    #USER_VAL_FEAT_PATH = os.path.join(BASE_DIR, "features_user_val.parquet")
    #val_user_features = make_validation_user_features(full_df, TARGET_VAL_PATH, USER_VAL_FEAT_PATH)

    #SEQ_VAL_DATA_PATH = os.path.join(BASE_DIR, "features_sequence_val.parquet")
    #make_validation_sequences(full_df, TARGET_VAL_PATH, SEQ_VAL_DATA_PATH)
    train_only_df = full_df[full_df['t_dat'] < VALID_START_DATE].copy()
    
    
    
    TARGET_VAL_PATH = os.path.join(BASE_DIR, "features_target_val.parquet")
    
    make_user_features(train_only_df,TARGET_VAL_PATH)
    
    
    # 2. Validation용 정답지 먼저 생성 (이건 full_df가 필요함)
    TARGET_VAL_PATH = os.path.join(BASE_DIR, "features_target_val.parquet")
    #make_validation_target_file(full_df, VALID_START_DATE, DATASET_MAX_DATE, TARGET_VAL_PATH)

    # 3. 🌟 [핵심 수정] Validation용 피처 생성 시 반드시 'train_df'를 넣으세요!
    #USER_VAL_FEAT_PATH = os.path.join(BASE_DIR, "features_user_val.parquet")
    # full_df 대신 train_df를 넣어야 9/16 이후 데이터가 침범하지 않습니다.
    #val_user_features = make_validation_user_features(train_df, TARGET_VAL_PATH, USER_VAL_FEAT_PATH)

    # 4. 🌟 [핵심 수정] Validation용 시퀀스 생성 시에도 'train_df'를 넣으세요!
    #SEQ_VAL_DATA_PATH = os.path.join(BASE_DIR, "features_sequence_val.parquet")
    # full_df 대신 train_df를 넣어야 시퀀스 내에 '미래의 정답'이 포함되지 않습니다.
    #make_validation_sequences(train_df, TARGET_VAL_PATH, SEQ_VAL_DATA_PATH,processor)
    
    
    #final_sanity_check(SEQ_VAL_DATA_PATH, TARGET_VAL_PATH)   


    USER_VAL_FEAT_PATH = "D:/trainDataset/localprops/features_user_val.parquet"
    #analyze_and_visualize_user_stats(USER_FEAT_PATH_PQ)


    #check_sequence_distribution(SEQ_DATA_PATH_PQ, SEQ_VAL_DATA_PATH)
    '''
    # 2. Item Stats
    make_item_features(train_df)
    del train_df; gc.collect()
    
    # 3. User Stats (Train Only)
    train_only_df = full_df[full_df['t_dat'] < VALID_START_DATE].copy()
    make_user_features(train_only_df)
    del train_only_df; gc.collect()
    
    # 4. Sequences (Train Only)
    train_seq_df = full_df[full_df['t_dat'] < VALID_START_DATE].copy()
    del full_df; gc.collect() # full_df 삭제하여 메모리 최대로 확보
    
    make_sequences(train_seq_df)
    
    
    
    
    
    
    
    
    # 3. 함수 호출
    make_validation_target_file(
        full_df=full_df, 
        valid_start_date=VALID_START_DATE, 
        max_date=DATASET_MAX_DATE, 
        save_path=TARGET_VAL_PATH
    )

    '''
