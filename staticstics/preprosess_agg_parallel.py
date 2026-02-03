import os
import pandas as pd
import numpy as np
import gc
import json
import ijson
from datetime import timedelta
from pandarallel import pandarallel

# ==========================================
# 0. Global Settings
# ==========================================
# 16GB RAM 기준: Worker 2~3개 추천 (안전하게 2)
WORKER_COUNT = 2
pandarallel.initialize(progress_bar=True, nb_workers=WORKER_COUNT, verbose=1)

BASE_DIR = r"D:\trainDataset\localprops"
RAW_FILE_PATH = os.path.join(BASE_DIR, "transactions_train_filtered.json")
CACHE_FILE_PATH = os.path.join(BASE_DIR, "cached_transactions_1yr.parquet")

# Output Paths
USER_FEAT_PATH_PQ = os.path.join(BASE_DIR, "features_user.parquet")
USER_FEAT_PATH_JS = os.path.join(BASE_DIR, "features_user.json")
ITEM_FEAT_PATH_PQ = os.path.join(BASE_DIR, "features_item.parquet")
ITEM_FEAT_PATH_JS = os.path.join(BASE_DIR, "features_item.json")
SEQ_DATA_PATH_PQ = os.path.join(BASE_DIR, "features_sequence.parquet")
SEQ_DATA_PATH_JS = os.path.join(BASE_DIR, "features_sequence.json")
WEEKLY_HISTORY_PATH = os.path.join(BASE_DIR, "history_weekly_sales.parquet")
MONTHLY_HISTORY_PATH = os.path.join(BASE_DIR, "history_monthly_sales.parquet")

# Date Config
TRAIN_START_DATE = pd.to_datetime("2019-09-23")
DATASET_MAX_DATE = pd.to_datetime("2020-09-22")
VALID_START_DATE = pd.to_datetime("2020-09-16") 

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

def make_sequences(df):
    print("\n🔗 [Sequences] Building with Parallel Processing...")
    
    # 날짜 정수 변환 (int32)
    df['days_int'] = ((df['t_dat'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')).astype(np.int32)
    
    # 필요한 컬럼만 복사
    mini_df = df[['customer_id', 'article_id', 'days_int']].copy()
    gc.collect()
    
    grouped = mini_df.groupby('customer_id')
    seq_df = grouped.parallel_apply(process_sequence_row)
    
    seq_df = seq_df.reset_index()
    save_dataframe(seq_df, SEQ_DATA_PATH_PQ, SEQ_DATA_PATH_JS)
    return seq_df

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # 1. Load Data
    full_df, train_df = load_data()
    gc.collect()
    
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
    
    print("\n✨ All Feature Engineering Completed Successfully!")