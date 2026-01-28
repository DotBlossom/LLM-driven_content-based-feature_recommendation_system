import pandas as pd
import os
import json
from datetime import timedelta

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
BASE_DIR = r"D:\trainDataset\localprops"
RAW_FILE_PATH = os.path.join(BASE_DIR, "transactions_train_filtered.json")

# ★ 캐시 파일 경로 설정 (이 파일이 있으면 로딩 속도 10배 이상 빨라짐)
CACHE_FILE_PATH = os.path.join(BASE_DIR, "cached_transactions_1yr.parquet")

# H&M 데이터셋의 실제 마지막 날짜
DATASET_MAX_DATE = pd.Timestamp("2020-09-22")

# Local Test 기간 (7일)
TEST_DAYS = 7
TEST_START_DATE = DATASET_MAX_DATE - timedelta(days=TEST_DAYS - 1) # 2020-09-16

# 학습 데이터 사용 기간 (최근 1년)
TRAIN_START_DATE = DATASET_MAX_DATE - timedelta(days=365) # 2019-09-23

print(f"[Config] Train Range (Input): {TRAIN_START_DATE.date()} ~ {(TEST_START_DATE - timedelta(days=1)).date()}")
print(f"[Config] Test Range (Target): {TEST_START_DATE.date()} ~ {DATASET_MAX_DATE.date()}")

# ---------------------------------------------------------
# Step 1. 데이터 로드 (캐싱 로직 적용)
# ---------------------------------------------------------
if os.path.exists(CACHE_FILE_PATH):
    print(f"\n[Cache Hit] 캐시 파일이 발견되었습니다: {CACHE_FILE_PATH}")
    print("빠른 속도로 데이터를 로드합니다...")
    # Parquet 로드 (날짜 타입 등이 그대로 유지됨)
    df = pd.read_parquet(CACHE_FILE_PATH)
    print("Done!")
    
else:
    print(f"\n[Cache Miss] 캐시 파일이 없습니다. 원본 JSON을 로드하고 처리합니다...")
    print(f"Reading from {RAW_FILE_PATH}...")
    
    # 1. JSON 로드
    df = pd.read_json(RAW_FILE_PATH)
    
    # 2. 전처리 (시간이 오래 걸리는 작업들)
    print("Processing Dates & Types...")
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    df['article_id'] = df['article_id'].astype(str) # 문자열 변환
    
    # 3. 기간 필터링 (최근 1년)
    print(f"Filtering date range ({TRAIN_START_DATE.date()} ~ {DATASET_MAX_DATE.date()})...")
    df = df[(df['t_dat'] >= TRAIN_START_DATE) & (df['t_dat'] <= DATASET_MAX_DATE)]
    
    # 4. 정렬 (Sequential 모델 필수)
    print("Sorting data...")
    df = df.sort_values(by=['customer_id', 't_dat']).reset_index(drop=True)
    
    if len(df) == 0:
        raise ValueError("데이터가 0건입니다. 날짜 범위를 확인해주세요.")
        
    # 5. 캐시 저장 (다음 실행을 위해)
    print(f"Saving cache to {CACHE_FILE_PATH}...")
    df.to_parquet(CACHE_FILE_PATH, index=False)
    print("Cache Saved!")

print(f" -> Total Interactions Loaded: {len(df)}")

# ---------------------------------------------------------
# Step 2. Train / Local Test(Target) 분할
# ---------------------------------------------------------
print("\nSplitting Train vs Local Test...")

# 1) Train Data (~ 2020-09-15)
df_train = df[df['t_dat'] < TEST_START_DATE].copy()

# 2) Local Test Target Data (2020-09-16 ~ 2020-09-22)
df_test_target = df[df['t_dat'] >= TEST_START_DATE].copy()

print(f" -> Train Rows: {len(df_train)}")
print(f" -> Test Target Rows: {len(df_test_target)}")

# ---------------------------------------------------------
# Step 3. 시퀀스 생성 및 데이터셋 구축
# ---------------------------------------------------------
print("\nBuilding Sequences...")

# A. Train Sequence 생성 (유저별 과거 이력)
train_seq_dict = df_train.groupby('customer_id')['article_id'].apply(list).to_dict()

# B. Local Test Dataset 생성
local_test_dataset = {}

# Test 기간에 구매 기록이 있는 유저들을 그룹화
test_target_group = df_test_target.groupby('customer_id')['article_id'].apply(list)

count_no_history = 0

for user_id, target_items in test_target_group.items():
    if user_id in train_seq_dict:
        input_seq = train_seq_dict[user_id]
        
        # Test 데이터셋 구성
        local_test_dataset[user_id] = {
            "input": input_seq,      # 과거 기록 (Train)
            "target": target_items   # 미래 기록 (Test Target)
        }
    else:
        # 과거 기록이 없는 신규 유저 (이번 평가 제외)
        count_no_history += 1

print(f"\n[Result Summary]")
print(f"1. Train Sequences : {len(train_seq_dict)} users")
print(f"2. Local Test Cases: {len(local_test_dataset)} users")
print(f"   (Skipped {count_no_history} cold-start users)")

# ---------------------------------------------------------
# Step 4. 최종 결과 저장 (JSON)
# ---------------------------------------------------------
def save_json(data, filename):
    path = os.path.join(BASE_DIR, filename)
    print(f"Saving {filename}...")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

save_json(train_seq_dict, "final_train_seq.json")
save_json(local_test_dataset, "final_local_test_dataset.json")

print("\nAll Process Complete.")