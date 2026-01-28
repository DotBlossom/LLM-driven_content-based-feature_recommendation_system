import pandas as pd
import json
import os
import numpy as np
from datetime import timedelta

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
BASE_DIR = r"D:\trainDataset\localprops"
FILE_PATH = os.path.join(BASE_DIR, "transactions_train_filtered.json")

# 날짜 기준 설정 (마지막 1주일 Test, 그 전 1주일 Valid)
TEST_DAYS = 7
VALID_DAYS = 7

def save_json(data, filename):
    path = os.path.join(BASE_DIR, filename)
    print(f"Saving {len(data)} entries to {filename}...")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def run_gts_split():
    print(f"Reading data from {FILE_PATH}...")
    # 데이터 로드
    df = pd.read_json(FILE_PATH)
    
    # 날짜 변환 및 정렬 (필수)
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    df = df.sort_values(by=['customer_id', 't_dat']).reset_index(drop=True)
    
    # Article ID 문자열 변환 (Leading Zero 보존)
    df['article_id'] = df['article_id'].astype(str).str.zfill(10)
    
    # 전체 기간 확인
    max_date = df['t_dat'].max()
    min_date = df['t_dat'].min()
    print(f"Data Range: {min_date.date()} ~ {max_date.date()}")
    
    # ---------------------------------------------------------
    # 2. Global Time Split Point 설정
    # ---------------------------------------------------------
    # Test 시작일 (마지막 7일)
    test_start_date = max_date - timedelta(days=TEST_DAYS)
    
    # Valid 시작일 (Test 시작 7일 전)
    valid_start_date = test_start_date - timedelta(days=VALID_DAYS)
    
    print(f"\n[Split Configuration]")
    print(f" - Train Period : ~ {valid_start_date.date()}")
    print(f" - Valid Period : {valid_start_date.date()} ~ {test_start_date.date()}")
    print(f" - Test Period  : {test_start_date.date()} ~ {max_date.date()}")
    
    # ---------------------------------------------------------
    # 3. 데이터 그룹화 (User별 시퀀스 생성)
    # ---------------------------------------------------------
    print("\nGrouping data by user...")
    # 속도를 위해 numpy array 변환
    # (user_id, article_id, timestamp)
    data_values = df[['customer_id', 'article_id', 't_dat']].values
    
    # 유저별 데이터를 담을 딕셔너리
    # key: user_id, value: list of (article_id, timestamp)
    user_history = {}
    
    for uid, aid, t in data_values:
        if uid not in user_history:
            user_history[uid] = []
        user_history[uid].append((aid, t))
        
    print(f"Total Users: {len(user_history)}")

    # ---------------------------------------------------------
    # 4. GTS Logic 적용 (Target: Last)
    # ---------------------------------------------------------
    print("Applying GTS Split (Target='Last')...")
    
    train_seqs = {} # 순수 학습용 (Valid/Test 기간 제외)
    valid_data = {} # 검증용 (Input + Target)
    test_data = {}  # 테스트용 (Input + Target)
    
    for user, items in user_history.items():
        # 시간순 정렬되어 있다고 가정 (위에서 sort_values 함)
        
        # 1. 구간별 아이템 분류
        train_items = []
        valid_candidates = []
        test_candidates = []
        
        for aid, t in items:
            if t <= valid_start_date:
                train_items.append(aid)
            elif t <= test_start_date:
                valid_candidates.append(aid)
            else:
                test_candidates.append(aid)
        
        # -----------------------------------------------------
        # A. Train Set 구성
        # -----------------------------------------------------
        # Train에는 Valid/Test 기간 데이터가 절대 들어가면 안됨
        if len(train_items) > 0:
            train_seqs[user] = train_items
            
        # -----------------------------------------------------
        # B. Valid Set 구성 (Target = Valid 기간의 Last Item)
        # -----------------------------------------------------
        if valid_candidates:
            # Target: Valid 기간의 마지막 아이템
            target = valid_candidates[-1]
            
            # Input Sequence: Train 전체 + Valid 기간 중 Target 이전 아이템들
            # (Last 타겟이므로 Valid 기간의 마지막 빼고 전부가 Input에 추가됨)
            input_seq = train_items + valid_candidates[:-1]
            
            # 최소 길이 필터링 (Input이 없으면 예측 불가하므로 제외 가능)
            if len(input_seq) > 0:
                valid_data[user] = {
                    "input": input_seq,
                    "target": target
                }

        # -----------------------------------------------------
        # C. Test Set 구성 (Target = Test 기간의 Last Item)
        # -----------------------------------------------------
        if test_candidates:
            # Target: Test 기간의 마지막 아이템
            target = test_candidates[-1]
            
            # Input Sequence: Train 전체 + Valid 전체 + Test 기간 중 Target 이전
            # ★중요: Test 시점에서는 Valid 기간 데이터도 '과거'이므로 Input으로 사용 가능 (Retraining 효과)
            input_seq = train_items + valid_candidates + test_candidates[:-1]
            
            if len(input_seq) > 0:
                test_data[user] = {
                    "input": input_seq,
                    "target": target
                }

    # ---------------------------------------------------------
    # 5. 결과 저장
    # ---------------------------------------------------------
    print("\nSaving datasets...")
    save_json(train_seqs, "gts_train_seq.json")
    save_json(valid_data, "gts_valid_dataset.json")
    save_json(test_data, "gts_test_dataset.json")
    
    # 통계 출력
    print("\n[Summary]")
    print(f"Train Sequence Users : {len(train_seqs)}")
    print(f"Validation Cases     : {len(valid_data)} (Target in Valid Period)")
    print(f"Test Cases           : {len(test_data)} (Target in Test Period)")
    print("Done!")

if __name__ == "__main__":
    run_gts_split()