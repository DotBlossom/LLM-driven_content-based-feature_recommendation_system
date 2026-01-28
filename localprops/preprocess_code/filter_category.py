import json
import pandas as pd
import os

# 1. 경로 설정
BASE_DIR = r"D:\trainDataset\localprops"
FILE_FILTERED_ID = os.path.join(BASE_DIR, "filtered_id.json")
FILE_TRANSACTIONS_CSV = os.path.join(BASE_DIR, "transactions_train.csv")
FILE_OUTPUT = os.path.join(BASE_DIR, "transactions_train_filtered.json")

def process_filtering():
    # --- Step 1: ID 파일 읽기 ---
    print(f"[Step 1] Loading {FILE_FILTERED_ID}...")
    with open(FILE_FILTERED_ID, 'r', encoding='utf-8') as f:
        article_ids = json.load(f)
    
    # [중요] ID 정규화: 무조건 문자열로 변환 후 10자리로 맞춤 (앞에 0 채우기)
    # 예: 123 -> "0000000123", "0101" -> "0000000101"
    valid_ids_set = set(str(aid).zfill(10) for aid in article_ids)
    
    # 디버깅용 출력
    sample_id = next(iter(valid_ids_set))
    print(f"   -> Loaded {len(valid_ids_set)} IDs.")
    print(f"   -> Sample ID from JSON (Normalized): '{sample_id}'")

    # --- Step 2: 트랜잭션 파일 읽기 ---
    print(f"[Step 2] Reading {FILE_TRANSACTIONS_CSV}...")
    
    # CSV 읽기 (article_id를 일단 문자열로 읽음)
    df = pd.read_csv(FILE_TRANSACTIONS_CSV, dtype={'article_id': str})
    
    print(f"   -> Original Count: {len(df)}")
    
    # [중요] CSV 데이터도 동일하게 정규화 (10자리 맞춤)
    # 데이터가 숫자로 읽혔든, 짧은 문자로 읽혔든 10자리로 강제 변환
    df['article_id'] = df['article_id'].astype(str).str.zfill(10)
    
    # 디버깅용 출력
    print(f"   -> Sample ID from CSV (Normalized): '{df['article_id'].iloc[0]}'")

    # --- Step 3: 필터링 ---
    print("[Step 3] Filtering data...")
    filtered_df = df[df['article_id'].isin(valid_ids_set)]
    
    print(f"   -> Filtered Count: {len(filtered_df)}")

    if len(filtered_df) == 0:
        print("WARNING: 여전히 매칭되는 데이터가 없습니다. 샘플 ID를 직접 비교해보세요.")
    else:
        # --- Step 4: 저장 ---
        print(f"[Step 4] Saving to {FILE_OUTPUT}...")
        filtered_df.to_json(FILE_OUTPUT, orient='records', indent=4)
        print("Done!")

if __name__ == "__main__":
    process_filtering()