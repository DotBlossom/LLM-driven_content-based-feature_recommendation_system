import polars as pl
import time
import os

import json

target_dir = r'C:\Users\candyform\Desktop\inferenceCode\localprops'
input_csv_file = os.path.join(target_dir, 'transactions_train.csv')

start_time = time.time()

def save_pretty_json(df: pl.DataFrame, filename: str):

    print(f"'{filename}' 저장 중... (변환 과정이 있어 조금 더 걸릴 수 있습니다)")
    
    # 1. Polars 데이터를 Python 리스트(Dictionary List)로 변환
    data_list = df.to_dicts()
    
    # 2. Python 표준 json 라이브러리로 저장
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(
            data_list, 
            f, 
            indent=4,            # 들여쓰기 4칸 (가독성 핵심)
            ensure_ascii=False   # 한글/특수문자 깨짐 방지
        )
    print(f"완료! '{filename}'")

# 1. Lazy Loading (scan_csv)
# 계획 only
# dtypes: ID는 문자열(Utf8)로 지정
q = pl.scan_csv(
    input_csv_file,
    dtypes={'article_id': pl.Utf8, 'customer_id': pl.Utf8}
)

# 2. 집계 로직 작성 (실제 연산은 아직 수행 안 됨)
# 기능 1: 고객별 구매 이력 집계
customer_agg = q.group_by("customer_id").agg([
    pl.col("article_id"),
    pl.col("t_dat")
])

# 기능 2: 상품별 판매량 집계
article_agg = q.group_by("article_id").len().rename({"len": "purchase_count"})

# 3. 실제 연산 수행 및 메모리에 로드 (collect)
print("집계 연산 수행 중...")
df_customer = customer_agg.collect() 
df_article = article_agg.collect()

print(f"처리 완료: {time.time() - start_time:.2f}초")

# 4. 저장
print("파일 저장 중...")
df_customer.write_json("customer_summary.json")
df_article.write_json("article_counts.json")
print("모든 작업 완료!")

save_pretty_json(df_customer.head(100), "customer_sample_view.json")
save_pretty_json(df_article, "article_counts_pretty.json")