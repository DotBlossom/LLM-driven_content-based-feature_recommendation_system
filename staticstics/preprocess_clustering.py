import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================
# [실제 사용 시]: df = pd.read_csv('transactions.csv')
# 테스트를 위해 더미 데이터를 조금 더 늘려서 생성합니다.
data = {
    't_dat': [
        '2018-09-20', '2018-09-20', '2018-09-20', # UserA: 평일, 한 카테고리 집중
        '2018-09-22', '2018-09-23',               # UserB: 주말, 비쌈, 다양함
        '2018-09-20', '2018-09-20',               # UserC: 평일, 쌈, 재구매
        '2018-09-29', '2018-09-29', '2018-09-29'  # UserD: 주말, 많이 삼
    ],
    'customer_id': [
        'UserA', 'UserA', 'UserA', 
        'UserB', 'UserB', 
        'UserC', 'UserC', 
        'UserD', 'UserD', 'UserD'
    ],
    'article_id': [
        '0663713001', '0663713001', '0663713002', # UserA (같은거 2번삼)
        '0541518023', '0999999999',               # UserB
        '0111111111', '0111111111',               # UserC (재구매)
        '0663713001', '0541518023', '0999999999'  # UserD
    ],
    'price': [
        0.05, 0.05, 0.04,  # UserA
        0.08, 0.09,        # UserB (비쌈)
        0.01, 0.01,        # UserC (쌈)
        0.03, 0.03, 0.02   # UserD
    ]
}
df = pd.DataFrame(data)

# 날짜 변환
df['t_dat'] = pd.to_datetime(df['t_dat'])

# 카테고리 코드 생성 (앞 3자리 가정)
df['category_code'] = df['article_id'].astype(str).str[:3]

print(">>> 데이터 로드 완료. Feature Engineering 시작...")

# ==========================================
# 2. Feature Engineering (지표 추출)
# ==========================================

# --- [Basic Features] ---

# 2-1. Basket Size (구매 규모)
f_basket = df.groupby('customer_id')['article_id'].count().reset_index()
f_basket.columns = ['customer_id', 'basket_size']

# 2-2. Price Sensitivity (절대적 가격 민감도)
f_price = df.groupby('customer_id')['price'].mean().reset_index()
f_price.columns = ['customer_id', 'avg_price']

# 2-3. Category Entropy (취향 다양성)
def calculate_entropy(x):
    counts = x.value_counts()
    return entropy(counts)

f_entropy = df.groupby('customer_id')['category_code'].apply(calculate_entropy).reset_index()
f_entropy.columns = ['customer_id', 'cat_entropy']

# 2-4. Long-tail Ratio (마이너 취향 비율)
item_counts = df['article_id'].value_counts()
top_20_percent_idx = int(len(item_counts) * 0.2)
if top_20_percent_idx == 0: top_20_percent_idx = 1 # 데이터가 적을 때 예외처리
head_items = set(item_counts.index[:top_20_percent_idx])

df['is_tail'] = df['article_id'].apply(lambda x: 1 if x not in head_items else 0)
f_longtail = df.groupby('customer_id')['is_tail'].mean().reset_index()
f_longtail.columns = ['customer_id', 'long_tail_ratio']

# --- [Advanced Features] ---

# 2-5. Time Pattern (주말 쇼핑 비율)
df['day_of_week'] = df['t_dat'].dt.dayofweek # 0:월 ~ 6:일
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

f_time = df.groupby('customer_id')['is_weekend'].mean().reset_index()
f_time.columns = ['customer_id', 'weekend_ratio']

# 2-6. Loyalty (재구매율)
def calc_repurchase_rate(x):
    counts = x.value_counts()
    repurchased_items = counts[counts > 1].sum() 
    total_items = counts.sum()
    return repurchased_items / total_items if total_items > 0 else 0

f_loyalty = df.groupby('customer_id')['article_id'].apply(calc_repurchase_rate).reset_index()
f_loyalty.columns = ['customer_id', 'repurchase_rate']

# 2-7. Relative Price Position (상대적 가격 포지션)
category_avg_prices = df.groupby('category_code')['price'].mean().to_dict()
df['cat_avg_price'] = df['category_code'].map(category_avg_prices)
df['relative_price_ratio'] = df['price'] / df['cat_avg_price']

f_rel_price = df.groupby('customer_id')['relative_price_ratio'].mean().reset_index()
f_rel_price.columns = ['customer_id', 'relative_price_pos']

# ==========================================
# 3. 데이터 병합 (User Profile 완성)
# ==========================================
user_features = f_basket.merge(f_price, on='customer_id') \
                        .merge(f_entropy, on='customer_id') \
                        .merge(f_longtail, on='customer_id') \
                        .merge(f_time, on='customer_id') \
                        .merge(f_loyalty, on='customer_id') \
                        .merge(f_rel_price, on='customer_id')

# ==========================================
# 4. Fine-grained Clustering (페르소나 세분화)
# ==========================================
scaler = StandardScaler()

# 모델링에 사용할 모든 변수 (7개)
cols_for_cluster = [
    'basket_size', 'avg_price', 'cat_entropy', 'long_tail_ratio', 
    'weekend_ratio', 'repurchase_rate', 'relative_price_pos'
]
X_scaled = scaler.fit_transform(user_features[cols_for_cluster])

# [설정] 실제 데이터에서는 K=10 ~ 20 추천. 
# 현재 더미 데이터 유저가 4명이므로 에러 방지를 위해 K=2로 설정함.
K_CLUSTERS = 2 
kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42)
user_features['cluster_id'] = kmeans.fit_predict(X_scaled)

print(f">>> 군집화 완료 (K={K_CLUSTERS}). 통계 추출 및 JSON 변환 시작...")

# ==========================================
# 5. 최종 통계 추출 및 자동 태깅 (JSON 생성)
# ==========================================
stats_output = []

# 전체 평균값 미리 계산 (비교용)
global_means = user_features[cols_for_cluster].mean()

for cluster_id in sorted(user_features['cluster_id'].unique()):
    sub_df = user_features[user_features['cluster_id'] == cluster_id]
    
    # --- 통계치 계산 ---
    stats = {}
    for col in cols_for_cluster:
        stats[col] = round(sub_df[col].mean(), 2)
    
    # Basket Size는 std도 추가
    basket_std = round(sub_df['basket_size'].std(), 2)
    if pd.isna(basket_std): basket_std = 0.0
    
    # --- Persona Auto-Tagging Logic (동적 이름 생성) ---
    desc = []
    
    # 1. Time
    if stats['weekend_ratio'] > 0.6: desc.append("Weekend_Shopper")
    elif stats['weekend_ratio'] < 0.3: desc.append("Weekday_Shopper")
    
    # 2. Loyalty
    if stats['repurchase_rate'] > 0.3: desc.append("Loyal_Rebuyer")
    
    # 3. Price & Spending
    if stats['relative_price_pos'] < 0.9: desc.append("Discount_Hunter")
    elif stats['relative_price_pos'] > 1.1: desc.append("Premium_Picker")
    
    # 4. Diversity
    if stats['cat_entropy'] < 0.5: desc.append("Specialist")
    elif stats['cat_entropy'] > 1.5: desc.append("Explorer")
    
    # 5. Long-tail
    if stats['long_tail_ratio'] > 0.6: desc.append("Hipster")
    
    # 태그 조합
    if not desc: desc.append("Standard_Shopper")
    persona_name_tag = " & ".join(desc)
    
    # --- JSON 구조 조립 ---
    stats_output.append({
        "cluster_id": int(cluster_id),
        "persona_name": f"Cluster {cluster_id}: {persona_name_tag}",
        "stats": {
            "basket_size": {"mean": stats['basket_size'], "std": basket_std},
            "category_entropy": stats['cat_entropy'],
            "price_sensitivity": {
                "avg_price": stats['avg_price'],
                "relative_pos": stats['relative_price_pos'], # 1.0보다 높으면 비싼거 선호
                "description": "High Spending" if stats['avg_price'] > global_means['avg_price'] else "Low Spending"
            },
            "shopping_pattern": {
                "weekend_ratio": stats['weekend_ratio'],     # 1에 가까우면 주말러
                "repurchase_rate": stats['repurchase_rate'], # 높으면 재구매 성향
                "long_tail_ratio": stats['long_tail_ratio']  # 높으면 마이너 취향
            }
        },
        "user_count": int(len(sub_df))
    })

# ==========================================
# 6. 결과 출력 및 저장
# ==========================================
json_result = json.dumps(stats_output, indent=2, ensure_ascii=False)
print(json_result)

# 파일 저장
# with open('final_persona_stats.json', 'w', encoding='utf-8') as f:
#     f.write(json_result)