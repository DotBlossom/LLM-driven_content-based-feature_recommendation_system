import pandas as pd
import numpy as np
import random
from tqdm import tqdm

class HMLogImporter:
    """
    H&M Transaction Logs(CSV)를 읽어서 CatBoost Ranker 학습 데이터로 변환
    전략: Positive(구매) 1개당 Negative(랜덤 비구매) K개를 생성
    """
    
    def __init__(self, csv_path: str, item_vector_store: dict):
        """
        csv_path: transactions_train.csv 경로
        item_vector_store: {article_id: [vector...]} 형태의 딕셔너리 (이미 로드되어 있어야 함)
        """
        self.csv_path = csv_path
        self.item_vector_store = item_vector_store
        
        # 전체 상품 ID 리스트 (Negative Sampling용)
        self.all_product_ids = list(item_vector_store.keys())

    def load_and_preprocess(self, limit=100000, negative_ratio=5):
        """
        Args:
            limit: CSV에서 읽을 최대 행 수 (메모리 보호)
            negative_ratio: 구매 1건당 생성할 가짜 비구매(Negative) 데이터 수
        
        Returns:
            X (Features), y (Labels), group_ids (Query IDs)
        """
        print(f"📂 Loading H&M Logs from {self.csv_path} (limit={limit})...")
        
        # 1. CSV 로드 (필요한 컬럼만)
        # H&M article_id는 '0108775015' 같은 문자열이므로 dtype=str 지정 중요
        df = pd.read_csv(self.csv_path, nrows=limit, usecols=['customer_id', 'article_id', 't_dat'], dtype={'article_id': str})
        
        # 2. 데이터 컨테이너
        user_vecs_list = []
        item_vecs_list = []
        labels_list = []
        groups_list = [] # CatBoost Group ID
        
        # H&M 유저에 대한 벡터는 우리에게 없으므로, '평균 유저 벡터' 또는 '0 벡터'를 사용해야 함
        # (전이 학습에서는 유저 취향보다 아이템 간의 관계를 학습하는 게 목표이므로 괜찮음)
        default_user_vec = np.zeros(128) 

        # 3. 유저 단위로 그룹핑 (빠른 처리를 위해)
        grouped = df.groupby('customer_id')['article_id'].apply(list)
        
        print("⚙️ Generating Negative Samples & Vectors...")
        
        # 각 유저별 처리
        # group_counter는 CatBoost가 인식할 정수형 Group ID
        group_counter = 0
        
        for customer_id, bought_items in tqdm(grouped.items()):
            # 유저가 구매한 상품들 (Positive)
            for prod_id in bought_items:
                if prod_id not in self.item_vector_store:
                    continue # 벡터 없는 상품은 스킵

                # [Positive Sample] Label = 1
                user_vecs_list.append(default_user_vec)
                item_vecs_list.append(self.item_vector_store[prod_id])
                labels_list.append(1.0)
                groups_list.append(group_counter)
                
                # [Negative Sampling] Label = 0
                # 구매하지 않은 상품을 랜덤으로 뽑음
                negatives = 0
                while negatives < negative_ratio:
                    random_pid = random.choice(self.all_product_ids)
                    
                    # 우연히 산 걸 뽑았으면 다시 뽑기
                    if random_pid in bought_items: 
                        continue
                        
                    user_vecs_list.append(default_user_vec)
                    item_vecs_list.append(self.item_vector_store[random_pid])
                    labels_list.append(0.0)
                    groups_list.append(group_counter)
                    
                    negatives += 1
            
            # 다음 유저로 넘어감
            group_counter += 1

        # 4. Numpy 변환
        print("🔄 Converting to Numpy Arrays...")
        X_user = np.array(user_vecs_list, dtype=np.float32)
        X_item = np.array(item_vecs_list, dtype=np.float32)
        y = np.array(labels_list, dtype=np.float32)
        groups = np.array(groups_list, dtype=np.int32)
        
        print(f"✅ Data Prepared: {len(y)} samples (Pos:Neg = 1:{negative_ratio})")
        
        return X_user, X_item, y, groups