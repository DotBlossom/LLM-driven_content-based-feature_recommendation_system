from typing import Dict, List, Tuple
from requests import Session
import torch
from database import ProductInferenceVectors, UserProfile


def load_pretrained_vectors_from_db(db_session: Session) -> Tuple[torch.Tensor, Dict[int, int]]:
    """
    [Stage 0] 데이터 준비
    DB의 ProductInferenceVectors 테이블에서 (ID, Vector)를 로드하여
    모델 초기화용 Matrix와 ID Mapping을 생성합니다.
    """
    print("⏳ [DB Loader] Fetching product vectors from DB...")
    
    # 1. DB Query: ID와 Serving용 벡터(128d)만 가져옴
    results = db_session.query(
        ProductInferenceVectors.id, 
        ProductInferenceVectors.vector_embedding
    ).filter(
        ProductInferenceVectors.vector_embedding.isnot(None)
    ).all()
    
    if not results:
        raise ValueError("❌ DB에 저장된 아이템 벡터가 없습니다! Item Tower 추론을 먼저 수행하세요.")

    # 2. 메타데이터 설정
    num_products = len(results)
    vector_dim = 128  # Item Tower Output Dimension
    
    # 0번 인덱스는 Padding을 위해 비워둠 (Index 1부터 시작)
    # Shape: (전체상품수 + 1, 128)
    embedding_matrix = torch.zeros((num_products + 1, vector_dim), dtype=torch.float32)
    
    id_map = {} # Real DB ID -> Model Index (0, 1, 2...)
    
    # 3. 매트릭스 채우기
    print(f"📦 [DB Loader] Processing {num_products} items...")
    
    for idx, (real_id, vector_list) in enumerate(results, start=1):
        # vector_list가 문자열이나 리스트로 올 수 있으므로 변환 처리 필요할 수 있음
        # 여기서는 List[float]라고 가정
        
        # ID 매핑 (DB ID 1050 -> Model Index 1)
        id_map[real_id] = idx 
        
        # 텐서 할당
        embedding_matrix[idx] = torch.tensor(vector_list, dtype=torch.float32)
        
    print(f"✅ [DB Loader] Matrix Created. Shape: {embedding_matrix.shape}")
    
    return embedding_matrix, id_map

# train_service.py 내부에 추가하거나 utils로 분리

















'''
def fetch_training_data_from_db(db: Session, min_interactions: int = 2):
    """
    [Data Extractor]
    DB의 UserInteraction을 조회하여 -> 학습용 {history, target, profile} 리스트로 변환
    Sliding Window 방식으로 데이터를 증강합니다.
    """
    print("📊 [Data Fetcher] Loading user logs from DB...")
    
    # UserProfile과 그들의 Interactions를 한 번에 로딩 (Eager Loading 권장)
    # 여기서는 간단히 Query 수행
    users = db.query(UserProfile).all()
    
    training_samples = []
    
    for user in users:
        # 이력이 없는 유저는 스킵
        if not user.interactions:
            continue
            
        # 시간순 정렬 (과거 -> 최신)
        # DB 모델에 relationship이 'interactions'로 잡혀있다고 가정
        sorted_interactions = sorted(user.interactions, key=lambda x: x.timestamp)
        
        # 상품 ID 시퀀스 추출
        product_seq = [i.product_id for i in sorted_interactions]
        
        # 최소 길이 체크 (History 1개 + Target 1개 = 2개 이상이어야 학습 가능)
        if len(product_seq) < min_interactions:
            continue
            
        # --- [Sliding Window Logic] ---
        # 예: [A, B, C] -> ([A], B), ([A,B], C) 두 개의 샘플 생성
        for i in range(1, len(product_seq)):
            history_part = product_seq[:i]  # 입력: 과거 이력
            target_item = product_seq[i]    # 정답: 다음 아이템
            
            # 너무 긴 history는 모델 max_len에 맞춰 잘라주는 게 좋음 (Dataset에서도 하지만 여기서 미리 처리)
            if len(history_part) > 50:
                history_part = history_part[-50:]
            
            training_samples.append({
                "history": history_part,      # List[int]
                "target": target_item,        # int
                "gender": user.gender,        # int
                "age": user.age_level         # int
            })
            
    print(f"✅ Generated {len(training_samples)} real training samples from DB.")
    return training_samples

## example usage: context vector support
## context -> ranker context feature vector input

import torch
import math
import numpy as np

class ContextFeatureEngineer:
    def __init__(self, output_dim=20):
        self.output_dim = output_dim
        
    def _encode_cyclical_time(self, value, max_val):
        """
        [핵심 기법 1] 시간의 연속성 보존 (Cyclical Encoding)
        23시와 0시는 숫자로는 멀지만(23 차이), 실제로는 1시간 차이입니다.
        이를 Sin/Cos 좌표로 변환하여 원형 시계처럼 표현합니다.
        """
        sin_val = math.sin(2 * math.pi * value / max_val)
        cos_val = math.cos(2 * math.pi * value / max_val)
        return [sin_val, cos_val]

    def _log_scale(self, value):
        """
        [핵심 기법 2] 값의 스케일 압축 (Log Transformation)
        조회수 같은 데이터는 롱테일(Long-tail) 분포를 가집니다. (0 ~ 100만)
        로그를 취해 격차를 줄여야 모델이 학습하기 좋습니다.
        log(x + 1) : 0일 때 에러 방지
        """
        return math.log1p(max(0, value))

    def _one_hot(self, value, num_classes):
        """범주형 데이터 변환"""
        vec = [0] * num_classes
        if 0 <= value < num_classes:
            vec[value] = 1
        return vec

    def process(self, raw_context: dict) -> torch.Tensor:
        """
        raw_context = {
            'hour': 14,             # 0-23
            'weekday': 0,           # 0(Mon)-6(Sun)
            'view_count_1h': 150,   # 최근 1시간 조회수
            'item_ctr': 0.05,       # 최근 CTR
            'last_visit_min': 30,   # 마지막 접속 후 흐른 시간(분)
            'device_type': 0        # 0:Mobile, 1:PC, 2:Tablet
        }
        """
        features = []
        
        # 1. Time (Hour) -> 2 dims
        features.extend(self._encode_cyclical_time(raw_context.get('hour', 0), 24))
        
        # 2. Weekday -> 7 dims
        features.extend(self._one_hot(raw_context.get('weekday', 0), 7))
        
        # 3. Real-time Stats -> 2 dims
        # 조회수는 로그 스케일링
        features.append(self._log_scale(raw_context.get('view_count_1h', 0)))
        # CTR은 이미 0~1이므로 그대로 (혹은 스케일링)
        features.append(raw_context.get('item_ctr', 0.0))
        
        # 4. User Freshness -> 1 dim
        # 10분 전 접속과 1000분 전 접속의 차이를 로그로 표현
        features.append(self._log_scale(raw_context.get('last_visit_min', 0)))
        
        # 5. Device -> 3 dims
        features.extend(self._one_hot(raw_context.get('device_type', 0), 3))
        
        # 현재까지 차원 수 계산: 2 + 7 + 1 + 1 + 1 + 3 = 15차원
        
        # 6. Padding (남은 5차원 0으로 채우기)
        current_dim = len(features)
        if current_dim < self.output_dim:
            features.extend([0.0] * (self.output_dim - current_dim))
        
        return torch.tensor(features, dtype=torch.float32)

# --- 사용 예시 ---
engineer = ContextFeatureEngineer(output_dim=20)

# 현재 상황 (오후 2시, 월요일, 모바일 접속, 인기있는 상품)
current_ctx = {
    'hour': 14,
    'weekday': 0,
    'view_count_1h': 1205, # 조회수 높음
    'item_ctr': 0.12,
    'last_visit_min': 5,   # 방금 전 접속
    'device_type': 0       # 모바일
}

context_tensor = engineer.process(current_ctx)

print(f"Context Vector Shape: {context_tensor.shape}")
print(f"Context Vector Data: {context_tensor}")

'''