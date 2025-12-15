from typing import Dict, Tuple
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
