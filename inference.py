import torch
import torch.nn.functional as F
from sqlalchemy.orm import Session
from sqlalchemy import text # SQL 직접 실행 (pgvector 연산용)
from typing import List, Dict, Tuple

# 기존 모듈 임포트
from model import SymmetricUserTower
from database import UserProfile, UserInteraction
from database import ProductInferenceVectors # 아이템 벡터 테이블
from utils.util import load_pretrained_vectors_from_db
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
user_model_path = os.path.join("models", "user_tower_latest.pth")

class RecommendationService:
    def __init__(self, db_session: Session, model_path: str = user_model_path):
        """
        [초기화] 서버 시작 시 1회 실행
        1. 아이템 벡터 로딩 (Lookup Table)
        2. User Tower 모델 로딩 및 가중치 복원
        """
        print("🚀 [Inference Service] Initializing...")
        
        # 1. DB에서 아이템 벡터 매트릭스 & ID 맵 로딩 (학습 때와 동일)
        self.pretrained_matrix, self.product_id_map = load_pretrained_vectors_from_db(db_session)
        self.num_total_products = len(self.product_id_map)
        
        # 2. 모델 아키텍처 생성
        self.model = SymmetricUserTower(
            num_total_products=self.num_total_products,
            max_seq_len=50,
            input_dim=128
        )
        
        # 3. Lookup Table 주입 (Freeze)
        self.model.load_pretrained_weights(self.pretrained_matrix, freeze=True)
        
        # 4. 학습된 가중치(pth) 로드
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            print(f"✅ Loaded model weights from {model_path}")
        except FileNotFoundError:
            print("⚠️ Warning: Model file not found. Using initialized weights (Random).")
        
        self.model.to(DEVICE)
        self.model.eval() # 추론 모드 (Dropout Off, LayerNorm 통계 고정)
        print("✅ Recommendation Service Ready.")

    def _prepare_user_input(self, user: UserProfile) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        [전처리] DB 유저 객체를 모델 입력 텐서로 변환
        """
        # 1. Interaction -> Model Index Sequence
        # 시간순 정렬
        interactions = sorted(user.interactions, key=lambda x: x.timestamp)
        # Real ID -> Model Index 변환 (없으면 0)
        history_indices = [self.product_id_map.get(i.product_id, 0) for i in interactions]
        
        # 최근 50개만 유지
        if len(history_indices) > 50:
            history_indices = history_indices[-50:]
        # 없으면 0으로 채우기 (패딩) - 모델이 알아서 처리하지만 길이 1 이상은 필요
        if not history_indices:
            history_indices = [0]
            
        history_tensor = torch.tensor([history_indices], dtype=torch.long).to(DEVICE) # (1, Seq_Len)
        
        # 2. Profile -> Tensor
        profile_data = {
            'gender': torch.tensor([user.gender], dtype=torch.long).to(DEVICE),
            'age': torch.tensor([user.age_level], dtype=torch.long).to(DEVICE)
        }
        
        return history_tensor, profile_data

    def get_user_vector(self, db: Session, user_id: int) -> List[float]:
        """
        [추론] user_id -> 128차원 벡터 생성
        """
        # 1. 유저 정보 조회
        user = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not user:
            raise ValueError(f"User {user_id} not found.")
            
        # 2. 입력 데이터 준비
        history_tensor, profile_data = self._prepare_user_input(user)
        
        # 3. 모델 추론 (No Grad)
        with torch.no_grad():
            # (1, 128)
            user_vector_tensor = self.model(history_tensor, profile_data)
            
        # 4. 리스트로 변환 (DB 쿼리용)
        return user_vector_tensor.squeeze().cpu().tolist()

    def retrieve_similar_items(self, db: Session, user_vector: List[float], top_k: int = 10):
        """
        [검색] PGVector를 사용하여 유저 벡터와 가장 가까운 상품 검색
        """
        # pgvector 연산자 (<->: L2 Distance, <=>: Cosine Distance, <#>: Inner Product)
        # 추천 시스템에서는 보통 Inner Product(<#>)나 Cosine Distance(<=>)를 사용합니다.
        # 여기서는 Cosine Distance 사용 (값이 작을수록 유사함)
        
        # SQLAlchemy로 Vector 연산 쿼리 작성
        # 주의: ProductInferenceVectors 테이블에 vector_serving 컬럼이 pgvector 타입이어야 함
        
        results = db.query(
            ProductInferenceVectors.id,
            ProductInferenceVectors.category,
            ProductInferenceVectors.vector_embedding.cosine_distance(user_vector).label("distance")
        ).filter(
            ProductInferenceVectors.vector_embedding.isnot(None)
        ).order_by(
            "distance" # 거리 오름차순 (가까운 순)
        ).limit(top_k).all()
        
        print(f"[RETRIEVAL DEBUG] Found {len(results)} candidates from DB.")
        
        return results