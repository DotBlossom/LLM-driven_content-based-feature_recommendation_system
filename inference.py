'''
import numpy as np
import torch
import torch.nn.functional as F
from sqlalchemy.orm import Session
from sqlalchemy import text # SQL 직접 실행 (pgvector 연산용)
from typing import List, Dict, Tuple

# 기존 모듈 임포트
from model import SymmetricUserTower
from database import UserProfile, UserInteraction
from database import ProductInferenceVectors # 아이템 벡터 테이블
from ranker import GBDTRankingModel
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
        
        self.ranking_model = GBDTRankingModel(model_path="catboost_ranker.cbm")
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
    
    
    
    def get_item_vectors_by_ids(self, db: Session, product_ids: List[int]) -> Dict[int, List[float]]:
        """
        [Helper] 제품 ID 리스트를 받아 해당 제품들의 임베딩 벡터를 조회
        SQLAlchemy의 IN 절을 사용하여 배치 조회 (Batch Retrieval)
        """
        # 1. 빈 리스트 예외 처리 (불필요한 DB 호출 방지)
        if not product_ids:
            return {}

        # 2. 쿼리 작성 (필요한 컬럼만 선택하여 최적화)
        # SELECT id, vector_embedding FROM product_inference_vectors WHERE id IN (...)
        results = db.query(
            ProductInferenceVectors.id, 
            ProductInferenceVectors.vector_embedding
        ).filter(
            ProductInferenceVectors.id.in_(product_ids)
        ).all()

        # 3. 딕셔너리 매핑 (Ranking 로직에서 O(1) 조회를 위함)
        item_vectors = {}
        
        for pid, embedding in results:
            if embedding is not None:
                # pgvector는 설정에 따라 numpy array나 문자열로 반환될 수 있으므로
                # 확실하게 list[float] 형태로 변환하여 저장합니다.
                
                if hasattr(embedding, 'tolist'): # numpy array인 경우
                    vec_list = embedding.tolist()
                else: # 이미 리스트인 경우
                    vec_list = embedding
                
                item_vectors[pid] = vec_list

        return item_vectors
    
    
    def train_ranking_pipeline(self, db, params):
        """
        [GBDT Training Pipeline]
        로그 수집 -> Numpy 변환 -> CatBoost 학습
        """
        # 1. 로그 수집 
        logs = self.fetch_interaction_logs(db, params.log_limit)
        if not logs: return

        # 2. 데이터셋 구성 (List -> Numpy)
        user_vecs_list = []
        item_vecs_list = []
        labels_list = []
        groups_list = [] # 유저 ID (랭킹 그룹)

        # DB에서 벡터 미리 가져오기 (User/Item Store)
        # ... (기존 코드의 user_vector_store, item_vector_store 생성 로직 활용) ...
        user_vector_store = {}
        item_vector_store = {}
        for uid, pid, label, _ in logs:
            u_vec = user_vector_store.get(uid, np.zeros(128))
            i_vec = item_vector_store.get(pid, np.zeros(128))
            
            user_vecs_list.append(u_vec)
            item_vecs_list.append(i_vec)
            labels_list.append(label)
            groups_list.append(uid) # 중요: 유저 ID를 그룹 ID로 사용

        # 3. Numpy 변환
        X_user = np.array(user_vecs_list)
        X_item = np.array(item_vecs_list)
        y = np.array(labels_list)
        groups = np.array(groups_list)

        # 4. CatBoost 학습 호출
        self.ranking_model.train(X_user, X_item, y, groups)
        
        return {"status": "success", "model": "CatBoost"}
    
    
'''