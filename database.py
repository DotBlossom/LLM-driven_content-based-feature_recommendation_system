import os
from dotenv import load_dotenv
from sqlalchemy import Boolean, create_engine,select, Column, Integer , String
from typing import List, Dict, Any, Optional
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel, Field

from sqlalchemy.orm import sessionmaker, mapped_column
from sqlalchemy.orm import DeclarativeBase # 이 줄을 추가하여 임포트합니다.

'''
/stage1/batch-embed: Stage 1 모델을 사용하여 Raw Data -> DB(vector_pre) 파이프라인 수행.

/stage2/batch-inference: Stage 2 모델을 사용하여 vector_pre -> vector_triplet 변환 수행. (DB 저장 없이 결과만 리턴하여 서빙 로직에서 즉시 사용 가능하도록 설계)


'''

# 1. .env 파일 로드 (환경변수로 등록됨)
load_dotenv()

# 2. 환경변수에서 가져오기
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

# DB 세션 생성 함수 (Dependency)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:

        db.close()



class Vectors(Base):
    __tablename__ = "vectors"

    id = Column(Integer, primary_key=True, index=True)
    vector_embedding = mapped_column(Vector(128), nullable=True)
    vector_serving = mapped_column(Vector(128), nullable=True)
    category = Column(String, nullable=True, index=True)
    # category 추가 필요

class ProductInferenceVectors(Base):
    __tablename__ = "product_inference_vectors"

    id = Column(Integer, primary_key=True, index=True)
    vector_embedding = mapped_column(Vector(128), nullable=True)
    vector_serving = mapped_column(Vector(128), nullable=True)
    category = Column(String, nullable=True, index=True)
    # category 추가 필요



## real data / valid data

class ProductServiceVectors(Base):
    __tablename__ = "product_service_vectors"

    id = Column(Integer, primary_key=True, index=True)
    vector_embedding = mapped_column(Vector(128), nullable=True)
    category = Column(String, nullable=True, index=True)


class ProductServiceInput(Base):
    __tablename__ = "product_service_input"

    # product_id INTEGER PRIMARY KEY
    product_id = Column(Integer, primary_key=True)
    
    # feature_data JSONB
    # 이 컬럼 안에 "clothes"와 "reinforced_feature_value"가 들어있음
    feature_data = Column(JSONB)
    is_vectorized = Column(Boolean, default=False, index=True)








class ProductInput(Base):
    __tablename__ = "product_input"

    # product_id INTEGER PRIMARY KEY
    product_id = Column(Integer, primary_key=True)
    
    # feature_data JSONB
    # 이 컬럼 안에 "clothes"와 "reinforced_feature_value"가 들어있음
    feature_data = Column(JSONB)
    
    #추론해야할 실제 데이터임을 구분하는 bool 추가필요
    
# dataform


class ProductInferenceInput(Base):
    __tablename__ = "product_inference_input"

    # product_id INTEGER PRIMARY KEY
    product_id = Column(Integer, primary_key=True)
    
    # feature_data JSONB
    # 이 컬럼 안에 "clothes"와 "reinforced_feature_value"가 들어있음
    feature_data = Column(JSONB)
    is_vectorized = Column(Boolean, default=False, index=True)

    #아이템벡터로 변환된 상품인지 체크하는 flag 필요 스키마
# dataform




class ProductInputSchema(BaseModel):
    product_id: int
    feature_data: Dict[str, Any] # 혹은 json 구조에 맞는 dict


"""
# --- [Triplet: 128차원 전용] ---
class TripletCreate(BaseModel):
    id: int
    vector: List[float] = Field(..., min_items=128, max_items=128, description="128차원 벡터")

class TripletSearch(BaseModel):
    vector: List[float] = Field(..., min_items=128, max_items=128)
    top_k: int = 5

# --- [Pre] ---
class PreCreate(BaseModel):
    id: int
    vector: List[float] = Field(..., min_items=512, max_items=512, description="512차원 벡터")

# JSON 데이터 입력용 스키마
class ProductFeatureCreate(BaseModel):
    product_id: int
    feature_data: Dict[str, Any] # 자유로운 JSON 포맷



# 1-1. Stage 1 입력용 (Raw Product Data)
class ClothesInfo(BaseModel):
    category: List[str]
    # 기타 속성들...
"""

class BatchProductInput(BaseModel):
    products: List[ProductInputSchema]

class EmbeddingOutput(BaseModel):
    processed_count: int
    failed_ids: List[int]



# 1-2. Stage 2 입력용 (Pre-computed Vectors)
class VectorItem(BaseModel):
    id: Optional[int] = None # 서빙용일 경우 ID가 없을 수도 있음
    vector: List[float] = Field(..., min_items=128, max_items=128)

class BatchVectorInput(BaseModel):
    items: List[VectorItem]

class BatchVectorOutput(BaseModel):
    results: List[Dict[str, Any]] # [{'id': 1, 'vector': [...]}, ...]





from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, func, Index
from sqlalchemy.orm import relationship, mapped_column
from database import Base # 사용자의 database.py에서 Base를 가져옴

class UserProfile(Base):
    """
    [유저 프로필 테이블]
    User Tower의 [User Token] 생성을 위한 정적 메타데이터 저장
    """
    __tablename__ = "user_profiles"

    user_id = Column(Integer, primary_key=True, index=True) # 실제 서비스의 User ID
    
    # 메타데이터 (User Tower 입력용)
    # 예: gender (0:Unk, 1:M, 2:F), age (0:Unk, 1:10대, 2:20대...)
    gender = Column(Integer, default=0) 
    age_level = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 관계 설정 (1:N)
    interactions = relationship("UserInteraction", back_populates="user")


class UserInteraction(Base):
    """
    [유저 행동 로그 테이블]
    User Tower의 [History Sequence] 생성을 위한 시계열 데이터
    """
    __tablename__ = "user_interactions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    user_id = Column(Integer, ForeignKey("user_profiles.user_id"), index=True)
    
    # 상품 ID (ProductInferenceVectors 테이블의 ID와 매핑됨)
    product_id = Column(Integer, nullable=False, index=True)
    
    # 행동 타입 (purchase, view, cart 등 - 나중에 가중치 줄 때 사용 가능)
    interaction_type = Column(String, default="view") 
    
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    # 관계 설정
    user = relationship("UserProfile", back_populates="interactions")

    # 복합 인덱스 (쿼리 성능 최적화: 특정 유저의 이력을 시간순 조회)
    __table_args__ = (
        Index('idx_user_timestamp', "user_id", "timestamp"),
    )



from pydantic import BaseModel, Field
from typing import List, Optional

class InteractionItem(BaseModel):
    product_id: int
    interaction_type: str = "view" # 기본값

class UserDataInput(BaseModel):
    """
    API 요청 바디: 유저 1명의 정보와 구매 이력 리스트
    """
    user_id: int
    gender: int = Field(..., description="0:Unk, 1:M, 2:F")
    age_level: int = Field(..., description="0:Unk, 1:10s, 2:20s...")
    
    # 한 번에 여러 상품 이력을 올릴 수 있도록 리스트로 받음
    history: List[InteractionItem] 

class BatchUserUploadRequest(BaseModel):
    """
    N명의 유저 데이터를 한 번에 업로드
    """
    users: List[UserDataInput]




"""
class DBDataLoader:
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def fetch_training_data(self) -> List[Dict[str, Any]]:

        session = self.SessionLocal()
        training_data = []
        
        try:
            print("[DB] Fetching vectors and categories (Single Table Scan)...")
            
            stmt = (
                select(Vectors.vector_pre, Vectors.category)
                .where(
                    Vectors.vector_pre.is_not(None),  # 벡터가 있고
                    Vectors.category.is_not(None)     # 카테고리도 있는 데이터만
                )
            )
            
            # DB 실행 (pgvector가 vector_pre를 자동으로 list/numpy로 변환해줌)
            results = session.execute(stmt).all()
            
            # RichAttributeDataset이 기대하는 포맷으로 변환
            # (기존 데이터셋 코드와의 호환성을 위해 딕셔너리 구조 유지)
            for vec_pre, cat_str in results:
                formatted_item = {
                    "vector": vec_pre, 
                    "clothes": {
                        # DB에 "top/t-shirt"로 저장되어 있다고 가정하고 리스트로 감쌈
                        "category": [cat_str] 
                    }
                }
                training_data.append(formatted_item)
                
            print(f"[DB] Successfully loaded {len(training_data)} items without JOIN.")
            return training_data
            
        except Exception as e:
            print(f"[Error] Failed to fetch data: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            session.close()

    def update_triplet_vectors(self, id_vector_map: Dict[int, List[float]]):

        session = self.SessionLocal()
        try:
            # 대량 업데이트 로직 (mappings 활용 권장)
            mappings = [
                {"id": p_id, "vector_triplet": vec} 
                for p_id, vec in id_vector_map.items()
            ]
            session.bulk_update_mappings(Vectors, mappings)
            session.commit()
            print(f"[DB] Successfully updated {len(mappings)} triplet vectors.")
        except Exception as e:
            session.rollback()
            print(f"[Error] Update failed: {e}")
        finally:
            session.close()
        
"""