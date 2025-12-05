import os
from dotenv import load_dotenv
from sqlalchemy import create_engine,select, Column, Integer , String
from sqlalchemy.orm import sessionmaker, DeclarativeBase, mapped_column
from typing import List, Dict, Any
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel, Field


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
    vector_triplet = mapped_column(Vector(128), nullable=True)
    vector_pre = mapped_column(Vector(512), nullable=True)
    category = Column(String, nullable=True, index=True)
    # category 추가 필요

class ProductFeature(Base):
    __tablename__ = "product_feature"

    # product_id INTEGER PRIMARY KEY
    product_id = Column(Integer, primary_key=True)
    
    # feature_data JSONB
    feature_data = Column(JSONB)


# --- [Triplet: 128차원 전용] ---
class TripletCreate(BaseModel):
    id: int
    vector: List[float] = Field(..., min_items=128, max_items=128, description="128차원 벡터")

class TripletSearch(BaseModel):
    vector: List[float] = Field(..., min_items=128, max_items=128)
    top_k: int = 5

# --- [Pre: 512차원 전용] ---
class PreCreate(BaseModel):
    id: int
    vector: List[float] = Field(..., min_items=512, max_items=512, description="512차원 벡터")

# JSON 데이터 입력용 스키마
class ProductFeatureCreate(BaseModel):
    product_id: int
    feature_data: Dict[str, Any] # 자유로운 JSON 포맷




class DBDataLoader:
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def fetch_training_data(self) -> List[Dict[str, Any]]:
        """
        Vectors 테이블 단일 조회로 (Vector 512d, Category) 쌍을 가져옵니다.
        """
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
        """
        학습된 결과(128d)를 DB에 업데이트
        """
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
        
