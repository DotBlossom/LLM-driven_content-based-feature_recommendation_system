from fastapi import FastAPI, Depends, HTTPException, APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, Integer
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase, mapped_column
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from database import get_db, Base


controller_router = APIRouter()


class Vectors(Base):
    __tablename__ = "vectors"

    id = Column(Integer, primary_key=True, index=True)
    vector_triplet = mapped_column(Vector(128), nullable=True)
    vector_pre = mapped_column(Vector(512), nullable=True)


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



@controller_router.post("db/triplet/")
def save_triplet(item: TripletCreate, db: Session = Depends(get_db)):

    # 1. 먼저 DB에 해당 ID가 있는지 확인
    existing_vector = db.query(Vectors).filter(Vectors.id == item.id).first()

    if existing_vector:
        # [CASE 1: 수정] 이미 데이터가 존재함
        # vector_pre는 건드리지 않고, vector_triplet만 교체합니다.
        existing_vector.vector_triplet = item.vector
        message = "Updated existing record (Triplet updated, Pre preserved)"
    else:
        # [CASE 2: 신규] 데이터가 없음 -> 새로 생성
        new_vector = Vectors(
            id=item.id,
            vector_triplet=item.vector,
            vector_pre=None  # 처음 만들어지니까 나머지는 비움
        )
        db.add(new_vector)
        message = "Created new record"

    db.commit()
    return {"id": item.id, "type": "triplet", "message": message}


@controller_router.post("db/pre/")
def save_pre(item: PreCreate, db: Session = Depends(get_db)):

    # 1. 먼저 DB에 해당 ID가 있는지 확인
    existing_vector = db.query(Vectors).filter(Vectors.id == item.id).first()

    if existing_vector:
        # [CASE 1: 수정] 이미 데이터가 존재함
        # vector_triplet은 건드리지 않고, vector_pre만 교체합니다.
        existing_vector.vector_pre = item.vector
        message = "Updated existing record (Pre updated, Triplet preserved)"
    else:
        # [CASE 2: 신규] 데이터가 없음 -> 새로 생성
        new_vector = Vectors(
            id=item.id,
            vector_triplet=None, # 처음 만들어지니까 나머지는 비움
            vector_pre=item.vector
        )
        db.add(new_vector)
        message = "Created new record"

    db.commit()
    return {"id": item.id, "type": "pre", "message": message}



@controller_router.post("db/triplet/search/")
def search_triplet(req: TripletSearch, db: Session = Depends(get_db)):

    # vector_triplet이 NULL이 아닌 데이터 중에서만 검색
    results = db.query(Vectors).filter(Vectors.vector_triplet.isnot(None))\
        .order_by(Vectors.vector_triplet.l2_distance(req.vector))\
        .limit(req.top_k).all()
    
    return results



class ProductFeature(Base):
    __tablename__ = "product_feature"

    # product_id INTEGER PRIMARY KEY
    product_id = Column(Integer, primary_key=True)
    
    # feature_data JSONB
    feature_data = Column(JSONB)

# JSON 데이터 입력용 스키마
class ProductFeatureCreate(BaseModel):
    product_id: int
    feature_data: Dict[str, Any] # 자유로운 JSON 포맷



@controller_router.post("/db/features/")
def create_feature(item: ProductFeatureCreate, db: Session = Depends(get_db)):

    # 이미 존재하는지 확인
    existing = db.query(ProductFeature).filter(ProductFeature.product_id == item.product_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Product ID already exists")

    db_feature = ProductFeature(
        product_id=item.product_id,
        feature_data=item.feature_data
    )
    db.add(db_feature)
    db.commit()
    db.refresh(db_feature)
    return db_feature

@controller_router.get("/db/features/{product_id}")
def get_feature(product_id: int, db: Session = Depends(get_db)):

    feature = db.query(ProductFeature).filter(ProductFeature.product_id == product_id).first()
    if not feature:
        raise HTTPException(status_code=404, detail="Product feature not found")
    return feature
