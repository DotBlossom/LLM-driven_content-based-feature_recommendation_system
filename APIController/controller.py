from fastapi import BackgroundTasks, FastAPI, Depends, HTTPException, APIRouter
from pydantic import BaseModel
from sqlalchemy import JSON, create_engine, Column, Integer,String, select
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase, mapped_column
from sqlalchemy.dialects.postgresql import insert as pg_insert
from pgvector.sqlalchemy import Vector
from database import get_db, Base, Vectors,TripletCreate,PreCreate,TripletSearch, ProductFeature, ProductFeatureCreate
from database import DBDataLoader, EmbeddingOutput, BatchVectorInput, BatchVectorOutput,BatchProductInput, ProductInput, FeatuerImp
from model import train_model, load_and_infer, train_simcse_from_db
from typing import List, Dict, Any, Optional

import os
import torch
import numpy as np

controller_router = APIRouter()



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



@controller_router.get("/invoke/")
def run_db_based_training():
    print(">>> [System] Initializing Stage 2 Training Pipeline...")

    # 1. DB 연결 및 데이터 로드 
    loader = DBDataLoader()
    
    # 결과: [{'vector': [...], 'clothes': {'category': ['top/tee']}}, ...]
    product_list = loader.fetch_training_data()
    
    if not product_list:
        print(">>> [System] No training data found. Aborting.")
        return

    # 2. Dataset/Dataloader 및 학습 실행
    print(f">>> [System] Starting Training with {len(product_list)} samples...")
    
    save_path = "models/stage2_triplet_optimized.pth"
    
    history = train_model(
        product_list=product_list,
        epochs=10, 
        batch_size=64, # (e.g., 256, 1024)
        save_path=save_path
    )
    
    print(f">>> [System] Training Complete. Model saved at {save_path}")


    # run_inference_and_update(loader, save_path) 

if __name__ == "__main__":
    run_db_based_training()
    
    

# --- [6] API Implementation ---



# [DB 테이블 1] Raw Feature 데이터 저장용
class RawItem(Base):
    __tablename__ = "raw_items"
    id = Column(Integer, primary_key=True, index=True)
    features = Column(JSON)  # N개의 feature를 JSON 형태로 저장 (예: {"category": "A", "price": 100})

# [DB 테이블 2] 학습된 임베딩 저장용
class ItemEmbedding(Base):
    __tablename__ = "item_embeddings"
    item_id = Column(Integer, primary_key=True) # RawItem의 ID와 매핑
    vector = Column(JSON) # 리스트 형태의 벡터 저장


# Pydantic 모델 (요청 바디 검증용)
class FeatureInput(BaseModel):
    features: Dict[str, Any] # 예: {"title": "...", "desc": "..."}

@controller_router.post("/data/upload")
async def upload_raw_features(inputs: List[FeatureInput], db: Session = Depends(get_db)):
    """
    [API 1] N개의 Raw Feature 데이터를 받아 DB에 저장합니다.
    """
    new_items = [RawItem(features=item.features) for item in inputs]
    db.add_all(new_items)
    db.commit()
    
    return {"message": f"Successfully saved {len(new_items)} items.", "status": "success"}

# --- API 2. 학습 요청 (Background Task) ---
@controller_router.post("/train/start")
async def start_training(background_tasks: BackgroundTasks):
    """
    [API 2] DB에 있는 데이터로 SimCSE 학습을 시작합니다. (비동기 실행)
    """
    # 백그라운드에서 실행되도록 넘김 (API는 즉시 응답)
    background_tasks.add_task(train_simcse_from_db)
    
    return {"message": "Training started in the background.", "status": "processing"}
