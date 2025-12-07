from fastapi import BackgroundTasks, FastAPI, Depends, HTTPException, APIRouter
from pydantic import BaseModel
from sqlalchemy import JSON, create_engine, Column, Integer,String, select
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase, mapped_column
from sqlalchemy.dialects.postgresql import insert as pg_insert
from pgvector.sqlalchemy import Vector
from database import ProductInferenceInput, get_db, Base
from database import  ProductInput
from train import  train_simcse_from_db
from typing import List, Dict, Any, Optional

import os
import torch
import numpy as np

controller_router = APIRouter()

"""

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
    
"""

# ---  API Implementation ---

class ProductCreateRequest(BaseModel):
    product_id: int
    feature_data: Dict[str, Any]  # JSONB 컬럼에 매핑될 딕셔너리




@controller_router.post("/products/ingest")
def ingest_products(payload: List[ProductCreateRequest], db: Session = Depends(get_db)):
    """
    [API 1] N개의 상품 데이터를 받아서 DB에 저장합니다.
    초기 저장 시 is_vectorized는 default로 False가 됩니다.
    """
    try:
        saved_count = 0
        for item in payload:
            # 중복 체크 (이미 있으면 업데이트 or 패스 정책 결정)
            existing = db.query(ProductInferenceInput).filter_by(product_id=item.product_id).first()
            
            if existing:
                existing.feature_data = item.feature_data
                existing.is_vectorized = False # 데이터가 바뀌었으니 다시 벡터화해야 함
            else:
                new_product = ProductInferenceInput(
                    product_id=item.product_id,
                    feature_data=item.feature_data,
                    is_vectorized=False
                )
                db.add(new_product)
            saved_count += 1
        
        db.commit()
        return {"status": "success", "message": f"Saved {saved_count} items."}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))