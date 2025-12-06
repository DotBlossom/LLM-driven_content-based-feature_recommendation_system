
from fastapi import FastAPI, Depends, HTTPException, APIRouter
from pydantic import BaseModel
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Tuple
from database import ProductInput, Vectors, get_db
import utils.vocab as vocab 
import numpy as np
from model import CoarseToFineItemTower
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert  

serving_controller_router = APIRouter()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# productList -> (CoarseToFineItemTower)를 I : N개로 확장?
# I : productList(featureForm)


# 가상의 ProductInput 타입과 vocab 객체 (기존 코드 문맥 따름)
# DEVICE, vocab 등은 전역 변수 혹은 인자로 관리된다고 가정

def preprocess_batch_input(products: List[ProductInput]) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_std_tokens = []
    batch_re_tokens = []
    
    for product in products:
        std_tokens = []
        re_tokens = []
        

        feature_data: Dict[str, Any] = getattr(product, 'feature_data', {})
        
        clothes_data = feature_data.get("clothes", {})
        re_data = feature_data.get("reinforced_feature_value", {})
        
        # 1. Standard Features 처리
        if clothes_data:
            # 딕셔너리 구조: {'key_name': ['value1', 'value2']}
            for key, values in clothes_data.items():
                if isinstance(values, list): # 값이 리스트인지 확인
                    for v in values:
                        tid = vocab.get_std_id(v)
                        if tid > 0:
                            std_tokens.append(tid)
        
        # 2. Reinforced Features 처리
        if re_data:
            for key, values in re_data.items():
                if isinstance(values, list): # 값이 리스트인지 확인 (이전 대화에서 리스트 형태였음)
                    for v in values:
                        tid = vocab.get_re_id(v)
                        if tid > 0:
                            re_tokens.append(tid)
        
        # 예외 처리: 데이터가 비어있으면 [0] (PAD) 넣기
        # (빈 리스트는 텐서 변환 시 문제를 일으킬 수 있음)
        if not std_tokens: std_tokens = [0]
        if not re_tokens: re_tokens = [0]
        
        # 리스트를 텐서로 변환하여 배치 리스트에 추가
        batch_std_tokens.append(torch.tensor(std_tokens, dtype=torch.long))
        batch_re_tokens.append(torch.tensor(re_tokens, dtype=torch.long))
    
    # 3. Padding 처리 및 Batch Tensor 생성
    # batch_first=True -> 결과 모양이 (Batch_Size, Max_Seq_Len)이 됨
    # padding_value=0 -> 빈 공간을 0으로 채움
    t_std_batch = pad_sequence(batch_std_tokens, batch_first=True, padding_value=0).to(DEVICE)
    t_re_batch = pad_sequence(batch_re_tokens, batch_first=True, padding_value=0).to(DEVICE)
    
    return t_std_batch, t_re_batch


## Batch API Layer


class TrainingItem(BaseModel):

    product_id: int
    feature_data: Dict[str, Any]

    class Config:
        # Pydantic에게 SQLAlchemy 객체로부터 속성을 읽어오도록 지시합니다. (핵심)
        from_attributes = True

@serving_controller_router.post("/update-vectors")
def process_and_save_vectors(
    products: List[TrainingItem], 
    db: Session = Depends(get_db),
    batch_size: int = 32
):
    
    """
    1. 데이터를 배치로 잘라 모델에 통과시킴
    2. 결과 벡터와 원본 상품의 ID, Category를 매핑
    3. DB에 즉시 저장 (Upsert)
    """
    CoarseToFineItemTower.eval()
    total_count = len(products)
    print(f"총 {total_count}개의 상품 벡터 생성을 시작합니다.")

    with torch.no_grad():
        # 1. 배치 단위 루프
        for i in range(0, total_count, batch_size):
            # -------------------------------------------------------
            # A. 데이터 준비 & 모델 Inference
            # -------------------------------------------------------
            batch_products = products[i : i + batch_size]
            
            # (이전 단계에서 만든 전처리 함수)
            t_std, t_re = preprocess_batch_input(batch_products)
            
            # 모델 실행 -> (Batch_Size, 128)
            batch_output = CoarseToFineItemTower(t_std, t_re)
            
            # CPU로 이동 및 Numpy 변환
            batch_vectors = batch_output.cpu().numpy()

            # -------------------------------------------------------
            # B. ID 매핑 및 DB 객체 생성
            # -------------------------------------------------------
            # batch_products[k] 와 batch_vectors[k] 는 서로 같은 상품입니다.
            
            # DB 작업을 위한 딕셔너리 리스트 생성 (Bulk Insert용)
            insert_data_list = []
            
            for product, vector in zip(batch_products, batch_vectors):
                insert_data_list.append({
                    "id": product.product_id,                  # PK
                    "category": product.feature_data.clothes.category,      # 메타데이터 수정필요!!!!!!!!!!!!
                    "vector_pre": vector.tolist(),     # numpy array -> list[float]
                    # "vector_triplet": None           # 필요하다면 null 처리 or 생략
                })

            # -------------------------------------------------------
            # C. DB 저장 (Upsert 처리)
            # -------------------------------------------------------
            if insert_data_list:
                # PostgreSQL의 INSERT ... ON CONFLICT DO UPDATE 구문 사용
                stmt = insert(Vectors).values(insert_data_list)
                
                # PK(id)가 이미 존재하면, vector_pre와 category를 업데이트한다.
                upsert_stmt = stmt.on_conflict_do_update(
                    index_elements=['id'],  # 충돌 기준 컬럼 (PK)
                    set_={
                        "vector_pre": stmt.excluded.vector_pre,
                        "category": stmt.excluded.category
                    }
                )
                
                db.execute(upsert_stmt)
                db.commit() # 배치 단위 커밋 (메모리 절약 및 트랜잭션 관리)
                
            print(f"Processing... {min(i + batch_size, total_count)} / {total_count}")

    print("모든 벡터 저장 완료.")
    
    
    