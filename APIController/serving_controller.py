
import os
from fastapi import BackgroundTasks, FastAPI, Depends, HTTPException, APIRouter
from pydantic import BaseModel
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Tuple
from database import ProductInferenceInput, ProductInferenceVectors, ProductInput, Vectors, get_db
from train import train_simcse_from_db
from utils.dependencies import get_global_batch_size, get_global_encoder, get_global_projector
import utils.vocab as vocab 
import numpy as np
from model import ALL_FIELD_KEYS, CoarseToFineItemTower, OptimizedItemTower, SimCSEModelWrapper
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert  
import torch.nn as nn

serving_controller_router = APIRouter()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "models"

# API 3 입력용
class ProductIdListSchema(BaseModel):
    product_ids: List[int]


import torch
from typing import List, Tuple, Dict, Any

# 전역 변수 ALL_FIELD_KEYS가 정의되어 있어야 합니다.
# 예: ALL_FIELD_KEYS = ["category", "season", "color", ...] 

def preprocess_batch_input(products: List[ProductInput]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    [Residual Field Embedding용 전처리]
    딕셔너리를 순회하는 것이 아니라, 고정된 'ALL_FIELD_KEYS'를 순회하여
    Tensor의 각 인덱스가 항상 특정 필드(속성)를 가리키도록 정렬합니다.
    """
    batch_std_ids = []
    batch_re_ids = []
    
    for product in products:
        # 1. 데이터 추출
        feature_data: Dict[str, Any] = getattr(product, 'feature_data', {})
        clothes_data = feature_data.get("clothes", {})
        re_data = feature_data.get("reinforced_feature_value", {})
        
        row_std_ids = []
        row_re_ids = []

        # 2. [핵심] 고정된 Key 리스트를 순회 (순서 및 위치 보장)
        for key in ALL_FIELD_KEYS:
            
            # --- A. STD ID 추출 ---
            std_val = clothes_data.get(key)
            
            # 리스트로 들어오는 경우 첫 번째 값 사용 (단일 라벨 가정)
            if isinstance(std_val, list) and len(std_val) > 0:
                std_val = std_val[0]
            elif isinstance(std_val, list) and len(std_val) == 0:
                std_val = None
                
            # vocab.py의 함수 호출 (Key 정보도 함께 전달하여 확장성 확보)
            # 값이 없으면(None) 내부에서 0(PAD) 반환
            s_id = vocab.get_std_id(key, std_val)
            row_std_ids.append(s_id)
            
            
            # --- B. RE ID 추출 (Hashing) ---
            re_val_list = re_data.get(key)
            re_val = None
            
            # RE 데이터는 보통 List 형태이므로 첫 번째 값 추출
            if re_val_list and isinstance(re_val_list, list) and len(re_val_list) > 0:
                re_val = re_val_list[0]
            elif isinstance(re_val_list, str):
                re_val = re_val_list
            
            # Hashing 함수 호출 (저장 X, 즉시 변환)
            # 값이 없으면(None) 내부에서 0(PAD) 반환
            r_id = vocab.get_re_hash_id(re_val)
            row_re_ids.append(r_id)

        # 3. 행 단위 추가
        # 이제 row_std_ids의 길이는 항상 len(ALL_FIELD_KEYS)로 고정됨
        batch_std_ids.append(row_std_ids)
        batch_re_ids.append(row_re_ids)
    
    # 4. 텐서 변환 (pad_sequence 불필요 -> torch.tensor로 직변환)
    # Shape: (Batch_Size, Num_Fields)
    t_std_batch = torch.tensor(batch_std_ids, dtype=torch.long, device=DEVICE)
    t_re_batch = torch.tensor(batch_re_ids, dtype=torch.long, device=DEVICE)

    return t_std_batch, t_re_batch



def generate_item_vectors(
    products: List[ProductInput], 
    encoder: nn.Module 
    
) -> Dict[int, List[float]]:
    """
    [Core Inference Logic]
    ProductInput 리스트 -> Encoder(Stage1) -> L2 Normalize -> {product_id: vector} 반환
    """
    if not products:
        return {}

    # 1. 모델 Wrapper 설정 및 Eval 모드
    model = encoder.to(DEVICE)
    model.eval()

    # 2. 전처리 (collate_fn 로직 포함된 함수 사용 가정)
    try:
        t_std, t_re = preprocess_batch_input(products)
    except Exception as e:
        print(f"❌ Preprocessing Error: {e}")
        return {}

    t_std = t_std.to(DEVICE)
    t_re = t_re.to(DEVICE)

    # 3. 추론 (No Grad)
    with torch.no_grad():
        raw_v = model(t_std, t_re)
        final_vectors_tensor = F.normalize(raw_v, p=2, dim=1)
    # 4. 결과 변환
    vectors_list = final_vectors_tensor.cpu().numpy().tolist()
    
    result_map = {}
    for idx, product in enumerate(products):
        result_map[product.product_id] = vectors_list[idx]

    return result_map




def run_pipeline_and_save(
    db_session: Session, 
    products: List[ProductInferenceInput],
    encoder: nn.Module     
    
):
    """
    [공통 로직] 
    DB 객체 리스트 -> Pydantic 변환 -> 추론 -> 벡터 저장 -> Flag 업데이트
    """
    if not products:
        return 0

    # 1. DB 객체(ORM)를 모델 입력용 Pydantic 객체로 변환
    input_list = [
        ProductInput(product_id=p.product_id, feature_data=p.feature_data)
        for p in products
    ]


    # 1-1. load

    encoder_path = os.path.join(MODEL_DIR, "encoder_stage1.pth")
    #projector_path = os.path.join(MODEL_DIR, "projector_stage2.pth")

    if os.path.exists(encoder_path): #and os.path.exists(projector_path):
        try:
            encoder_state = torch.load(encoder_path, map_location=DEVICE)
            #projector_state = torch.load(projector_path, map_location=DEVICE)
            
            encoder.load_state_dict(encoder_state)
            #projector.load_state_dict(projector_state)
            
            print("✅ Models loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading state dicts: {e}")
            raise e
    else:
        raise FileNotFoundError(f"❌ Model files not found in {MODEL_DIR}")


    # 2. 실제 모델 추론 실행 (generate_item_vectors 호출)
    #    결과는 {product_id: [0.12, 0.55, ...]} 형태
    try:
        vector_map = generate_item_vectors(input_list, encoder)
    except Exception as e:
        print(f"❌ Inference Failed: {e}")
        raise e

    # 3. 결과 저장 및 플래그 업데이트
    for p in products:
        # 혹시 모를 에러로 특정 ID가 누락됐는지 확인
        if p.product_id not in vector_map:
            continue
            
        vector_val = vector_map[p.product_id]
        
        # 벡터 테이블에 저장 (Upsert 로직)
        existing_vec = db_session.query(ProductInferenceVectors).filter_by(id=p.product_id).first()
        if existing_vec:
            existing_vec.vector_embedding= vector_val
        else:
            new_vec = ProductInferenceVectors(id=p.product_id, vector_embedding=vector_val)
            db_session.add(new_vec)
        
        # [작업 완료 Flag 처리]
        p.is_vectorized = True
    
    db_session.commit()
    print("✅ Saved Item Vectors (by encoder) successfully.")
    return len(vector_map)





# --- API 2. 학습 요청 (Background Task) ---
@serving_controller_router.post("/train/start")
async def start_training(background_tasks: BackgroundTasks,
                         encoder_instance: CoarseToFineItemTower = Depends(get_global_encoder), 
                         projector_instance: OptimizedItemTower = Depends(get_global_projector),
                         g_batch_size: int = Depends(get_global_batch_size)):
    """
    [API 2] DB에 있는 데이터로 SimCSE 학습을 시작합니다. (비동기 실행)
    """
    # 백그라운드에서 실행되도록 넘김 (API는 즉시 응답)
    background_tasks.add_task(train_simcse_from_db,
        encoder=encoder_instance,
        projector=projector_instance,
        batch_size = g_batch_size
    )
    
    return {"message": "Training started in the background.", "status": "processing"}


# batch size 맞춰야함
@serving_controller_router.post("/vectors/process-pending")
def process_pending_vectors(
    batch_size: int = Depends(get_global_batch_size),
    db: Session = Depends(get_db),
    # [수정] 모델 인스턴스 주입
    encoder: CoarseToFineItemTower = Depends(get_global_encoder),
    projector: OptimizedItemTower = Depends(get_global_projector)
):
    # 1. 처리되지 않은 데이터 조회
    pending_products = db.query(ProductInferenceInput)\
                         .filter(ProductInferenceInput.is_vectorized == False)\
                         .limit(batch_size)\
                         .all()
    
    if not pending_products:
        return {"status": "success", "message": "No pending products to process."}

    # 2. 공통 파이프라인 실행 (모델 전달)
    processed_count = run_pipeline_and_save(db, pending_products, encoder)
    
    return {
        "status": "success", 
        "processed_count": processed_count, 
        "message": "Batch processing completed."
    }


# ------------------------------------------------------------------
# API 3. 특정 ID 리스트 기반 벡터화 (On-Demand Processing)
# ------------------------------------------------------------------
@serving_controller_router.post("/vectors/process-by-ids")
def process_vectors_by_ids(
    payload: ProductIdListSchema, 
    db: Session = Depends(get_db),
    # [수정] 모델 인스턴스 주입
    encoder: CoarseToFineItemTower = Depends(get_global_encoder)
    #projector: OptimizedItemTower = Depends(get_global_projector)
):
    # 1. ID 조회
    target_products = db.query(ProductInferenceInput)\
                        .filter(ProductInferenceInput.product_id.in_(payload.product_ids))\
                        .all()
    
    if not target_products:
        raise HTTPException(status_code=404, detail="No products found for given IDs.")

    # 2. 공통 파이프라인 실행 (모델 전달)
    processed_count = run_pipeline_and_save(db, target_products, encoder)
    
    return {
        "status": "success", 
        "processed_count": processed_count, 
        "message": "On-demand processing completed."
    }





# productList -> (CoarseToFineItemTower)를 I : N개로 확장?
# I : productList(featureForm)


# 가상의 ProductInput 타입과 vocab 객체 (기존 코드 문맥 따름)
# DEVICE, vocab 등은 전역 변수 혹은 인자로 관리된다고 가정



## Batch API Layer
'''
class EmbeddingRequestItem(BaseModel):

    product_id: int
    feature_data: Dict[str, Any]

    class Config:
        # Pydantic에게 SQLAlchemy 객체로부터 속성을 읽어오도록 지시합니다. (핵심)
        from_attributes = True

@serving_controller_router.post("/update-vectors")
def process_and_save_vectors(
    products: List[EmbeddingRequestItem], 
    db: Session = Depends(get_db),
    batch_size: int = 32
):
    

    #1. 데이터를 배치로 잘라 모델에 통과시킴
    #2. 결과 벡터와 원본 상품의 ID, Category를 매핑
    #3. DB에 즉시 저장 (Upsert)

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
    
'''    