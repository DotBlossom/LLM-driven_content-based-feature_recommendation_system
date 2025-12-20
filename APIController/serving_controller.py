
import logging
import os
from fastapi import BackgroundTasks, FastAPI, Depends, HTTPException, APIRouter
from pydantic import BaseModel
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Tuple
from database import ProductInferenceInput, ProductInferenceVectors, UserSession, get_db
#from inference import RecommendationService
from train import UserTowerTrainDataset, train_final_user_tower, train_simcse_from_db #train_user_tower_task
from utils.dependencies import get_global_batch_size, get_global_encoder, get_global_projector #get_global_rec_service
import utils.vocab as vocab 
import numpy as np
from model import ALL_FIELD_KEYS, CoarseToFineItemTower, FinalUserTower, OptimizedItemTower, SimCSEModelWrapper, load_pretrained_vectors_from_db
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert  
import torch.nn as nn


serving_controller_router = APIRouter()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "models"



class ProductInput(BaseModel):
    """
    [모델 입력용 Pydantic 스키마]
    DB ORM 객체에서 변환하여 모델 입력으로 사용
    """
    product_id: int
    feature_data: Dict[str, Any]


## process-pending -> product_service로 스키마로딩 변경




# API 3 입력용
class ProductIdListSchema(BaseModel):
    product_ids: List[int]


import torch
from typing import List, Tuple, Dict, Any

# 전역 변수 ALL_FIELD_KEYS가 정의되어 있어야 합니다.
# 예: ALL_FIELD_KEYS = ["category", "season", "color", ...] 

def flatten_geometry_features(feature_data: Dict[str, Any]) -> None:
    """
    feature_data 내부의 'structural.geometry'를 찾아서
    상위 레벨인 'reinforced_feature_value'에 'geo_' 접두어로 풀어냅니다.
    (In-place modification)
    """
    re_data = feature_data.get("reinforced_feature_value", {})
    if not re_data:
        return

    # geometry 데이터가 있으면 꺼냄 (Dictionary에서 pop)
    geometry_data = re_data.pop("structural.geometry", None)
    
    # 딕셔너리 형태라면 펼쳐서 상위에 병합
    if geometry_data and isinstance(geometry_data, dict):
        for sub_key, sub_val in geometry_data.items():
            # 예: width_flow -> geo_width_flow
            new_key = f"geo_{sub_key}"
            re_data[new_key] = sub_val

def preprocess_batch_input(products: List[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    [Residual Field Embedding용 전처리 - Batch Optimization Ver.]
    
    최적화 원리:
    1. 개별 토크나이징 호출(N*M번)을 제거하고,
    2. 유효한 텍스트만 모아서 단 한 번의 Batch Tokenizing 수행
    """

            
    # 배치 크기 및 필드 수 계산
    B = len(products)
    F = len(ALL_FIELD_KEYS)
    S = vocab.RE_MAX_TOKEN_LEN
    
    # 1. 결과 텐서 미리 초기화 (전부 PAD로 채움)
    # 이렇게 하면 데이터가 없는 곳(None/Empty)은 건드릴 필요가 없어짐 (자동 패딩 효과)
    # Shape: (Batch, Num_Fields, Seq_Len)
    t_re_batch = torch.full((B, F, S), vocab.RE_TOKENIZER.pad_token_id, dtype=torch.long, device=DEVICE)
    
    batch_std_ids = []
    
    # [Batch Tokenizing을 위한 수집통]
    flat_texts = []      # 토크나이징 할 텍스트들
    flat_indices = []    # 그 텍스트가 들어갈 위치 (batch_idx, field_idx)
    
    for i, product in enumerate(products):
        
        raw_feature_data: Dict[str, Any] = getattr(product, 'feature_data', {})
        
        feature_data = raw_feature_data.copy()
        flatten_geometry_features(feature_data)
        
        clothes_data = feature_data.get("clothes", {})
        re_data = feature_data.get("reinforced_feature_value", {})
        
        # 1-3. 데이터 섹션 분리
        clothes_data = feature_data.get("clothes", {})
        re_data = feature_data.get("reinforced_feature_value", {})


        # ========================================================
        
        row_std_ids = []
        
        for j, key in enumerate(ALL_FIELD_KEYS):
            
            # --- A. STD ID (Lookup은 빠르므로 루프 유지) ---
            std_val = clothes_data.get(key)
            if isinstance(std_val, list):
                std_val = std_val[0] if std_val else None
            
            s_id = vocab.get_std_id(key, std_val)
            row_std_ids.append(s_id)
            
            # --- B. RE Text 수집 (토크나이징 X) ---
            re_val_list = re_data.get(key)
            re_text = None
            
            if re_val_list:
                if isinstance(re_val_list, list) and len(re_val_list) > 0:
                    re_text = str(re_val_list[0])
                elif isinstance(re_val_list, str):
                    re_text = re_val_list
            
            # 유효한 텍스트가 있는 경우에만 수집 리스트에 추가
            if re_text and re_text.strip():
                flat_texts.append(re_text)
                flat_indices.append((i, j)) # 좌표 기억 (i번째 상품, j번째 필드)
        
        batch_std_ids.append(row_std_ids)

    # 2. [핵심] Batch Tokenization (단 1회 호출)
    if flat_texts:
        # Rust 기반의 고속 병렬 처리 수행
        encoded = vocab.RE_TOKENIZER(
            flat_texts,
            padding='max_length',
            max_length=S,
            truncation=True,
            return_tensors='pt'
        )
        
        # encoded['input_ids'] shape: (N_valid_texts, Seq_Len)
        valid_tokens = encoded['input_ids'].to(DEVICE)
        
        # 3. [Scatter] 결과 텐서에 제자리 찾아 넣기 (Fancy Indexing)
        # rows: 배치 인덱스들, cols: 필드 인덱스들
        rows = [idx[0] for idx in flat_indices]
        cols = [idx[1] for idx in flat_indices]
        
        # 한 번에 할당 (for문 없이 텐서 연산으로 처리)
        t_re_batch[rows, cols] = valid_tokens

    # 4. STD 텐서 변환
    t_std_batch = torch.tensor(batch_std_ids, dtype=torch.long, device=DEVICE)

    return t_std_batch, t_re_batch



def generate_item_vectors(
    products: List[ProductInferenceInput], 
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
    encoder: CoarseToFineItemTower = Depends(get_global_encoder)

):
    total_processed_count = 0
    
    while True:
        # 1. 처리되지 않은 데이터 조회 (batch_size? 일단)
        pending_products = db.query(ProductInferenceInput)\
                             .filter(ProductInferenceInput.is_vectorized == False)\
                             .limit(batch_size)\
                             .all()
        
        if not pending_products:
            break

        # 2. 공통 파이프라인 실행
        current_count = run_pipeline_and_save(db, pending_products, encoder)
        
        total_processed_count += current_count

    if total_processed_count == 0:
        return {"status": "success", "message": "No pending products to process."}

    return {
        "status": "success", 
        "processed_count": total_processed_count, 
        "message": f"All pending batches processed successfully. (Total: {total_processed_count})"
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




# ------------------------------------------------------------------
# API 4. User Tower Train
# ------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class TrainRequest(BaseModel):
    epochs: int = 5
    batch_size: int = 1
    learning_rate: float = 1e-4
    max_seq_len: int = 50
    
    # 모델 저장 경로 (옵션)
    save_path: str = "./user_tower_latest.pth"

class TrainResponse(BaseModel):
    status: str
    final_avg_loss: float
    trained_epochs: int
    model_save_path: str
    message: str

# ==========================================
# 2. Service Layer (Pipeline Logic)
# ==========================================
def preprocess_db_sessions(sessions: list, product_id_map: dict, max_seq_len: int) -> List[dict]:
    """DB 세션 데이터를 학습용 데이터(Dict)로 변환"""
    data = []
    
    # 매핑 테이블
    GENDER_MAP = {'M': 1, 'F': 2, 'UNKNOWN': 0}
    SEASON_MAP = {'SPRING_AUTUMN': 1, 'SUMMER': 2, 'WINTER': 3, 'UNKNOWN': 0}
    
    for sess in sessions:
        if not sess.events: continue
        
        # 1. Sort Events
        events = sorted(sess.events, key=lambda e: e.timestamp)
        if len(events) < 2: continue # 최소 2개 (History 1 + Target 1)
        
        # 2. Map Product IDs
        seq_indices = [product_id_map.get(e.product_id, 0) for e in events]
        
        # 3. Create Row
        # 마지막 아이템 = Target, 그 이전 = History
        data.append({
            'history': seq_indices[:-1], 
            'target_idx': seq_indices[-1],
            'gender': GENDER_MAP.get(sess.user.gender, 0),
            'season': SEASON_MAP.get(sess.season, 0),
            'age': 0 # 필요시 추가 구현
        })
        
    return data

def run_training_pipeline(db: Session, config: TrainRequest) -> TrainResponse:
    logger.info("🚀 Starting Training Pipeline via API...")

    # 1. Load Pretrained Item Vectors (Fixed)
    # item_matrix: (Num_Total_Items + 1, 128)
    item_matrix, id_map = load_pretrained_vectors_from_db(db)
    logger.info(f"✅ Loaded {len(id_map)} item vectors from DB.")

    # 2. Fetch User Sessions (Training Data)
    # 실제로는 기간 쿼리 등을 추가해야 함
    sessions = db.query(UserSession).join(UserSession.user).join(UserSession.events).limit(5000).all()
    
    if not sessions:
        raise HTTPException(status_code=400, detail="No session data found in DB.")

    # 3. Preprocessing
    training_data = preprocess_db_sessions(sessions, id_map, config.max_seq_len)
    logger.info(f"✅ Prepared {len(training_data)} training samples.")
    
    if len(training_data) == 0:
        raise HTTPException(status_code=400, detail="Valid training data is empty after preprocessing.")

    # 4. DataLoader Setup
    dataset = UserTowerTrainDataset(training_data, id_map, max_seq_len=config.max_seq_len)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True
    )

    # 5. Initialize Final Use
    #r Tower
    num_total_items = item_matrix.size(0)
    user_tower = FinalUserTower(
        num_total_products=num_total_items - 1,
        pretrained_item_matrix=item_matrix, # Weight 초기화용
        max_seq_len=config.max_seq_len
    )

    # 6. Execute Training Loop
    # train_final_user_tower 함수가 학습된 모델을 반환한다고 가정
    trained_model = train_final_user_tower(
        user_tower=user_tower,
        pretrained_item_matrix=item_matrix, # Loss 계산용
        train_loader=train_loader,
        epochs=config.epochs,
        lr=config.learning_rate
    )

    # 7. Save Model
    torch.save(trained_model.state_dict(), config.save_path)
    logger.info(f"💾 Model saved to {config.save_path}")

    return TrainResponse(
        status="success",
        final_avg_loss=0.0, # Loop에서 마지막 Loss를 리턴받도록 수정 필요 (여기선 Dummy)
        trained_epochs=config.epochs,
        model_save_path=config.save_path,
        message=f"Training completed with {len(training_data)} samples."
    )

# ==========================================
# 3. API Endpoints
# ==========================================
@serving_controller_router.post("/train/user-tower", response_model=TrainResponse)
def trigger_training_job(req: TrainRequest, db: Session = Depends(get_db)):
    """
    유저 타워 학습을 실행합니다. (Synchronous for Demo)
    실제 운영 환경에서는 BackgroundTasks 또는 Celery를 권장합니다.
    """
    try:
        result = run_training_pipeline(db, req)
        return result
    except Exception as e:
        logger.error(f"Training Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))










'''


@serving_controller_router.post("/train/user-tower/start")
async def start_user_tower_training(
    background_tasks: BackgroundTasks,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    db: Session = Depends(get_db)
):
    """
    [User Tower Training API]
    1. DB에서 학습된 Item Vector를 로딩합니다.
    2. 유저 로그 데이터를 사용하여 User Tower를 학습시킵니다. (백그라운드)
    """
    
    # 백그라운드 태스크 등록
    # 주의: db 세션은 백그라운드 태스크가 끝날 때까지 살아있어야 하거나,
    # 태스크 내부에서 새로 생성하는 것이 안전할 수 있습니다. 
    # 여기서는 간단히 전달하지만, 실제로는 scoped_session 사용 권장.
    background_tasks.add_task(
        train_user_tower_task,
        db_session=db,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )
    
    return {
        "status": "success",
        "message": "User Tower training started in background.",
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr
        }
    }


@serving_controller_router.get("/recommend/{user_id}")
def recommend_products_to_user(
    user_id: int, 
    top_k: int = 5,
    db: Session = Depends(get_db),
    rec_service: RecommendationService = Depends(get_global_rec_service)
):
    """
    [User-to-Item 추천]
    1. 유저의 현재 상태(이력)를 기반으로 User Vector 추론
    2. DB에서 유사한 상품 검색
    """

    if rec_service is None:
        raise HTTPException(status_code=503, detail="Recommendation Service not ready")
    
    try:
        # 1. User Vector 추론
        user_vector = rec_service.get_user_vector(db, user_id)
        
        # 2. 유사 상품 검색 (Retrieval)
        candidates = rec_service.retrieve_similar_items(db, user_vector, top_k=top_k)
        
        # 3. 결과 포맷팅
        response = []
        for pid, category, dist in candidates:
            response.append({
                "product_id": pid,
                "category": category,
                "score": 1 - dist # Cosine Distance는 0에 가까울수록 좋으므로 Score로 변환 (1에 가까울수록 좋음)
            })
            
        return {
            "user_id": user_id,
            "recommendations": response
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@serving_controller_router.get("/recommend/ranker/{user_id}")
def recommend_products_to_user_ranker(
    user_id: int, 
    top_k: int = 5, # 최종적으로 보여줄 개수 (예: 10개)
    db: Session = Depends(get_db),
    rec_service: RecommendationService = Depends(get_global_rec_service)
):
    """
    [2-Stage Recommendation Pipeline]
    Stage 1. Retrieval: User Vector와 유사한 후보군을 넉넉하게 검색 (Top-K * 5)
    Stage 2. Ranking: DCN 모델을 사용하여 후보군을 정밀 재정렬
    """

    if rec_service is None:
        raise HTTPException(status_code=503, detail="Recommendation Service not ready")
    
    try:
        # ==========================================
        # [Stage 1] Retrieval (Candidate Generation)
        # ==========================================
        
        # 1. User Vector 추론 (기존 로직)
        user_vector_np = rec_service.get_user_vector(db, user_id)
        
        # 2. 후보군 검색 (Retrieval)
        # 랭킹 모델이 재정렬할 여지를 주기 위해, 요청된 top_k보다 더 많이(예: 5배) 검색합니다.
        candidate_k = top_k * 2
        # candidates 구조: [(pid, category, dist), ...]
        candidates = rec_service.retrieve_similar_items(db, user_vector_np, top_k=candidate_k)
        
        if not candidates:
            return {"user_id": user_id, "recommendations": []}

        # ==========================================
        # [Stage 2] Ranking (Re-ranking)
        # ==========================================
        
        # 3. 랭킹 모델 입력을 위한 데이터 준비
        candidate_pids = [c[0] for c in candidates]
        
        # 3-1. 후보 아이템들의 벡터 조회 (DB Query)
        # {pid: vector_list} 형태의 딕셔너리 반환 가정
        item_vector_map = rec_service.get_item_vectors_by_ids(db, candidate_pids)
        
        # 3-2. Tensor 변환 준비
        valid_candidates = [] # 벡터가 존재하는 유효한 후보만 필터링
        item_vectors_list = []
        
        for pid, category, dist in candidates:
            if pid in item_vector_map:
                valid_candidates.append({
                    "product_id": pid,
                    "category": category,
                    "base_score": 1 - dist # Retrieval 점수 (참고용)
                })
                item_vectors_list.append(item_vector_map[pid])
        
        if not valid_candidates:
             raise HTTPException(status_code=404, detail="Candidate vectors not found")

        # Tensor 변환
        user_tensor = torch.tensor(user_vector_np, dtype=torch.float32).to(DEVICE) # (128,)
        item_tensor = torch.tensor(item_vectors_list, dtype=torch.float32).to(DEVICE) # (N, 128)
        
        # Context Vector (선택 사항)
        # 만약 시간대, 요일 등의 컨텍스트 피처를 쓴다면 여기서 생성
        # 현재는 사용하지 않는다고 가정 (None 전달) 또는 0 벡터
        context_tensor = None 
        # context_tensor = torch.zeros(20, dtype=torch.float32).to(DEVICE) 

        # 4. 랭킹 모델 예측 (Inference)
        # rec_service 내부에 로드된 ranking_model 사용
        # predict_for_user는 (N,) 형태의 확률값(Score)을 반환
        ranking_scores = rec_service.ranking_model.predict_for_user(
            user_vec=user_tensor,
            item_vecs=item_tensor,
            context_vec=context_tensor
        )
        
        # 5. 점수 할당 및 정렬
        # Tensor를 리스트로 변환
        scores_list = ranking_scores.tolist()
        
        for i, candidate in enumerate(valid_candidates):
            candidate["ranking_score"] = scores_list[i]
            
        # 랭킹 점수(ranking_score) 기준 내림차순 정렬
        valid_candidates.sort(key=lambda x: x["ranking_score"], reverse=True)
        
        # 6. 최종 Top-K 자르기
        final_recommendations = valid_candidates[:top_k]
        
        return {
            "user_id": user_id,
            "count": len(final_recommendations),
            "recommendations": final_recommendations
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # 로그 기록 필요
        print(f"Error in recommendation: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")



# productList -> (CoarseToFineItemTower)를 I : N개로 확장?
# I : productList(featureForm)


# 가상의 ProductInput 타입과 vocab 객체 (기존 코드 문맥 따름)
# DEVICE, vocab 등은 전역 변수 혹은 인자로 관리된다고 가정



## Batch API Layer

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