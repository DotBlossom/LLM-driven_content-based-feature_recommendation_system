
import logging
import os
from fastapi import BackgroundTasks, FastAPI, Depends, HTTPException, APIRouter,status
from pydantic import BaseModel, Field
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Optional, Tuple
from database import ProductInferenceInput, ProductInferenceVectors, UserSession, get_db
#from inference import RecommendationService
#from train import UserTowerTrainDataset, train_final_user_tower, train_simcse_from_db #train_user_tower_task
#from utils.dependencies import get_global_batch_size, get_global_encoder, get_global_projector #get_global_rec_service
from item_tower import HybridItemTower, OptimizedItemTower, train_simcse_from_db
from utils.dependencies import get_global_batch_size, get_global_encoder, get_global_projector
from utils.inference_utils import generate_and_save_item_vectors
import utils.vocab as vocab 
import numpy as np
# from model import ALL_FIELD_KEYS, CoarseToFineItemTower, FinalUserTower, OptimizedItemTower, SimCSEModelWrapper, load_pretrained_vectors_from_db
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





            
@serving_controller_router.post("/train/item-tower")
def train_item_tower(encoder: nn.Module = Depends(get_global_encoder),
                     projector: nn.Module = Depends(get_global_projector),
                     db: Session = Depends(get_db),
                     batch_size: int = Depends(get_global_batch_size),
                     epochs: int = 5,
                     lr: float = 5e-5,
                     checkpoint_path : str =None ):
            
    train_simcse_from_db(encoder, projector, db_session=db, batch_size=batch_size, epochs=epochs, lr = lr, checkpoint_path= checkpoint_path)
'''
class ItemTowerFineTuneRequest(BaseModel):
    checkpoint_path: str = Field(..., description="Epoch 3 체크포인트 경로 (예: models/encoder_ep03...pth)")
    epochs: int = Field(2, description="추가 학습 에포크 (기본 2)")
    batch_size: int = Field(64, description="배치 사이즈")
    lr: float = Field(5e-5, description="Fine-tuning 학습률 (기본 5e-5)")
    dropout_prob: float = Field(0.2, description="Fine-tuning 드롭아웃 (기본 0.2)")
    temperature: float = Field(0.08, description="Temperature (기본 0.08)")

@serving_controller_router.post("/train/item-tower/finetune/sync")
def start_item_tower_finetune_sync(
    req: ItemTowerFineTuneRequest,
    db: Session = Depends(get_db)  # DB 세션 주입
):
    """
    [동기 실행] Item Tower Fine-tuning
    - 요청을 보내면 학습이 완료될 때까지 기다렸다가 응답을 반환합니다.
    - 클라이언트 Timeout을 매우 길게 설정해야 합니다.
    """
    
    # 1. 체크포인트 파일 확인
    if not os.path.exists(req.checkpoint_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint file not found: {req.checkpoint_path}")

    print(f"⏳ [Sync] Fine-tuning requested. This may take a while...")

    try:
        # 이 파라미터들은 프로젝트 설정(config)에서 가져오거나 상수로 정의되어 있어야 합니다.
        encoder = HybridItemTower(
            std_vocab_size=STD_VOCAB_SIZE,
            num_std_fields=NUM_STD_FIELDS,
            embed_dim=EMBED_DIM,
            output_dim=EMBED_DIM
        )
        projector = OptimizedItemTower(
            input_dim=EMBED_DIM, 
            output_dim=EMBED_DIM
        )
    # 3. 학습 함수 직접 호출 (여기서 시간이 오래 걸림)
    try:
        train_simcse_from_db(
            encoder = Depends(get_global_encoder),
            projector = Depends(get_global_projector),
            db_session=db,              # 주입받은 세션 전달
            batch_size=req.batch_size,
            epochs=req.epochs,
            lr=req.lr,
            checkpoint_path=req.checkpoint_path,  # ✅ 체크포인트 로드
            dropout_prob=req.dropout_prob,        # ✅ 드롭아웃 적용 (0.2)
            temperature=req.temperature
        )
    except Exception as e:
        print(f"❌ Training Error: {e}")
        raise HTTPException(status_code=500, detail=f"Training Failed: {str(e)}")

    # 4. 완료 후 응답 반환
    return {
        "status": "success",
        "message": "Fine-tuning completed successfully.",
        "details": {
            "resumed_from": req.checkpoint_path,
            "trained_epochs": req.epochs,
            "final_lr": req.lr,
            "used_dropout": req.dropout_prob
        }
    }

'''
class VectorUpdateResponse(BaseModel):
    status: str
    message: str
    saved_path_matrix: str
    saved_path_ids: str
    item_count: int
    vector_shape: list
@serving_controller_router.post("/bg/inference/refresh-item-vectors" ,response_model=VectorUpdateResponse)
def update_item_vectors_api(
    save_dir: str = "models",  # 저장 경로를 파라미터로 받을 수 있게 함
    db: Session = Depends(get_db),
    checkpoint_path : Optional[str] = None
):
    """
    [관리자용] DB의 모든 아이템을 로드하여 Pre-trained Vector Matrix를 생성 및 갱신합니다.
    - User Tower 학습 전에 반드시 수행되어야 합니다.
    - 수행 시간이 오래 걸릴 수 있습니다. (대량 데이터 시 BackgroundTasks 권장)
    """
    try:
        print(f"🔄 [API] Request received: Update item vectors in '{save_dir}'")
        
        # 1. 벡터 생성 함수 호출 (리팩토링된 함수)
        # 반환값: (Tensor, List[str])
        final_tensor, ordered_ids = generate_and_save_item_vectors(db, save_dir,checkpoint_path=checkpoint_path)
        
        if final_tensor is None:
             raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Vector generation returned None. Check server logs."
            )

        # 2. 결과 응답 생성
        tensor_path = os.path.join(save_dir, "pretrained_item_matrix.pt")
        ids_path = os.path.join(save_dir, "item_ids.pt")

        return VectorUpdateResponse(
            status="success",
            message="Item vectors successfully updated and aligned.",
            saved_path_matrix=tensor_path,
            saved_path_ids=ids_path,
            item_count=len(ordered_ids),
            vector_shape=list(final_tensor.shape)
        )

    except Exception as e:
        print(f"❌ [API Error] {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update vectors: {str(e)}"
        )
'''

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

