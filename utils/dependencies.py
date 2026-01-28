# dependencies.py
from typing import Optional

import torch
from database import SessionLocal
#from inference import RecommendationService
from model import CoarseToFineItemTower, OptimizedItemTower, SimCSEModelWrapper

# 1. 모델 인스턴스를 저장할 전역 변수
global_encoder: Optional[CoarseToFineItemTower] = None
global_projector: Optional[OptimizedItemTower] = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#rec_service: RecommendationService = None

'''
def initialize_rec_service():
    global rec_service
    
    # DB 세션 열기: 모델 로딩에 필요한 아이템 벡터, ID 맵 등을 DB에서 가져오기 위해 필요
    db = SessionLocal()
    try:
        # RecommendationService 초기화
        # model_path는 'models/user_tower_latest.pth'와 같이 상대 경로를 권장합니다.
        model_path = "models/user_tower_symmetric_final.pth" 
        rec_service = RecommendationService(db_session=db, model_path=model_path)
    except Exception as e:
        print(f"❌ Recommendation Service 초기화 중 오류 발생: {e}")
        # 오류 발생 시 앱 시작을 중단하거나, rec_service를 None으로 유지하여 503 오류를 반환하도록 처리
        rec_service = None
    finally:
        # 모델 로딩 후 DB 세션을 즉시 닫아줍니다.
        db.close()
'''     


# 2. 모델 로딩 함수 (main.py의 startup 이벤트에서 호출됨)
def initialize_global_models():
    """
    모델 인스턴스를 로드하고 전역 변수에 저장합니다.
    FastAPI의 startup 이벤트 핸들러에서 호출
    """
    global global_encoder
    global global_projector
    
    print("🚀 앱 시작: CoarseToFineItemTower 로딩 중...")
    global_encoder = CoarseToFineItemTower(embed_dim=64, output_dim=128)
    print("✅ CoarseToFineItemTower 로드 완료.")

    print("🚀 앱 시작: OptimizedItemTower 로딩 중...")
    global_projector = OptimizedItemTower(input_dim=128, output_dim=128)
    print("✅ OptimizedItemTower 로드 완료.")

    global global_batch_size
    global_batch_size = 128
    print(f"✅ Global Batch Size set to: {global_batch_size}")
    

# 3. 의존성 주입(DI) 제공자 함수
def get_global_encoder() -> CoarseToFineItemTower:
    """저장된 CoarseToFineItemTower 인스턴스를 반환하는 의존성 주입 함수."""
    if global_encoder is None:
        # 이 예외는 startup 이벤트가 실행되지 않았을 때만 발생
        raise Exception("Encoder model has not been loaded yet. Check application startup events.")
    return global_encoder

def get_global_projector() -> OptimizedItemTower:
    """저장된 OptimizedItemTower 인스턴스를 반환하는 의존성 주입 함수."""
    if global_projector is None:
        raise Exception("Projector model has not been loaded yet. Check application startup events.")
    return global_projector

def get_global_batch_size() -> int:
    
    if global_batch_size is None:
        raise Exception("global batch size has not been defined")
    return global_batch_size

'''
def get_global_rec_service() -> RecommendationService:
    """저장된 RecommendationService 인스턴스를 반환하는 의존성 주입 함수."""
    if rec_service is None:
        raise Exception("Recommendation Service has not been initialized yet. Check application startup events.")
    return rec_service
'''