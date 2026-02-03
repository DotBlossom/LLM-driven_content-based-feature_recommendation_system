from typing import AsyncGenerator
from fastapi import FastAPI, APIRouter
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn

from utils.dependencies import initialize_global_models #initialize_rec_service

from APIController.controller import controller_router
from database import engine, Base
from APIController.serving_controller import serving_controller_router

# from train import train_router



        
@asynccontextmanager
async def lifespan(app:FastAPI) -> AsyncGenerator[None, None]:

    # 🌟 1. STARTUP (앱 시작 시 실행)
    
    print("✨ Lifespan 시작: DB conn ...")
    #Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("등록된 테이블 목록:", Base.metadata.tables.keys())
    print("✨ Lifespan 시작: DB conn 완료...")
    
    print("✨ Lifespan 시작: 모델 로딩 중...")
    
    
    # dependencies.py에 정의된 모델 로딩 로직을 호출합니다.
    # 모델 로딩이 완료된 후, 앱이 요청을 처리할 준비가 됩니다.
    initialize_global_models()

    
    
    print("✅ 모델 로딩 및 준비 완료.")
    
    # initialize_rec_service()
    print("✅ 추천시스템 로딩 및 준비 완료.")
    
    # yield 전의 코드는 Startup 시점에 실행됩니다.
    yield
    
    # 🌟 2. SHUTDOWN (앱 종료 시 실행)
    # yield 후의 코드는 애플리케이션이 종료될 때(서버가 꺼질 때) 실행됩니다.
    print("🔥 Lifespan 종료: 리소스 정리 중...")
    # 예: 데이터베이스 연결 해제, 캐시 정리, 모델 파일 메모리에서 삭제 등
    # cleanup_global_models() # 필요한 경우 정리 함수를 호출할 수 있습니다.
    print("👋 리소스 정리 완료.")
######################################


app = FastAPI(title="Model Inference API", lifespan=lifespan)

# CORS configuration for test
origins = [
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

#router
api_router = APIRouter(prefix="/ai-api")
api_router.include_router(serving_controller_router, prefix="/serving")
#api_router.include_router(train_router, prefix="/train")
app.include_router(api_router)

#separatable Instance
control_router = APIRouter(prefix="/api")
control_router.include_router(controller_router, prefix="/controller")

app.include_router(control_router)





#health check line
@app.get("/")
def home():
    cuda_status = torch.cuda.is_available()
    return {
    "message": "FastAPI가 정상 작동 중입니다!",
    "cuda_available": cuda_status  # boolean 값 그대로 전달
    }


#query test
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query_param": q}



if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5050, reload=True)