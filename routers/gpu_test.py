from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

import torch
from pytorch_metric_learning import losses, miners, distances

from pydantic import BaseModel
from typing import List, Optional
import model

gpu_test_router = APIRouter() 

class ClothesInfo(BaseModel):
    category: List[str] 

class ProductItem(BaseModel):
    id: int
    clothes: ClothesInfo
    vector: Optional[List[float]] = None 

class TrainRequest(BaseModel):
    products: List[ProductItem]
    epochs: int = 5
    batch_size: int = 32

class InferenceRequest(BaseModel):
    vector: List[float]

'''
@gpu_test_router.post("/train")
async def train_endpoint(req: TrainRequest, background_tasks: BackgroundTasks):
    product_list = [item.dict() for item in req.products]
    background_tasks.add_task(model.train_model, product_list, req.epochs, req.batch_size)
    return {"message": "Training started in background."}


@gpu_test_router.post("/train_sync")
def train_sync_endpoint(req: TrainRequest):
    product_list = [item.dict() for item in req.products]
    try:
        history = model.train_model(product_list, req.epochs, req.batch_size)
        return {"status": "success", "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@gpu_test_router.post("/inference")
def inference_endpoint(req: InferenceRequest):
    if len(req.vector) != 512:
        raise HTTPException(status_code=400, detail="Input vector must be 512 dimensions.")
    try:
        optimized_vec = model.load_and_infer(req.vector)
        return {"input_dim": 512, "output_dim": 128, "vector": optimized_vec}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found. Please train the model first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@gpu_test_router.get("/metric")
def test_result():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    # 2. Metric Learning 테스트 (Triplet Loss 예제)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ GPU 모드로 Metric Learning을 테스트합니다.")
    else:
        device = torch.device("cpu")
        print("⚠️ CPU 모드입니다.")

    # 더미 데이터 생성 (배치사이즈 32, 128차원 벡터)
    embeddings = torch.randn(32, 128).to(device)
    labels = torch.randint(0, 10, (32,)).to(device)

    # 라이브러리 기능 사용 (거리 계산 -> 마이닝 -> 로스 계산)
    distance_func = distances.CosineSimilarity()
    loss_func = losses.TripletMarginLoss(distance=distance_func)
    miner_func = miners.TripletMarginMiner(distance=distance_func)

    # 마이닝 및 로스 계산
    hard_pairs = miner_func(embeddings, labels)
    loss = loss_func(embeddings, labels, hard_pairs)

    print(f"계산된 Loss 값: {loss.item()}")
    print("🎉 설치가 완벽합니다!")
    return {
        "계산된 Loss 값" : loss.item(),
        "completed" : "yes"
    }
    
    
'''