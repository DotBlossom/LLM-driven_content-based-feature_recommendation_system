from fastapi import APIRouter
import torch
from pytorch_metric_learning import losses, miners, distances



gpu_test_router = APIRouter() 


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