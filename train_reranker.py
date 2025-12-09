# train_reranker.py
import torch
import numpy as np
from utils import vocab
from model import CoarseToFineItemTower  # 사용자님의 SimCSE 클래스
from model_reranker import build_reranker_model # 방금 만든 모듈

# 설정값
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBED_DIM = 64
TOTAL_ITEM_COUNT = 10000  # 전체 아이템 수 (예시)
RE_MAX_CAPACITY = 5000    # RE 최대 크기 (예시)
BATCH_SIZE = 16

def get_dummy_data(size=100):
    """
    [테스트용] DeepCTR 입력 형식에 맞는 더미 데이터 생성 함수
    실제로는 DB에서 조회한 user_logs를 prepare_deepfm_input() 함수로 변환해서 사용
    """
    data = {
        "user_height": np.random.rand(size),
        "user_weight": np.random.rand(size),
        "re_attributes": np.random.randint(1, RE_MAX_CAPACITY, size=(size, 10)), # (B, 10)
        "history_item_id": np.random.randint(1, TOTAL_ITEM_COUNT, size=(size, 50)), # (B, 50)
    }
    # STD 피처들 (category, color 등)
    for key in vocab.STD_VOCAB_CONFIG.keys():
        data[key] = np.random.randint(1, vocab.STD_VOCAB_SIZE, size=size)
        
    labels = np.random.randint(0, 2, size=size) # 0:비클릭, 1:클릭
    return data, labels

def main():
    print("🚀 Starting Reranker Training Pipeline...")

    # ------------------------------------------------------
    # 1. SimCSE Encoder 로드 (Pre-trained Weights 가져오기 위함)
    # ------------------------------------------------------
    print("1️⃣ Loading SimCSE Encoder...")
    simcse_encoder = CoarseToFineItemTower(embed_dim=EMBED_DIM).to(DEVICE)
    
    # 실제 학습된 파일이 있다면 로드 (없으면 랜덤 가중치로 진행됨)
    try:
        simcse_encoder.load_state_dict(torch.load("saved_models/encoder_stage1.pth"))
        print("   ✅ SimCSE weights loaded from file.")
    except FileNotFoundError:
        print("   ⚠️ No pre-trained file found. Using random init (for testing).")

    # ------------------------------------------------------
    # 2. DeepFM Reranker 빌드 (가중치 이식 포함)
    # ------------------------------------------------------
    print("2️⃣ Building DeepFM Reranker...")
    reranker_model = build_reranker_model(
        simcse_encoder=simcse_encoder,
        total_item_count=TOTAL_ITEM_COUNT,
        re_max_capacity=RE_MAX_CAPACITY,
        embedding_dim=EMBED_DIM,
        device=DEVICE
    )
    
    # 모델 컴파일 (Optimizer, Loss 설정)
    reranker_model.compile(
        optimizer="adam", 
        loss="binary_crossentropy", 
        metrics=["binary_crossentropy", "auc"]
    )

    # ------------------------------------------------------
    # 3. 데이터 준비 (DB -> DeepCTR Input Format)
    # ------------------------------------------------------
    print("3️⃣ Preparing Training Data...")
    # 실제 환경: train_input = prepare_deepfm_input(db_logs, db_products)
    train_input, train_labels = get_dummy_data(size=1000) 

    # ------------------------------------------------------
    # 4. 학습 실행
    # ------------------------------------------------------
    print("4️⃣ Training Start!")
    history = reranker_model.fit(
        train_input, 
        train_labels, 
        batch_size=BATCH_SIZE, 
        epochs=3, 
        validation_split=0.2,
        verbose=1
    )

    # ------------------------------------------------------
    # 5. 모델 저장
    # ------------------------------------------------------
    print("5️⃣ Saving Reranker Model...")
    torch.save(reranker_model.state_dict(), "saved_models/reranker_deepfm.pth")
    print("✅ Pipeline Finished.")

if __name__ == "__main__":
    main()