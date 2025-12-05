from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import torch
import model 
import vocab 
import numpy as np


# --- 라우터 선언 ---
embed_items_router = APIRouter()

# --- 설정 ---
# 모델의 기본 임베딩 차원(64d)과 출력 차원을 model.py에서 가져와 일관성을 유지합니다.
EMBED_DIM = model.EMBED_DIM_CAT  # 64
OUTPUT_DIM = model.OUTPUT_DIM_ITEM_TOWER # 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 모델 초기화 ---
# 이 파일이 import 될 때 모델이 메모리에 로드됩니다.
try:
    item_tower = model.CoarseToFineItemTower(
        embed_dim=EMBED_DIM,
        output_dim=OUTPUT_DIM
    ).to(DEVICE)
    item_tower.eval()
    print(f"Item Tower Model Loaded on {DEVICE} with Dual Embeddings.")
except Exception as e:
    print(f"Model Load Error: {e}")


# --- Request Schema ---
class ProductInput(BaseModel):
    id: int
    # 1. 일반 피쳐 (clothes) -> Standard Input
    clothes: Dict[str, List[str]] 
    # 2. 강화 피쳐 (reinforced) -> Reinforced Input
    reinforced_feature_value: Optional[Dict[str, List[str]]] = {}

class EmbeddingOutput(BaseModel):
    product_id: int
    vector: List[float]

# --- Helper: JSON -> Two Tensors (Std, Re) 분리 ---
def preprocess_split_input(product: ProductInput) -> Tuple[torch.Tensor, torch.Tensor]:
    std_tokens = []
    re_tokens = []
    
    # 1. Standard Features (clothes 필드) 처리
    for key, values in product.clothes.items():
        for v in values:
            tid = vocab.get_std_id(v)
            if tid > 0:
                std_tokens.append(tid)
    
    # 2. Reinforced Features 처리
    if product.reinforced_feature_value:
        for key, values in product.reinforced_feature_value.items():
            for v in values:
                tid = vocab.get_re_id(v) 
                if tid > 0:
                    re_tokens.append(tid)
    
    # 예외 처리: 데이터가 비어있으면 0(PAD) 넣기
    if not std_tokens: std_tokens = [0]
    if not re_tokens: re_tokens = [0]
    
    # (1, Seq_Len) 형태로 변환
    t_std = torch.tensor([std_tokens], dtype=torch.long).to(DEVICE)
    t_re = torch.tensor([re_tokens], dtype=torch.long).to(DEVICE)
    
    return t_std, t_re

# --- Endpoints ---

# product 여러개 및, product category 추출해서 return 필요.

@embed_items_router.post("/embed", response_model=EmbeddingOutput)
def embed_product(product: ProductInput):
    """
    상품 데이터를 받아 Coarse-to-Fine 임베딩을 생성합니다.
    """
    try:
        # 1. 전처리 (두 개의 텐서로 분리)
        t_std, t_re = preprocess_split_input(product)
        
        # 2. 추론 (std와 re를 따로 넣음)
        with torch.no_grad():
            vector = item_tower(t_std, t_re)
        # DB 에 Return + category 저장.            
        # 3. 결과 반환
        return {
            "product_id": product.id,
            "vector": vector.cpu().numpy()[0].tolist()
        }
        
    except Exception as e:
        # 모델 초기화 오류가 발생했을 경우를 대비하여 item_tower의 존재 유무를 확인합니다.
        if 'item_tower' not in globals() or item_tower is None:
            raise HTTPException(status_code=503, detail="Model initialization failed. Cannot process request.")
            
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")