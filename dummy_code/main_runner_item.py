# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import torch
import dummy_code.model as model
import utils.vocab as vocab # 이전에 만든 vocab.py (토큰 매핑용)

app = FastAPI(title="Coarse-to-Fine Item Tower")

# --- 설정 ---
EMBED_DIM = 64
OUTPUT_DIM = 512 # 최적화 타워로 보낼 차원
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 초기화 (새로운 모델로 교체!)
item_tower = model.CoarseToFineItemTower(
    vocab_size=vocab.VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    output_dim=OUTPUT_DIM
).to(DEVICE)
item_tower.eval()

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
            tid = vocab.get_token_id(v)
            if tid > 0:
                std_tokens.append(tid)
    
    # 2. Reinforced Features 처리
    if product.reinforced_feature_value:
        for key, values in product.reinforced_feature_value.items():
            for v in values:
                tid = vocab.get_token_id(v)
                if tid > 0:
                    re_tokens.append(tid)
    
    # 예외 처리: 데이터가 비어있으면 0(PAD) 넣기
    if not std_tokens: std_tokens = [0]
    if not re_tokens: re_tokens = [0] # 강화 피쳐가 없어도 0을 넣어줘야 에러 안 남
    
    # (1, Seq_Len) 형태로 변환
    t_std = torch.tensor([std_tokens], dtype=torch.long).to(DEVICE)
    t_re = torch.tensor([re_tokens], dtype=torch.long).to(DEVICE)
    
    return t_std, t_re

# --- Endpoints ---
@app.post("/embed", response_model=EmbeddingOutput)
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
            
        # 3. 결과 반환
        return {
            "product_id": product.id,
            "vector": vector.cpu().numpy()[0].tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

