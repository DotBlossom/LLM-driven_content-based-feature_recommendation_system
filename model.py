from typing import Any, Dict, List, Union
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import Column, select
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from pytorch_metric_learning import losses, miners, distances
from collections import defaultdict
import random
import os
import numpy as np
import math

import vocab
from database import SessionLocal

from sqlalchemy.orm import Session
import copy
import random
from tqdm import tqdm

# ItemTowerEmbedding(S1) * N -> save..DB -> stage2 (optimizer pass -> triplet)  

model_router = APIRouter()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainingItem(BaseModel):

    product_id: int
    feature_data: Dict[str, Any]

# --- Global Configuration (전체 시스템이 참조하는 공통 차원) ---
EMBED_DIM_CAT = 64 # Feature의 임베딩 차원 (Transformer d_model)
OUTPUT_DIM_TRIPLET = 128 # Stage 2 최종 압축 차원
OUTPUT_DIM_ITEM_TOWER = 128 # Stage 1 최종 출력 차원 (Triplet Tower Input)
RE_MAX_CAPACITY = 50000 # <<<<<<<<<<<< RE 토큰의 최대 개수를 미리 할당
# ----------------------------------------------------------------------
# 1. Utility Modules (Shared for both Item Tower and Optimization Tower)
# ----------------------------------------------------------------------

# --- Residual Block (Corrected for Skip Connection) ---
class ResidualBlock(nn.Module):

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        # 블록 내에서 차원을 유지하는 2개의 Linear Layer (Skip Connection 전 처리)
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        # x + block(x) -> 잔차 연결 (핵심!)
        return self.relu(residual + out)

# --- Deep Residual Head (Pyramid Funnel) ---
class DeepResidualHead(nn.Module):
    """
    Categorical Vector(64d) -> 256 -> 128
    """
    def __init__(self, input_dim, output_dim=OUTPUT_DIM_ITEM_TOWER):
        super().__init__()
        
        # 1. 내부 확장 (Expansion): 표현력을 위해 4배 확장은 유지 (64 -> 256)
        hidden_dim = input_dim * 4 
        
        self.expand = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 2. Deep Interaction (ResBlocks): 256차원에서 특징 추출
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim), # 256 유지
            ResidualBlock(hidden_dim)  # 256 유지
        )
        
        # 3. Projection (Compression): 바로 목표 차원(128)으로 압축
        self.project = nn.Linear(hidden_dim, output_dim) 
        
    def forward(self, x):
        x = self.expand(x)      # 64 -> 256
        x = self.res_blocks(x)  # 256 -> 256 (Deep Feature Extraction)
        x = self.project(x)     # 256 -> 128 (Final Output)
        return x
 
# ----------------------------------------------------------------------
# 3. Main Model: CoarseToFineItemTower (Stage 1)
# ----------------------------------------------------------------------
class CoarseToFineItemTower(nn.Module):
    """
    [Item Tower]: Standard/Reinforced 피쳐를 융합하고 512차원 벡터 생성.
    vocab.py의 이중 어휘 구조와 호환되도록 수정되었습니다.
    """
    def __init__(self, embed_dim=EMBED_DIM_CAT, nhead=4, output_dim=OUTPUT_DIM_ITEM_TOWER):
        super().__init__()
        
        # 1. vocab.py에서 STD와 RE의 분리된 어휘 크기를 가져옵니다.
        std_vocab_size, re_vocab_size = vocab.get_vocab_sizes()
        
        # A. Dual Embedding (64d)
        # 단일 임베딩 대신, 분리된 어휘 크기를 사용합니다.
        self.std_embedding = nn.Embedding(std_vocab_size, embed_dim, padding_idx=vocab.PAD_ID)
        self.re_embedding = nn.Embedding(RE_MAX_CAPACITY, embed_dim, padding_idx=vocab.PAD_ID)
        # B. Self-Attention Encoders (d_model=64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.std_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.re_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # C. Cross-Attention (d_model=64, nhead=4)
        # 이 레이어는 Q=STD, K/V=RE로 사용될 것입니다.
        # (수정됨) Shape Vector (128d)가 제거되어 입력은 64d가 됨.
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # D. Deep Residual Head (입력 차원: embed_dim = 64)
        head_input_dim = embed_dim
        self.head = DeepResidualHead(input_dim=head_input_dim, output_dim=output_dim)

    def forward(self, std_input: torch.Tensor, re_input: torch.Tensor) -> torch.Tensor:
        # 1. 임베딩 (STD와 RE 분리 처리)
        std_embed = self.std_embedding(std_input)
        re_embed = self.re_embedding(re_input)
        
        # 2. Self-Attention Encoders
        std_output = self.std_encoder(std_embed) # Shape: (B, L_std, D)
        re_output = self.re_encoder(re_embed)   # Shape: (B, L_re, D)
        
        # 3. Cross-Attention (STD(Q)가 RE(K/V)를 참조)
        # Query: STD (우리가 더 중요하다고 가정하는 기본적인 상품 정보)
        # Key/Value: RE (선택적으로 보강할 세부 정보)
        
        # query, key, value 인자를 명시적으로 사용합니다.
        attn_output, _ = self.cross_attn(
            query=std_output,  
            key=re_output,     
            value=re_output,   
            need_weights=False
        )
        
        # 4. 잔차 연결(Residual Connection) 및 Layer Normalization
        # STD의 원본 정보에 RE로부터 추출된 강화 정보(attn_output)를 더합니다.
        fused_output = self.layer_norm(std_output + attn_output)
        
        # 5. 풀링 (Sequence -> Vector)
        # 최종적으로 Item 임베딩을 얻기 위해 평균 풀링을 수행합니다.
        # Shape: (B, D)

        
        pooled_output = fused_output.mean(dim=1) 

        ## 5. Shape Fusion Logic (제거됨)
        # v_fused = torch.cat([v_final, shape_vecs], dim=1) # 이 코드가 제거됨.
        
        # 6. Deep Residual Head
        # Deep Head Pass (I : 64 -> O : 128)
        final_vector = self.head(pooled_output)

        return final_vector
    

# ----------------------------------------------------------------------
# 4. OptimizedItemTower (Stage 2 Adapter - Triplet Training)
# ----------------------------------------------------------------------

class OptimizedItemTower(nn.Module):
    """
    [Optimization Tower]: Stage 1의 vector non-liner
    """
    def __init__(self, input_dim=OUTPUT_DIM_ITEM_TOWER, output_dim=OUTPUT_DIM_TRIPLET):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )
        
    def forward(self, x):
        # [Log 1] 입력 데이터 확인
        if not self.training: # 추론(eval) 모드일 때만 로그 출력 (학습 땐 너무 많음)
            print(f"\n  [Model Internal] Input Vector Shape: {x.shape}")
            print(f"  [Model Internal] Input Sample (First 5): {x[0, :5].detach().cpu().numpy()}")

        # 레이어 통과
        x = self.layer(x)
        
        # 정규화 (L2 Normalization)
        x = F.normalize(x, p=2, dim=1)
        
        # [Log 3] 정규화 확인 (Norm이 1.0에 가까운지)
        if not self.training:
            norm_check = torch.norm(x, p=2, dim=1).mean().item()
            print(f"  [Model Internal] Output Normalized Shape: {x.shape} | Avg Norm: {norm_check:.4f} (Expected ~1.0)")
            
        return x
    
    
# ----------------------------------------------------------------------
# 5. Dataset & Sampler & Training Function (Stage 2 Logic) / first INPUT from DB
# ----------------------------------------------------------------------


class SimCSEModelWrapper(nn.Module):
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder      # 이것이 CoarseToFineItemTower
        self.projector = projector  # 이것이 OptimizedItemTower


    def forward(self, t_std, t_re):
        # 1. 받은 2개 인자를 encoder에게 그대로 토스
        enc_out = self.encoder(t_std, t_re) 
        
        # 2. 그 결과를 projector에게 토스
        return self.projector(enc_out)


class SimCSERecSysDataset(Dataset):
    def __init__(self, products: List[TrainingItem], dropout_prob: float = 0.2):
        self.products = products
        self.dropout_prob = dropout_prob

    def __len__(self):
        return len(self.products)

    def input_feature_dropout(self, product: TrainingItem) -> TrainingItem:
        """
        [Augmentation Logic]
        JSON 구조("clothes", "reinforced_feature_value")에 맞춰
        랜덤하게 속성(Key-Value)을 제거합니다.
        """
        # 원본 데이터 보호를 위해 Deep Copy (매우 중요)
        aug_p = copy.deepcopy(product)
        
        feature_dict = aug_p.feature_data
        
        # 1. Standard Features (clothes) Dropout
   
        clothes_data = feature_dict.get("clothes")
        if clothes_data:
            keys = list(clothes_data.keys())
            for k in keys:
                if random.random() < self.dropout_prob:
                    del clothes_data[k]
        
        # 2. Reinforced Features Dropout
  
        re_data = feature_dict.get("reinforced_feature_value")
        if re_data:
            keys = list(re_data.keys())
            for k in keys:
                if random.random() < self.dropout_prob:
                    del re_data[k]
                    
        return aug_p
    def __getitem__(self, idx):
        raw_product = self.products[idx]
        
        # SimCSE: 같은 상품을 두 번 변형해서 (View1, View2) 생성
        view1 = self.input_feature_dropout(raw_product)
        view2 = self.input_feature_dropout(raw_product)
        
        return view1, view2

