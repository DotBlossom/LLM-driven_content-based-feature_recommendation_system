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

import utils.vocab as vocab
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
RE_MAX_CAPACITY = 500 # <<<<<<<<<<<< RE 토큰의 최대 개수를 미리 할당
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
    [Item Tower]: Standard/Reinforced 피쳐를 융합하고 128차원 벡터 생성.
    vocab.py의 이중 어휘 구조와 호환되도록 수정되었습니다.
    """
    def __init__(self, embed_dim=EMBED_DIM_CAT, nhead=4, output_dim=OUTPUT_DIM_ITEM_TOWER):
        super().__init__()
        
        # 1. vocab.py에서 STD와 RE의 분리된 어휘 크기 import (re_vocab은 나중에 fix하거나, 변경될떄 변수로)
        std_vocab_size, re_vocab_size = vocab.get_vocab_sizes()
        
        # A. Dual Embedding (64d)
        # 아마 Re_vocab에 데이터 좀 넣어놓자. 오류난다면?
        # 단일 임베딩 대신, 분리된 어휘 크기를 사용
        self.std_embedding = nn.Embedding(std_vocab_size, embed_dim, padding_idx=vocab.PAD_ID)
        self.re_embedding = nn.Embedding(RE_MAX_CAPACITY, embed_dim, padding_idx=vocab.PAD_ID)
        # B. Self-Attention Encoders (d_model=64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.std_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.re_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # [고려중] Title Embedding (Tokenizer의 Vocab Size 사용)
        '''
        self.title_vocab_size = TOKENIZER.vocab_size
        self.title_embedding = nn.Embedding(self.title_vocab_size, embed_dim, padding_idx=vocab.PAD_ID)
        self.title_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        '''

        
        # C. Cross-Attention (d_model=64, nhead=4)
        # 이 레이어는 Q=STD, K/V=RE로 사용
        # (수정됨) Shape Vector (128d)가 제거되어 입력은 64d가 됨.
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # D. Deep Residual Head (입력 차원: embed_dim = 64)
        head_input_dim = embed_dim
        self.head = DeepResidualHead(input_dim=head_input_dim, output_dim=output_dim)

    def forward(self, std_input: torch.Tensor, re_input: torch.Tensor ) -> torch.Tensor:
        # 1. 임베딩 (STD와 RE 분리 처리)
        std_embed = self.std_embedding(std_input)
        re_embed = self.re_embedding(re_input)
        
        # 2. Self-Attention Encoders
        std_output = self.std_encoder(std_embed) # Shape: (B, L_std, D)
        re_output = self.re_encoder(re_embed)   # Shape: (B, L_re, D)
        
        ### ---------------
        #    제목에 대한 LLM 기반 slicing이 된다면, 제목을 Re attention에 concat하여 진행
        #    학습데이터셋은 reinforced에 text_align 붙여놓고, 그거 여기애서 cross atten (eng)
        ### ---------------

        ''' 
        title_embed = self.title_embedding(title_input)
        title_output = self.title_encoder(title_embed)
        re_context_output = torch.cat([re_output, title_output], dim=1)
        re_mask = (re_input == vocab.PAD_ID)
        title_mask = (title_input == vocab.PAD_ID)
        combined_key_padding_mask = torch.cat([re_mask, title_mask], dim=1)
        
        '''
        
        ### 배치환경에서 no data re_input attn 배제 
        re_padding_mask = (re_input == vocab.PAD_ID)
        
        is_all_padding = re_padding_mask.all(dim=1)
        
        # 곱셈을 위한 게이트 생성 (정보가 없으면 0.0, 있으면 1.0)
        valid_gate = (~is_all_padding).float().view(-1, 1, 1)
        
    
    
    
        
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
        
        # re_input이 all zero 일 경우(batch 연산 특성 고려)
        attn_output = attn_output * valid_gate
        
        
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
            nn.GELU(), #nn.ReLU(), 
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




# x = F.normalize(x, p=2, dim=1) 실제 추론떄는 h쪽 model load하여 쓰자. (same d)

'''

구조: Encoder -> Embedding(h) -> MLP Layer(Projection Head) -> Output(z) -> Loss

원리: z 공간에서는 Contrastive Loss에 의해 데이터가 구체 표면으로 찌그러지며 정보 손실
반면 그 전 단계인 h는 데이터의 원본 정보를 상대적 보존

학습할 때: Projection Head를 붙여서 z 값으로 Loss 계산.

서빙할 때: Projection Head를 떼어버리고 h 값을 사용.

효과: 이렇게 하면 Representation Quality가 10~15% 향상 data

'''


    
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



    
# ----------------------------------------------------------------------
# 6. userTowerClass 
# ----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(
        self, 
        pretrained_item_matrix: torch.Tensor, # SimCSE로 학습된 아이템 벡터 (Freeze)
        token_vocab_size: int,                # Tokenizer Vocab Size (for Concept)
        output_dim=128,                       # 최종 출력 차원
        history_max_len=50,
        nhead=4,
        dropout=0.2
    ):
        super().__init__()
        
        # 임베딩 차원 자동 감지 (보통 64 or 128)
        self.embed_dim = pretrained_item_matrix.shape[1] 
        
        # -------------------------------------------------------
        # 1. Item History Encoder (Behavior)
        # -------------------------------------------------------
        self.item_embedding = nn.Embedding.from_pretrained(
            pretrained_item_matrix, 
            freeze=True, 
            padding_idx=0
        )
        self.pos_embedding = nn.Embedding(history_max_len, self.embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=nhead, batch_first=True)
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # -------------------------------------------------------
        # 2. Cart Concept Encoder (Context)
        # -------------------------------------------------------
        self.concept_embedding = nn.Embedding(token_vocab_size, self.embed_dim, padding_idx=0)
        self.concept_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # -------------------------------------------------------
        # 3. Physical Profile Encoder 
        # -------------------------------------------------------
        # 키, 몸무게 2개의 수치를 받아서 embed_dim 크기로 확장합니다.
        # Input: (B, 2) -> Output: (B, embed_dim)
        self.profile_projector = nn.Sequential(
            nn.Linear(2, self.embed_dim),       # 2개(키, 체중)를 벡터 공간으로 투영
            nn.BatchNorm1d(self.embed_dim),     # 수치 스케일 보정
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, self.embed_dim) # 한 번 더 정제
        )

        # -------------------------------------------------------
        # 4. Fusion & Output
        # -------------------------------------------------------
        # (History + Concept + Profile) = 3 * D
        fusion_input_dim = self.embed_dim * 3 
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_input_dim),
            nn.BatchNorm1d(fusion_input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_input_dim, output_dim) # 최종 차원 (Item Tower와 동일하게)
        )

    def forward(self, history_ids, concept_input, profile_features):
        """
        Args:
            history_ids (Tensor): (B, L_hist) - 아이템 ID 시퀀스
            concept_input (Tensor): (B, L_txt) - 장바구니 컨셉 텍스트 토큰
            profile_features (Tensor): (B, 2) - [Normalized Height, Normalized Weight]
                                      예: [[1.2, -0.5], [0.0, 0.8], ...] (Z-score 권장)
        """
        
        # --- A. Process History (Behavior) ---
        hist_embed = self.item_embedding(history_ids)
        B, L, _ = hist_embed.shape
        
        positions = torch.arange(L, device=history_ids.device).unsqueeze(0)
        hist_embed = hist_embed + self.pos_embedding(positions)
        
        src_key_padding_mask = (history_ids == 0)
        hist_output = self.history_encoder(hist_embed, src_key_padding_mask=src_key_padding_mask)
        
        # Mean Pooling
        mask_expanded = (~src_key_padding_mask).unsqueeze(-1).float()
        user_history_vec = (hist_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        
        # --- B. Process Cart Concept (Context) ---
        concept_embed = self.concept_embedding(concept_input)
        concept_mask = (concept_input == 0)
        
        concept_output = self.concept_encoder(concept_embed, src_key_padding_mask=concept_mask)
        
        # Mean Pooling
        c_mask_expanded = (~concept_mask).unsqueeze(-1).float()
        user_concept_vec = (concept_output * c_mask_expanded).sum(dim=1) / c_mask_expanded.sum(dim=1).clamp(min=1e-9)


        # --- C. Process Physical Profile (Demographics) [NEW!] ---
        # profile_features: (B, 2) -> (B, D)
        user_profile_vec = self.profile_projector(profile_features)


        # --- D. Final Fusion ---
        # 3가지 벡터를 Concat: (B, D) + (B, D) + (B, D) -> (B, 3D)
        combined = torch.cat([user_history_vec, user_concept_vec, user_profile_vec], dim=1)
        
        final_user_vector = self.fusion_mlp(combined)
        
        return final_user_vector