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


# 필드 순서 정의 (Field Embedding), 임시, data 보고 결정
# key 순서 == 오는 json 데이타Load 순서
ALL_FIELD_KEYS = [
    "category", "season", "fiber_composition", "elasticity", "transparency", 
    "isfleece", "color", "gender", "category_specification", 
    # 필요한 만큼 추가...
]
FIELD_TO_IDX = {k: i for i, k in enumerate(ALL_FIELD_KEYS)}
NUM_TOTAL_FIELDS = len(ALL_FIELD_KEYS)


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
    [Item Tower - Residual Field Embedding Ver.]
    TabTransformer의 아이디어를 응용하여, STD와 RE를 하나의 시퀀스로 통합하고
    계층적 잔차 연결(Inheritance)을 통해 학습 안정성을 극대화한 구조.
    """
    def __init__(self, 
                 embed_dim=EMBED_DIM_CAT,     # 64
                 nhead=4, 
                 num_layers=2,                # TabTransformer는 얕아도 충분함
                 max_fields=50,               # 예상되는 최대 필드(컬럼) 개수
                 output_dim=OUTPUT_DIM_ITEM_TOWER):
        super().__init__()
        
        # 1. Vocab Size 가져오기
        std_vocab_size, _ = vocab.get_vocab_sizes()
        
        # 2. Embeddings
        # A. STD Value Embedding (Base) , RE Value Embedding (Delta / Child)
        self.std_embedding = nn.Embedding(std_vocab_size, embed_dim, padding_idx=vocab.PAD_ID)
        self.re_embedding = nn.Embedding(RE_MAX_CAPACITY, embed_dim, padding_idx=vocab.PAD_ID)
        
        # RE는 Delta(차이점)만 학습하므로 0 근처 초기화 (학습 초기 안정성)
        nn.init.normal_(self.re_embedding.weight, mean=0.0, std=0.01)

        # C. Field Embedding (Shared Key)
        # 각 컬럼(Color, Brand 등)의 역할을 나타내는 임베딩
        self.field_embedding = nn.Parameter(torch.randn(1, max_fields, embed_dim))
        
        # 3. Unified Transformer Encoder
        # STD와 RE가 한 공간에서 상호작용 (Cross-Attn 대신 Self-Attn 사용)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation='gelu'
            #,norm_first=True # 최신 트렌드 (안정적 수렴)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Projection Head
        # 입력 차원: (STD필드수 + RE필드수) * embed_dim -> Flatten 후 압축
        self.head = DeepResidualHead(input_dim=embed_dim, output_dim=output_dim) 
        


    def forward(self, std_input: torch.Tensor, re_input: torch.Tensor) -> torch.Tensor:
        """
        std_input: (Batch, Num_Fields) - 예: [Color_ID, Category_ID, ...]
        re_input:  (Batch, Num_Fields) - 예: [MatteBlack_ID, 0, ...] (순서가 STD와 대응되어야 함)
        """
        B, num_fields = std_input.shape
        
        # --- [Logic 1] Hierarchical Embedding Construction ---
        
        # (A) Field Embedding (Broadcasting)
        # 현재 배치의 필드 개수만큼 자름 (혹시 모를 가변 길이에 대비)
        field_emb = self.field_embedding[:, :num_fields, :] # (1, F, D)
        
        # (B) STD (Parent)
        std_val = self.std_embedding(std_input) # (B, F, D)
        std_token = std_val + field_emb
        
        # (C) RE (Child = Delta + Parent + Field)
        re_delta = self.re_embedding(re_input) # (B, F, D)
        
        # * 핵심: RE가 0(PAD)이어도 std_val + field_emb가 남아서 'Parent' 역할을 수행함
        # * detach(): RE의 그래디언트가 STD 임베딩을 망가뜨리지 않도록 차단
        re_token = re_delta + std_val.detach() + field_emb
        
        # --- [Logic 2] Unified Sequence ---
        # [STD_1, STD_2, ..., RE_1, RE_2, ...]
        combined_seq = torch.cat([std_token, re_token], dim=1) # (B, 2*F, D)
        
        # --- [Logic 3] Transformer & Pooling ---
        # PAD Masking: 여기서는 간단히 생략 (SimCLR 특성상 Noise도 정보가 됨)
        # 정교하게 하려면 src_key_padding_mask 추가 가능
        
        context_out = self.transformer(combined_seq) # (B, 2*F, D)
        
        # Mean Pooling (Flatten 대신 사용 -> 필드 수 변화에 강인함)
        pooled = context_out.mean(dim=1) # (B, D)
        
        return self.head(pooled) # (B, 128)
    
    
# ----------------------------------------------------------------------
# 4. OptimizedItemTower (Stage 2 Adapter - Triplet Training)
#    Projection Head --> Contrastive Loss(Opt.z) / Representation(Encoder)
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
    
        return F.normalize(x, p=2, dim=1)




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

    def _apply_dropout_and_convert(self, product: TrainingItem):
        """
        1. Feature Dropout 수행
        2. Dictionary -> Fixed Size Tensor 변환 (Hashing 포함)
        """
        # (1) Dropout Logic
        # 원본 데이터 보호 (Shallow copy of dict structure is enough usually, but deep for safety)
        feat_data = copy.deepcopy(product.feature_data)
        
        clothes = feat_data.get("clothes", {})
        reinforced = feat_data.get("reinforced_feature_value", {})
        
        # Random Dropout (Key 삭제)
        if self.dropout_prob > 0:
            for k in list(clothes.keys()):
                if random.random() < self.dropout_prob:
                    del clothes[k]
            for k in list(reinforced.keys()):
                if random.random() < self.dropout_prob:
                    del reinforced[k]

        # (2) Tensor Conversion Logic (Alignment)
        std_ids = []
        re_ids = []
        
        # 미리 정의된 ALL_FIELD_KEYS 순서대로 순회하며 ID 추출
        for key in ALL_FIELD_KEYS:
            # A. STD ID 추출
            std_val = clothes.get(key) # 없으면 None
            # None이면 MockVocab 내부에서 PAD_ID(0) 반환
            s_id = vocab.get_std_id(key, std_val) 
            std_ids.append(s_id)
            
            # B. RE ID 추출 (Hashing)
            # RE 데이터는 리스트 형태일 수 있음 (["Matte Black"]) -> 첫번째 값 사용
            re_val_list = reinforced.get(key)
            re_val = None
            if re_val_list and isinstance(re_val_list, list) and len(re_val_list) > 0:
                re_val = re_val_list[0]
            elif isinstance(re_val_list, str):
                re_val = re_val_list
            
            # Hashing 함수 호출 (저장 X)
            r_id = vocab.get_re_hash_id(re_val)
            re_ids.append(r_id)
            
        return torch.tensor(std_ids, dtype=torch.long), torch.tensor(re_ids, dtype=torch.long)

    def __getitem__(self, idx):
        item = self.products[idx]
        
        # Contrastive Learning을 위한 2개의 View 생성
        # 각각 서로 다른 Dropout이 적용됨
        v1_std, v1_re = self._apply_dropout_and_convert(item)
        v2_std, v2_re = self._apply_dropout_and_convert(item)
        
        return (v1_std, v1_re), (v2_std, v2_re)
''' 
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
'''


    
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