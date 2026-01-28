import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict, List, Tuple
from fastapi import APIRouter
from pydantic import BaseModel
from torch.utils.data import Dataset
import random
import utils.vocab as vocab
import copy
import random

# --- Global Configuration (전체 시스템이 참조하는 공통 차원) ---
EMBED_DIM_CAT = 64 # Feature의 임베딩 차원 (Transformer d_model)
OUTPUT_DIM_SIMCSE = 128 # Stage 2 최종 압축 차원
OUTPUT_DIM_ITEM_TOWER = 128 # Stage 1 최종 출력 차원 (Triplet Tower Input)
RE_MAX_CAPACITY = 500 # <<<<<<<<<<<< RE 토큰의 최대 개수를 미리 할당


class TrainingItem(BaseModel):

    product_id: int
    feature_data: Dict[str, Any]




# ----------------------------------------------------------------------
# 1. Utility Modules (SE-ResBlock & Deep Head)
#    - 이미 정의되어 있다고 가정하지만, 실행 가능성을 위해 포함합니다.
# ----------------------------------------------------------------------

class SEResidualBlock(nn.Module):
    """
    Squeeze-and-Excitation Residual Block
    입력 피처의 중요도를 스스로 재조정(Recalibration)하는 블록
    """
    def __init__(self, dim, dropout=0.2, expansion_factor=4):
        super().__init__()
        
        # Feature Transformation
        self.block = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.LayerNorm(dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.LayerNorm(dim),
        )
        
        # SE-Block (Attention)
        self.se_block = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        self.act = nn.GELU()
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        weight = self.se_block(out) # 중요도 계산
        out = out * weight          # 중요도 적용 (Scaling)
        return self.act(residual + out)

class DeepResidualHead(nn.Module):
    """
    Pyramid Funnel Architecture with SE-Blocks
    Fusion된 벡터를 받아 고차원 상호작용 후 압축
    """
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        
        mid_dim = input_dim * 2      
        hidden_dim = input_dim * 4   
        
        # 1. Expansion
        self.expand_layer1 = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.expand_layer2 = nn.Sequential(
            nn.Linear(mid_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 2. Deep Interaction (SE-Block)
        self.res_blocks = nn.Sequential(
            SEResidualBlock(hidden_dim, dropout=0.2),
            SEResidualBlock(hidden_dim, dropout=0.2)
        )
        
        # 3. Compression
        self.final_proj = nn.Linear(hidden_dim, output_dim)
        
        # 4. Global Skip Connection
        # 입력 차원과 출력 차원이 다를 수 있으므로 Linear로 차원 맞춤
        self.input_skip = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        m = self.expand_layer1(x)
        h = self.expand_layer2(m)
        h = self.res_blocks(h)
        main_out = self.final_proj(h)
        skip_out = self.input_skip(x)
        return main_out + skip_out

# ----------------------------------------------------------------------
# 2. Main Model: Hybrid Integrated Item Tower (Refactored)
# ----------------------------------------------------------------------

class HybridIntegratedItemTower(nn.Module):
    def __init__(self,
                 vocab,                 # vocab 객체 전달 필요
                 embed_dim=64,          # 모델 내부 임베딩 차원 (Structure)
                 nhead=4,
                 num_layers=2,
                 max_std_fields=20,     # STD 필드 개수
                 max_re_fields=20,      # RE 필드 개수
                 text_input_dim=384,    # S-BERT 출력 차원
                 output_dim=128):       # 최종 출력 차원
        super().__init__()

        # ======================================================
        # Part 1. Structure Encoding (STD + RE)
        # ======================================================
        std_vocab_size, _ = vocab.get_vocab_sizes()
        
        # Embeddings
        self.std_embedding = nn.Embedding(std_vocab_size, embed_dim, padding_idx=vocab.PAD_ID)
        self.re_token_embedding = nn.Embedding(vocab.RE_VOCAB_SIZE, embed_dim, padding_idx=vocab.RE_TOKENIZER.pad_token_id)

        # RE가 갖는 '변화량/특이사항' 정보가 초기에는 0(중립)에서 시작하도록 설정
        nn.init.normal_(self.re_token_embedding.weight, mean=0.0, std=0.01)
        # Field Embeddings
        self.std_field_emb = nn.Parameter(torch.randn(1, max_std_fields, embed_dim))
        self.re_field_emb = nn.Parameter(torch.randn(1, max_re_fields, embed_dim))

        # Transformer (Pre-LN Applied)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation='gelu',
            norm_first=True  # << [변경] Pre-LN 적용 (학습 안정성)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ======================================================
        # Part 2. Text Encoding (Adapter)
        # ======================================================
        # Text 벡터를 Structure 벡터와 비슷한 공간으로 투영
        self.text_adapter = nn.Sequential(
            nn.Linear(text_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # ======================================================
        # Part 3. Unified Head (Fusion & Projection)
        # ======================================================

        combined_dim = embed_dim + embed_dim  # Structure(64) + Text(64) = 128
        
        self.head = DeepResidualHead(input_dim=combined_dim, output_dim=output_dim)


    def forward(self, std_input, re_input, text_vector):
        """
        std_input: (Batch, N_std)
        re_input:  (Batch, N_re)
        text_vector: (Batch, 384)
        """
        B, num_std = std_input.shape
        _, num_re = re_input.shape

        # ---------------------------------------------------
        # A. Structure Vector Generation (Transformer)
        # ---------------------------------------------------
        # 1. Prepare Tokens
        std_val = self.std_embedding(std_input)
        std_f = self.std_field_emb[:, :num_std, :]
        std_tokens = std_val + std_f

        re_val = self.re_token_embedding(re_input)
        re_f = self.re_field_emb[:, :num_re, :]
        re_tokens = re_val + re_f

        # 2. Concat Sequence
        combined_seq = torch.cat([std_tokens, re_tokens], dim=1) # (B, N+M, D)

        # 3. Masking
        std_mask = (std_input != vocab.PAD_ID) # (B, N)
        re_mask = (re_input != vocab.RE_TOKENIZER.pad_token_id) # (B, M)
        full_mask = torch.cat([std_mask, re_mask], dim=1) # (B, N+M)

        # 4. Transformer Encoding
        # src_key_padding_mask는 True인 위치를 무시(masking)함 -> ~full_mask 전달
        context_out = self.transformer(combined_seq, src_key_padding_mask=~full_mask)

        # 5. Pooling (Mean)
        mask_expanded = full_mask.unsqueeze(-1).float()
        sum_emb = torch.sum(context_out * mask_expanded, dim=1)
        sum_cnt = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        v_struct = sum_emb / sum_cnt # (Batch, embed_dim)

        # ---------------------------------------------------
        # B. Text Vector Generation (Adapter)
        # ---------------------------------------------------
        v_text = self.text_adapter(text_vector) # (Batch, embed_dim)

        # ---------------------------------------------------
        # C. Unified Fusion (Concatenate & DNN)
        # ---------------------------------------------------
        # [변경] Gating 로직 삭제 -> 단순 Concat
        # 두 정보를 물리적으로 결합하여 DNN이 상호작용을 학습하게 함
        
        combined_features = torch.cat([v_struct, v_text], dim=1) # (B, embed_dim * 2)

        # ---------------------------------------------------
        # D. Final Output
        # ---------------------------------------------------
        # SE-Block이 포함된 Head가 중요도를 스스로 조절하며 최종 임베딩 생성
        output = self.head(combined_features) 

        return output
    
# ----------------------------------------------------------------------
# 3. OptimizedItemTower (Stage 2 Adapter - Triplet Training)
#    Projection Head --> Contrastive Loss(Opt.z) / Representation(Encoder)
# ----------------------------------------------------------------------

class OptimizedItemTower(nn.Module):
    """
    [Optimization Tower]: Projection Head, Distance/metric Learning용
    """
    def __init__(self, input_dim=OUTPUT_DIM_ITEM_TOWER, output_dim=OUTPUT_DIM_SIMCSE):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(), #nn.GELU(), 
            nn.Linear(input_dim, output_dim),
        )
        
    def forward(self, x):
        # [Log 1] 입력 데이터 확인
        if not self.training: 
            print(f"\n  [Model Internal] Input Vector Shape: {x.shape}")
            print(f"  [Model Internal] Input Sample (First 5): {x[0, :5].detach().cpu().numpy()}")

        # 레이어 통과
        x = self.layer(x)
        
        # 정규화 (L2 Normalization)
    
        return F.normalize(x, p=2, dim=1)


    
    
    
# ----------------------------------------------------------------------
# 4. Dataset & Sampler & Training Function (Stage 2 Logic) / first INPUT from DB
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
        """
        # 1. Deep Copy (원본 보존)
        feat_data = copy.deepcopy(product.feature_data)
        
        clothes = feat_data.get("clothes", {})
        reinforced = feat_data.get("reinforced_feature_value", {})
        
        # 2. Random Dropout (Key 삭제) - 여기가 데이터 증강(Augmentation) 핵심
        if self.dropout_prob > 0:
            # list(...)로 감싸야 삭제 중 딕셔너리 크기 변경 에러 방지
            for k in list(clothes.keys()):
                if random.random() < self.dropout_prob:
                    del clothes[k]
            for k in list(reinforced.keys()):
                if random.random() < self.dropout_prob:
                    del reinforced[k]

        
        # 3.preprocess_batch_input이 'feature_data' 속성을 참조하므로 그 형태를 맞춰줌.
        return TrainingItem(
            product_id=product.product_id,
            feature_data=feat_data # 드랍아웃 적용된 데이터
        )

    def __getitem__(self, idx):
        item = self.products[idx]
        
        # 뷰 1 생성 (드랍아웃 A 적용)
        view1_obj = self._apply_dropout_and_convert(item)
        
        # 뷰 2 생성 (드랍아웃 B 적용)
        view2_obj = self._apply_dropout_and_convert(item)
        
        return view1_obj, view2_obj
