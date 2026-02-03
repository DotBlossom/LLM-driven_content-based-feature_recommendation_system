'''
from typing import Any, Dict, List, Tuple
from fastapi import APIRouter
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import utils.vocab as vocab
from database import ProductInferenceVectors
from sqlalchemy.orm import Session
import copy
import random

#################### !!!!!!!! 카테고리 제외하고 학습하기

def prepare_training_data(raw_json):
    features = raw_json['feature_data']['clothes']
    
    # ✂️ 학습용 텍스트 만들 때는 카테고리 삭제!
    if 'category' in features:
        del features['category'] 
        
    # 남은 건: "color: black, material: wool..." (순수 특징들)
    return str(features)


# ItemTowerEmbedding(S1) * N -> save..DB -> stage2 (optimizer pass -> triplet)  

model_router = APIRouter()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 필드 순서 정의 (Field Embedding), 임시, data 보고 결정
# key 순서 == 오는 json 데이타Load 순서
ALL_FIELD_KEYS = vocab.ORDERED_FEATURE_KEYS 
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
class SEResidualBlock(nn.Module):

    def __init__(self, dim, dropout=0.2, expansion_factor=4):
        super().__init__()
        
        # 1. Feature Transformation (기존과 동일하되, SwiGLU 스타일로 확장 제안)
        # 여기서는 안정적인 기존 Linear 방식을 유지하되 SE를 추가함
        self.block = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor), # 내부 확장
            nn.LayerNorm(dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim), # 다시 압축
            nn.LayerNorm(dim),
        )
        
        # 2. SE-Block (Channel Attention, SE-Net 구조 반영, Gating=Relu 파트)
        # 입력 벡터의 각 차원(feature)에 대해 중요도(0~1)를 계산
        self.se_block = nn.Sequential(
            nn.Linear(dim, dim // 4),  # Squeeze (정보 압축)
            nn.ReLU(),
            nn.Linear(dim // 4, dim),  # Excitation (중요도 복원)
            nn.Sigmoid()               # 0~1 사이의 가중치로 변환
        )

        self.act = nn.GELU()
    
    def forward(self, x):
        residual = x
        
        # (A) Main Path
        out = self.block(x)
        
        # (B) SE-Attention Path
        # 벡터의 글로벌 정보를 보고, 어떤 차원을 강조할지 결정
        # MLP 출력값(out)에 중요도(weight)를 곱함
        weight = self.se_block(out)
        out = out * weight 
        
        # (C) Residual Connection
        return self.act(residual + out)




# --- Deep Residual Head (Pyramid Funnel) ---
class DeepResidualHead(nn.Module):
    """
    [Architecture]
    Input(64) -> [Expand 2x] -> 128 -> [Expand 2x] -> 256 
    -> [Deep Interaction (SE-ResBlock)] -> 256 
    -> [Compression] -> Output(128)
    + Global Skip Connection
    """
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        
        # 차원 정의 (64 -> 128 -> 256)
        mid_dim = input_dim * 2      # 128
        hidden_dim = input_dim * 4   # 256
        
        # 1. Progressive Expansion 
        self.expand_layer1 = nn.Sequential(
            nn.Linear(input_dim, mid_dim),  # 64 -> 128
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.expand_layer2 = nn.Sequential(
            nn.Linear(mid_dim, hidden_dim), # 128 -> 256
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 2. Deep Interaction (Peak Dimension에서 수행)
        # 가장 차원이 높은 256 상태에서 SE-Block으로 정밀한 특징 추출 수행
        self.res_blocks = nn.Sequential(
            SEResidualBlock(hidden_dim, dropout=0.2), # 256 유지
            SEResidualBlock(hidden_dim, dropout=0.2)  # 256 유지
        )
        
        # 3. Final Projection (Compression)
        # 256 -> 128 로 압축하여 최종 임베딩 생성
        self.final_proj = nn.Linear(hidden_dim, output_dim)
        
        # 4. Global Skip Connection (Input Shortcut) ResNet 잔차
        self.input_skip = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # --- [Step 1] Progressive Expansion ---
        m = self.expand_layer1(x)  # 64 -> 128
        h = self.expand_layer2(m)  # 128 -> 256
        
        # --- [Step 2] Feature Interaction (SE-Attention) ---
        h = self.res_blocks(h)     # 256 -> 256
        
        # --- [Step 3] Compression ---
        main_out = self.final_proj(h) # 256 -> 128
        
        # --- [Step 4] Global Shortcut ---
        skip_out = self.input_skip(x) # 64 -> 128
        
        return main_out + skip_out
    
    
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
        # self.re_embedding = nn.Embedding(RE_MAX_CAPACITY, embed_dim, padding_idx=vocab.PAD_ID)
        self.re_token_embedding = nn.Embedding(vocab.RE_VOCAB_SIZE, embed_dim, padding_idx=vocab.RE_TOKENIZER.pad_token_id)
        # RE는 Delta(차이점)만 학습하므로 0 근처 초기화 (학습 초기 안정성)
        nn.init.normal_(self.re_token_embedding.weight, mean=0.0, std=0.01)

        # C. Field Embedding (Shared Key)
        # 각 컬럼(Color, Brand 등)의 역할을 나타내는 임베딩
        self.field_embedding = nn.Parameter(torch.randn(1, max_fields, embed_dim))
        
        # 3. Unified Transformer Encoder
        # STD와 RE가 한 공간에서 상호작용 (Self-Attn 사용)
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
        
    # $$Input Sequence = [\underbrace{Token_A}_{STD자리}, \underbrace{Token_B}_{RE(잔차)자리}]$$

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
        
        re_tokens = self.re_token_embedding(re_input)
        re_mask = (re_input != vocab.RE_TOKENIZER.pad_token_id).float().unsqueeze(-1) # (B, F, S, 1)
        
        # 3. Sum / Count (유효 토큰 개수로 나누기)
        sum_re = torch.sum(re_tokens * re_mask, dim=2) # (B, F, D)
        count_re = torch.clamp(re_mask.sum(dim=2), min=1e-9) # (B, F, 1)
        
    
        # (C) RE (Child = Delta + Parent + Field)
        re_delta = sum_re / count_re # (B, F, D) -> 하나의 벡터로 압축됨!
        
    
        # re_delta = self.re_embedding(re_input) # (B, F, D)
        
        # * 핵심: RE가 0(PAD)이어도 std_val + field_emb가 남아서 'Parent' 역할을 수행함
        # * detach(): RE의 그래디언트가 STD 임베딩을 망가뜨리지 않도록 차단
        re_token = re_delta + std_val.detach() + field_emb
        
        # --- [Logic 2] Unified Sequence ---
        combined_seq = torch.cat([std_token, re_token], dim=1)
        
        # Mask 생성
        # 1. STD, RE가 유효한가?
        std_valid = (std_input != vocab.PAD_ID)
        re_valid = (re_input != vocab.RE_TOKENIZER.pad_token_id).any(dim=2)

        mask_part_std = std_valid
        mask_part_re = re_valid | std_valid

        full_mask = torch.cat([mask_part_std, mask_part_re], dim=1) # (B, 2*F)

        # 이후 Transformer에 전달
        context_out = self.transformer(combined_seq, src_key_padding_mask=~full_mask)

        # [Smart Pooling] 패딩 제외 평균
        mask_expanded = full_mask.unsqueeze(-1).float() # (B, 2*F, 1)
        sum_embeddings = torch.sum(context_out * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9) # 0으로 나누기 방지
        
        pooled = sum_embeddings / sum_mask
        
        return self.head(pooled)
    
# ----------------------------------------------------------------------
# 4. OptimizedItemTower (Stage 2 Adapter - Triplet Training)
#    Projection Head --> Contrastive Loss(Opt.z) / Representation(Encoder)
# ----------------------------------------------------------------------

class OptimizedItemTower(nn.Module):
    """
    [Optimization Tower]: Projection Head, Distance/metric Learning용
    """
    def __init__(self, input_dim=OUTPUT_DIM_ITEM_TOWER, output_dim=OUTPUT_DIM_TRIPLET):
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




# x = F.normalize(x, p=2, dim=1) 실제 추론떄는 h쪽 model load하여 쓰자. (same d)



    
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


# ----------------------------------------------------------------------
# 6. userTowerClass
#     
# ----------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# [참고] 님께서 작성하신 Utility Modules (SEResidualBlock, DeepResidualHead)가
# 이미 import 되어 있다고 가정합니다.

class LightGCNEncoder(nn.Module):
    """
    User-Item 그래프 구조를 학습하여 '장기적인 선호도(Long-term Preference)' 추출
    """
    def __init__(self, num_users, num_items, embed_dim, n_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.n_layers = n_layers
        
        # 초기 임베딩 (User + Item)
        # 0 ~ num_users-1: User
        # num_users ~ num_users+num_items-1: Item
        self.embedding = nn.Embedding(num_users + num_items, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, adj_matrix):
        """
        adj_matrix: Normalized Sparse Adjacency Matrix
        """
        all_emb = self.embedding.weight
        embs = [all_emb]

        # Graph Convolution (No Nonlinearity)
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embs.append(all_emb)
        
        # Layer Combination (Mean or Weighted Sum)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        # User 부분과 Item 부분을 분리해서 반환
        user_emb_final, item_emb_final = torch.split(light_out, [self.num_users, self.num_items])
        return user_emb_final, item_emb_final


class SequentialEncoder(nn.Module):
    """
    최근 행동 이력을 통해 '단기적인 의도(Short-term Intent)' 추출 (SASRec Style)
    """
    def __init__(self, num_items, embed_dim, max_len=50, num_blocks=2, num_heads=2):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.emb_dropout = nn.Dropout(0.1)
        
        # SASRec Block (Standard Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*4, 
            dropout=0.1, 
            activation='gelu',
            batch_first=True,
            norm_first=True # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        
        self.max_len = max_len

    def forward(self, history_ids):
        """
        history_ids: (Batch, Max_Len) - Padding은 0
        """
        seq_len = history_ids.size(1)
        
        # 1. Embedding + Positional
        # positions: [0, 1, 2, ... seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.long, device=history_ids.device)
        positions = positions.unsqueeze(0).expand(history_ids.size(0), -1)
        
        x = self.item_emb(history_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # 2. Causality Mask (미래 정보 참조 방지)
        # mask: (Seq_Len, Seq_Len) Upper triangular is -inf
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(history_ids.device)

        # 3. Padding Mask
        # key_padding_mask: (Batch, Seq_Len) True where value is 0
        pad_mask = (history_ids == 0)

        # 4. Transformer Passing
        # x: (Batch, Seq_Len, Dim)
        output = self.transformer(x, mask=mask, src_key_padding_mask=pad_mask)

        # 5. Gather Last Valid Item (시퀀스의 마지막 시점이 현재 의도)
        # (Batch, Dim) 추출 로직은 생략하고, 편의상 마지막 컬럼 혹은 
        # 실제 데이터 길이에 맞춰 gather 했다고 가정. 여기서는 간단히 마지막 토큰 사용.
        # 실제 구현시: last_indices = (history_ids != 0).sum(1) - 1 등을 사용
        
        # 여기서는 간단히 Mean Pooling 혹은 Last Token 사용 (성능은 Last가 보통 좋음)
        # 패딩이 아닌 토큰 중 마지막 벡터를 가져오는 로직:
        # (구현 편의상 Global Pooling 예시)
        mask_expanded = (~pad_mask).unsqueeze(-1).float()
        sum_emb = torch.sum(output * mask_expanded, dim=1)
        sum_cnt = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        seq_vector = sum_emb / sum_cnt

        return seq_vector


class HybridGNNUserTower(nn.Module):
    def __init__(self, 
                 num_users, 
                 num_items, 
                 embed_dim=64, 
                 output_dim=128, # Item Tower output_dim과 맞춰야 함
                 max_history_len=50):
        super().__init__()
        
        # 1. GNN Component (Long-term)
        self.gnn_encoder = LightGCNEncoder(num_users, num_items, embed_dim)
        
        # 2. Sequential Component (Short-term)
        self.seq_encoder = SequentialEncoder(num_items, embed_dim, max_len=max_history_len)
        
        # 3. Fusion Head (User's DeepResidualHead)
        # 입력: GNN(64) + Seq(64) = 128 -> 출력: 128
        self.fusion_head = DeepResidualHead(input_dim=embed_dim * 2, output_dim=output_dim)

    def forward(self, user_ids, history_ids, adj_matrix):
        """
        user_ids: (Batch,)
        history_ids: (Batch, Max_Len)
        adj_matrix: Global Sparse Adjacency Matrix (User-Item Graph)
        """
        
        # --- A. GNN Flow (Global Preference) ---
        # 전체 그래프를 통해 업데이트된 모든 유저 벡터를 가져옴
        # (Inference 시에는 미리 계산된 벡터를 Lookup만 하면 됨)
        all_user_embs, _ = self.gnn_encoder(adj_matrix) 
        user_long_term = all_user_embs[user_ids] # (Batch, embed_dim)

        # --- B. Sequential Flow (Current Intent) ---
        user_short_term = self.seq_encoder(history_ids) # (Batch, embed_dim)

        # --- C. Contrastive Learning Prep (Optional) ---
        # 학습 시, user_long_term과 user_short_term이 서로 "비슷한 정보를 담도록" 
        # Auxiliary Loss를 추가하면 성능이 올라감. (Self-Supervised Learning)
        
        # --- D. Fusion ---
        combined = torch.cat([user_long_term, user_short_term], dim=1) # (Batch, 128)
        
        # Item Tower의 DeepHead와 동일한 구조를 통과하여 최종 매핑
        final_user_vector = self.fusion_head(combined) # (Batch, 128)
        
        return final_user_vector

def load_pretrained_vectors_from_db(db_session: Session) -> Tuple[torch.Tensor, Dict[int, int]]:
    """
    [기능]
    1. DB에서 (product_id, vector) 쌍을 모두 가져옵니다.
    2. DB ID -> Model Index 매핑을 생성합니다.
    3. User Tower의 Embedding Layer에 넣을 Weight Matrix를 만듭니다.
    
    [Return]
    - embedding_matrix: (Num_Products + 1, 128) - 0번은 Padding
    - id_map: {real_db_id: model_index}
    """
    print("⏳ Fetching product vectors from DB...")
    
    # 1. DB Query (ID와 Serving Vector만 가져옴)
    # vector_serving이 우리가 사용할 최종 아이템 벡터라고 가정
    results = db_session.query(
        ProductInferenceVectors.id, 
        ProductInferenceVectors.vector_embedding
    ).filter(
        ProductInferenceVectors.vector_embedding.isnot(None)
    ).all()
    
    if not results:
        raise ValueError("DB에 저장된 아이템 벡터가 없습니다!")

    # 2. 메타데이터 설정
    num_products = len(results)
    vector_dim = 128 # 고정 차원
    
    # 0번 인덱스는 Padding을 위해 비워둠 (Index 1부터 시작)
    embedding_matrix = torch.zeros((num_products + 1, vector_dim), dtype=torch.float32)
    id_map = {} # Real ID -> Model Index

    # 3. 매핑 및 매트릭스 채우기
    print(f"📦 Processing {num_products} items...")
    
    for idx, (real_id, vector_list) in enumerate(results, start=1):
        # vector_list는 DB에서 List[float] 형태로 온다고 가정
        if vector_list is None: continue
            
        # 매핑 저장
        id_map[real_id] = idx 
        
        # 텐서에 값 할당
        embedding_matrix[idx] = torch.tensor(vector_list, dtype=torch.float32)
        
    print("✅ Pretrained Embedding Matrix Created.")
    print(f"   Shape: {embedding_matrix.shape}")
    
    return embedding_matrix, id_map


class FinalUserTower(nn.Module):
    def __init__(self, 
                 num_total_products: int,
                 pretrained_item_matrix: torch.Tensor = None,
                 max_seq_len: int = 50,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 output_dim: int = 128):
        super().__init__()
        
        # 1. Embeddings
        # num_total_products + 1 (Padding=0)
        self.item_embedding = nn.Embedding(num_total_products + 1, d_model, padding_idx=0)
        
        # [중요] Pretrained Weight 로드 (User History 입력용)
        if pretrained_item_matrix is not None:
            self.load_pretrained_weights(pretrained_item_matrix)
            
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.season_embedding = nn.Embedding(5, d_model, padding_idx=0) # 0:Pad, 1~4:Seasons
        self.gender_embedding = nn.Embedding(3, d_model, padding_idx=0) # 0:Unk, 1:M, 2:F
        
        # 2. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.user_query_token = nn.Parameter(torch.randn(1, 1, d_model)) 

        # 3. Heads
        self.deep_head = DeepResidualHead(input_dim=d_model, output_dim=output_dim)
        self.final_projector = OptimizedItemTower(input_dim=output_dim, output_dim=output_dim)

    def load_pretrained_weights(self, matrix):
        # Matrix Shape Check: (Num_Items + 1, Dim)
        expected_shape = self.item_embedding.weight.shape
        if matrix.shape != expected_shape:
            print(f"⚠️ Shape Mismatch! Model: {expected_shape}, Input: {matrix.shape}")
            # Shape이 다르면 일부만 로드하거나 에러 처리 (여기서는 단순 경고 후 슬라이싱)
            min_rows = min(expected_shape[0], matrix.shape[0])
            self.item_embedding.weight.data[:min_rows] = matrix[:min_rows]
        else:
            self.item_embedding.weight.data.copy_(matrix)
        
        # Fine-tuning 여부 (True: 학습됨, False: 고정됨)
        self.item_embedding.weight.requires_grad = False 

    def forward(self, history_ids, season_idx, gender_idx):
        """
        history_ids: (Batch, Seq_Len)
        season_idx: (Batch, )
        gender_idx: (Batch, )
        """
        B, L = history_ids.shape
        device = history_ids.device
        
        # [Step 1] Input Embedding & Context Injection
        seq_emb = self.item_embedding(history_ids) # (B, L, D)
        pos_emb = self.position_embedding(torch.arange(L, device=device)) # (L, D) -> Broadcasting
        
        # Global Context (Season, Gender)를 시퀀스 전체에 더함
        season_emb = self.season_embedding(season_idx).unsqueeze(1) # (B, 1, D)
        gender_emb = self.gender_embedding(gender_idx).unsqueeze(1) # (B, 1, D)
        
        x = seq_emb + pos_emb + season_emb + gender_emb # (B, L, D)
        
        # [Step 2] Append CLS Token & Masking
        cls_token = self.user_query_token.expand(B, -1, -1) # (B, 1, D)
        x = torch.cat([cls_token, x], dim=1) # (B, L+1, D)
        
        # Padding Mask (True가 마스킹됨)
        # 0은 Padding ID
        padding_mask = (history_ids == 0) # (B, L)
        cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=device) # CLS 토큰은 마스킹 안함
        full_mask = torch.cat([cls_mask, padding_mask], dim=1) # (B, L+1)
        
        # [Step 3] Transformer Encoding
        out = self.transformer(x, src_key_padding_mask=full_mask)
        
        # [Step 4] Extract CLS & Projection
        raw_user_vector = out[:, 0, :] # (B, d_model) -> 0번째 토큰(CLS)만 사용
        deep_feat = self.deep_head(raw_user_vector) 
        final_vector = self.final_projector(deep_feat)
        
        return final_vector
        
        '''