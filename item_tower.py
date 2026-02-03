from sqlalchemy import select
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup, AutoModel
from pydantic import BaseModel
from typing import Any, Dict, List
import copy
import random
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from database import ProductInferenceInput
from utils import vocab

import os


# --- Global Configuration ---
EMBED_DIM = 128
OUTPUT_DIM_ENCODER = 128       # Encoder(Representation) 출력
OUTPUT_DIM_PROJECTOR = 128     # Projector(SimCSE Loss용) 출력
FASHION_BERT_MODEL = "bert-base-uncased" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAD_ID = vocab.PAD_ID
UNK_ID = vocab.UNK_ID

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# 1. Models: Encoder + Projector + Wrapper
# ----------------------------------------------------------------------

# (A) Encoder: HybridItemTower 
class SEResidualBlock(nn.Module):
    """ Squeeze-and-Excitation Residual Block """
    def __init__(self, dim, dropout=0.2, expansion_factor=4):
        super().__init__()
        
        
        # 1. Feature Transformation (SwiGLU 스타일로?)
        self.block = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.LayerNorm(dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.LayerNorm(dim),
        )        
        # 2. SE-Block (Channel Attention, SE-Net 구조 반영, Gating=Relu 파트)
        # 입력 벡터의 각 차원(feature)에 대해 중요도(0~1)를 계산
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
        
        # (B) SE-Attention Path
        # MLP 출력값(out)에 중요도(weight)를 곱함
        weight = self.se_block(out)
        out = out * weight
        
        return self.act(residual + out)

class DeepResidualHead(nn.Module):

    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        
        # 차원 정의 
        mid_dim = input_dim * 2      # 256
        hidden_dim = input_dim * 4   # 512
        
        # 1. Progressive Expansion 
        self.expand_layer1 = nn.Sequential(
            nn.Linear(input_dim, mid_dim),  # step 1 128 -> 256
            nn.LayerNorm(mid_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.expand_layer2 = nn.Sequential(
            nn.Linear(mid_dim, hidden_dim), # step 2 256 -> 512
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 2. Deep Interaction 
        self.res_blocks = nn.Sequential(
            SEResidualBlock(hidden_dim, dropout=0.2), 
            SEResidualBlock(hidden_dim, dropout=0.2)  
        )
        
        # 3. Final Projection (Compression)s

        self.final_proj = nn.Linear(hidden_dim, output_dim)
        
        # 4. Global Skip Connection (Input Shortcut) ResNet 잔차
        self.input_skip = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # --- [Step 1] Progressive Expansion ---
        m = self.expand_layer1(x) 
        h = self.expand_layer2(m)  
        
        # --- [Step 2] Feature Interaction (SE-Attention) ---
        h = self.res_blocks(h)     
        
        # --- [Step 3] Compression ---
        main_out = self.final_proj(h) 
        
        # --- [Step 4] Global Shortcut ---
        skip_out = self.input_skip(x) 
        
        return main_out + skip_out
    

class HybridItemTower(nn.Module):
    def __init__(self,
                 std_vocab_size: int,
                 num_std_fields: int,
                 embed_dim: int = 128, 
                 output_dim: int = 128):
        super().__init__()

        # A. STD Encoder
        self.std_embedding = nn.Embedding(std_vocab_size, embed_dim, padding_idx=PAD_ID)
        self.std_field_emb = nn.Parameter(torch.randn(1, num_std_fields, embed_dim))

        # B. Fashion-BERT Setup
        print(f"Loading {FASHION_BERT_MODEL} ...")
        self.bert_config = AutoConfig.from_pretrained(FASHION_BERT_MODEL)
        self.bert_model = AutoModel.from_pretrained(FASHION_BERT_MODEL)
        bert_dim = self.bert_config.hidden_size 

        # C. RE Encoder (Embeddings from BERT -> Project)
        self.re_proj = nn.Sequential(
            nn.Linear(bert_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.re_field_position = nn.Parameter(torch.randn(1, 9, embed_dim))

        # D. Text Encoder (Product Name)
        self.text_proj = nn.Sequential(
            nn.Linear(bert_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # E. Fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False
            )
        self.head = DeepResidualHead(input_dim=embed_dim, output_dim=output_dim)

        self._debug_logged = False
    def _debug_log(self, stage: int, title: str, tensors: Dict[str, torch.Tensor]):
        """
        [내부 함수] 스테이지별로 텐서 정보를 깔끔하게 출력합니다.
        """
        if self._debug_logged:
            return

        # 헤더 출력 (Stage 0일 때)
        if stage == 0:
            print("\n" + "="*60)
            print(f"🧩 [HybridItemTower] Forward Flow Debugging Start")
            print("="*60)

        # 스테이지 타이틀
        print(f"🔹 [Stage {stage}] {title}")

        # 텐서 정보 분석 및 출력
        if tensors:
            for name, tensor in tensors.items():
                if isinstance(tensor, torch.Tensor):
                    shape_str = str(tuple(tensor.shape))
                    
                    # 값이 실수형이면 통계(평균, 표준편차)도 출력
                    if tensor.dtype in [torch.float32, torch.float16, torch.float64]:
                        mean_val = tensor.mean().item()
                        std_val = tensor.std().item()
                        info = f"Shape: {shape_str} | Mean: {mean_val:.4f} | Std: {std_val:.4f}"
                    else:
                        info = f"Shape: {shape_str} (Type: {tensor.dtype})"
                        
                    print(f"   - {name:<15}: {info}")
                else:
                    print(f"   - {name:<15}: {tensor} (Not a Tensor)")
        
        print("-" * 40)

        # 종료 처리 (Stage 99일 때)
        if stage == 99:
            print("✅ Debugging Log Finished.")
            print("="*60 + "\n")
            self._debug_logged = True
            
    def forward(self, 
                std_input: torch.Tensor,       
                re_input_ids: torch.Tensor,    
                re_attn_mask: torch.Tensor,    
                text_input_ids: torch.Tensor,  
                text_attn_mask: torch.Tensor): 
        
        B = std_input.shape[0]


        # 1. STD
        std_emb = self.std_embedding(std_input) 
        std_emb = std_emb + self.std_field_emb  

        self._debug_log(1, "STD Embedding", {"std_emb": std_emb})


        # 2. RE (Using Fashion-BERT Word Embeddings Only)
        flat_re_ids = re_input_ids.view(-1, re_input_ids.size(-1))
        with torch.no_grad(): 
             word_embs = self.bert_model.embeddings(input_ids=flat_re_ids) 
        
        re_feats = self.re_proj(word_embs) 
        
        # 패딩 제외 Pooling
        flat_mask = re_attn_mask.view(-1, re_attn_mask.size(-1)).unsqueeze(-1) 
        sum_re = torch.sum(re_feats * flat_mask, dim=1)
        count_re = torch.clamp(flat_mask.sum(dim=1), min=1e-9)
        re_vectors = sum_re / count_re 
        
        re_vectors = re_vectors.view(B, 9, -1) 
        re_vectors = re_vectors + self.re_field_position
        
        # [Log] Stage 2: RE Encoding
        self._debug_log(2, "RE Process", {
            "BERT Word Emb": word_embs,
            "Pooled RE Vec": re_vectors
        })

        # 3. Text (Product Name) -> Full BERT Context
        bert_out = self.bert_model(input_ids=text_input_ids, attention_mask=text_attn_mask)
        cls_token = bert_out.last_hidden_state[:, 0, :] # [CLS]
        text_vec = self.text_proj(cls_token).unsqueeze(1) 

        self._debug_log(3, "Text Encoder", {"CLS Vector": text_vec})
        
        # 4. Fusion
    
        combined_seq = torch.cat([std_emb, re_vectors, text_vec], dim=1) 
        self._debug_log(4, "Fusion Prep", {"Combined Seq": combined_seq})
        
        context_out = self.transformer(combined_seq) 
        final_vec = context_out.mean(dim=1) 
        out = self.head(final_vec)
        self._debug_log(99, "Final Projection", {"Output": out})
        
        return out
    
# (B) Projector: OptimizedItemTower for SimCSE
class OptimizedItemTower(nn.Module):
    """
    [Optimization Tower]: Projection Head for Contrastive Learning
    Representation(128) -> Hidden(128) -> Output(128) -> Normalize
    """
    def __init__(self, input_dim=OUTPUT_DIM_ENCODER, output_dim=OUTPUT_DIM_PROJECTOR):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )
        
    def forward(self, x):
        x = self.layer(x)
        return F.normalize(x, p=2, dim=1) # L2 Normalization for Cosine Similarity

# (C) Wrapper: SimCSEModelWrapper (Input Arguments 확장)
class SimCSEModelWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, projector: nn.Module):
        super().__init__()
        self.encoder = encoder      
        self.projector = projector  

    def forward(self, std, re_ids, re_mask, txt_ids, txt_mask):
        # 1. Encoder (Representation)
        # 5개의 인자를 Encoder에 전달
        enc_out = self.encoder(std, re_ids, re_mask, txt_ids, txt_mask)
        
        # 2. Projector (SimCSE Space)
        proj_out = self.projector(enc_out)
        
        return proj_out

# ----------------------------------------------------------------------
# 2. Data Structures & Dataset (Augmentation Logic)
# ----------------------------------------------------------------------

class TrainingItem(BaseModel):
    product_id: str
    feature_data: Dict[str, Any] # DB에서 긁어온 Raw JSON
    product_name: str            # Text Embedding용

class SimCSERecSysDataset(Dataset):
    def __init__(self, products: List[TrainingItem], dropout_prob: float):
        self.products = products
        self.dropout_prob = dropout_prob
        
        # Dropout 대상이 되는 Key 그룹 정의
        self.std_keys = vocab.get_std_field_keys() # ["product_type_name", ...]
        self.re_keys = vocab.RE_FEATURE_KEYS       # ["[CAT]", "[MAT]", ...]

    def __len__(self):
        return len(self.products)
    
    def _corrupt_data(self, item: TrainingItem) -> TrainingItem:
        # 원본 데이터 복사
        new_feature_data = copy.deepcopy(item.feature_data)
        new_name = item.product_name
        

        # 확률 설정
        KEY_DROP_PROB = self.dropout_prob - 0.1    # 키 자체를 날릴 확률 (통째로 삭제)
        VALUE_DROP_PROB = self.dropout_prob  # 리스트 내부의 값을 하나씩 날릴 확률 (부분 삭제)
        
        all_keys = list(new_feature_data.keys())
        
        for k in all_keys:
            val = new_feature_data[k]
            
            # (A) 값이 리스트인 경우 (예: [MAT]: ['Cotton', 'Poly'])
            if isinstance(val, list):
                # 1. 먼저 값을 솎아냄 (Value-level)
                # 살아남은 애들만 필터링
                surviving_values = [v for v in val if random.random() > VALUE_DROP_PROB]
                
                # 2. 만약 다 지워져서 빈 리스트가 되면 -> 키 자체를 삭제
                if not surviving_values:
                    del new_feature_data[k]
                else:
                    new_feature_data[k] = surviving_values
                    
            # (B) 값이 단일 값(문자열 등)인 경우 (예: product_type_name)
            else:
                # 그냥 키 자체를 날림 (Key-level)
                if random.random() < KEY_DROP_PROB:
                    del new_feature_data[k]

        # =======================================================
        # 2. Text Deletion (단어 구멍 뚫기)
        # =======================================================
        if new_name:
            words = new_name.split()
            # 단어가 2개 이상이면 하나를 삭제 (난이도 조절)
            if len(words) > 1: 
                if random.random() < 0.5: 
                    drop_idx = random.randint(0, len(words)-1)
                    del words[drop_idx]
                    new_name = " ".join(words)
            # 단어가 1개뿐이면 가끔 아예 삭제
            elif len(words) == 1:
                if random.random() < 0.1:
                    new_name = ""

        return TrainingItem(
            product_id=item.product_id,
            feature_data=new_feature_data,
            product_name=new_name
        )

    def _apply_dropout(self, item: TrainingItem) -> TrainingItem:
        """
        Feature Dropout 수행: 딕셔너리에서 Key를 확률적으로 제거.
        제거된 Key는 나중에 Collate 단계에서 vocab.get_std_id 호출 시 
        값이 없으므로 자동으로 PAD_ID 또는 UNK_ID가 됨.
        """
        if self.dropout_prob <= 0:
            return item

        # Deep Copy to preserve original
        new_feature_data = copy.deepcopy(item.feature_data)
        new_name = item.product_name
        # 1. STD & RE Keys Dropout
        # feature_data 안에 flattened된 형태로 있다고 가정 
        all_keys = list(new_feature_data.keys())
        
        for k in all_keys:
            # 주요 Feature Key인 경우에만 드랍아웃 시도
            if (k in self.std_keys or k in self.re_keys):
                if random.random() < self.dropout_prob:
                    del new_feature_data[k]
        
        # Text Dropout (Optional): 이름 자체를 지울지 말지 결정. 
        TEXT_DROPOUT_PROB = 0.5
        
        if random.random() < TEXT_DROPOUT_PROB:
            # 빈 문자열로 만들면 Tokenizer가 [CLS], [SEP] + Padding으로 처리
            new_name = ""
            
        return TrainingItem(
            product_id=item.product_id,
            feature_data=new_feature_data,
            product_name=new_name
        )

    def __getitem__(self, idx):
        item = self.products[idx]
        
        view1 = self._corrupt_data(item)
        view2 = self._corrupt_data(item)
        
        return view1, view2

# ----------------------------------------------------------------------
# 3. Collate Function (Tokenizer & Tensor Conversion)
# ----------------------------------------------------------------------

MAX_RE_LEN = 32  
MAX_TXT_LEN = 32
FIELD_PROMPT_MAP = {

    "[CAT]": "Clothing Category",   

    "[MAT]": "Fabric Material",     
    
    "[DET]": "Garment Detail",      
    
    "[FIT]": "Clothing Fit",        
    
    "[FNC]": "Apparel Function",    
    
    "[SPC]": "Product Specification", 
    
    "[COL]": "Garment Color",       
    
    "[CTX]": "Occasion",            
    
    "[LOC]": "Body Part"            
}
class SimCSECollator:
    """
    DataLoader에서 배치를 만들 때 토크나이징 및 텐서화를 수행하는 클래스.
    Tokenizer를 매번 로드하지 않기 위해 클래스로 감쌈.
    """
    def __init__(self, tokenizer_path=FASHION_BERT_MODEL):
        print(f"🔄 Initializing Collator with {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.std_keys = vocab.get_std_field_keys()
        self.re_keys = vocab.RE_FEATURE_KEYS
        
        # Global Config  나중에 MLops에 다 주입변수행으로
        self.max_re_len = MAX_RE_LEN
        self.max_txt_len = MAX_TXT_LEN
        

        self.sep = self.tokenizer.sep_token
        
        # data flow 초기 check flag
        self._has_logged_sample = False
    def _serialize_feature_value(self, value: Any) -> str:
        """
        리스트를 [SEP] 토큰으로 구분

        """
        if not value:
            return ""
        
        if isinstance(value, list):
            valid_items = [str(v) for v in value if v]
            if not valid_items:
                return ""
            # [SEP] 1 ...
            return f" {self.sep} ".join(valid_items)
            
        return str(value)

    def process_batch_items(self, items: List[TrainingItem], is_first_view: bool = False):
        """Raw Items -> Model Input Tensors 변환"""
        
    def process_batch_items(self, items: List[TrainingItem], is_first_view: bool = False):
        batch_std = []
        batch_re_ids = []
        batch_re_masks = []
        batch_txt = [] 

        # [Log] 첫 배치의 첫 번째 아이템의 모든 RE 필드 수집
        sample_log_re = [] 

        for idx, item in enumerate(items):
            # 1. STD
            std_ids = [vocab.get_std_id(item.feature_data.get(k, "")) for k in self.std_keys]
            batch_std.append(std_ids)

            # 2. RE 
            re_vals = [self._serialize_feature_value(item.feature_data.get(k)) for k in self.re_keys]
            
            curr_re_ids = []
            curr_re_masks = []
            for i, val in enumerate(re_vals):

                final_text = val
                if val:
                    # "[MAT]" -> "Material"
                    key_code = self.re_keys[i] 
                    prompt = FIELD_PROMPT_MAP.get(key_code, key_code) 
                    
                    # 텍스트 결합: "Material: Jersey"
                    final_text = f"{prompt}: {val}"

                enc = self.tokenizer(
                    final_text, # 👈 수정된 텍스트 입력
                    padding='max_length', 
                    truncation=True, 
                    max_length=self.max_re_len, 
                    add_special_tokens=True
                )
                curr_re_ids.append(enc['input_ids'])
                curr_re_masks.append(enc['attention_mask'])
                

                if idx == 0 and is_first_view and not self._has_logged_sample:
                    if val:
                        key_name = self.re_keys[i]
                        decoded = self.tokenizer.decode(enc['input_ids'], skip_special_tokens=False)

                        sample_log_re.append(f"      - {key_name}: '{final_text}' -> {decoded[:40]}...")

            batch_re_ids.append(curr_re_ids)    
            batch_re_masks.append(curr_re_masks) 
            batch_txt.append(item.product_name)

        # Tensor Stacking
        tensor_std = torch.tensor(batch_std, dtype=torch.long)
        tensor_re_ids = torch.tensor(batch_re_ids, dtype=torch.long)
        tensor_re_mask = torch.tensor(batch_re_masks, dtype=torch.long)

        txt_enc = self.tokenizer(
            batch_txt, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_txt_len, 
            return_tensors='pt'
        )
        
        # [Log Print]
        if not self._has_logged_sample and is_first_view:
            import sys
            msg = []
            msg.append("\n" + "="*60)
            msg.append(f"🔍 [Data Integrity Check] First Batch Sample")
            msg.append(f"   1. Product Name: '{items[0].product_name}'")
            if not items[0].product_name:
                msg.append(f"      ⚠️ WARNING: Product Name is EMPTY!")
            
            msg.append(f"   2. RE Features Found ({len(sample_log_re)} fields):")
            if sample_log_re:
                msg.extend(sample_log_re)
            else:
                msg.append("      ⚠️ NO RE FEATURES FOUND (Check Key Matching)")
            
            msg.append("="*60 + "\n")
            
            final_msg = "\n".join(msg)
            try:
                from tqdm import tqdm
                tqdm.write(final_msg)
            except ImportError:
                print(final_msg, flush=True)

            self._has_logged_sample = True

        return tensor_std, tensor_re_ids, tensor_re_mask, txt_enc['input_ids'], txt_enc['attention_mask']

    def __call__(self, batch):
        view1_list = [item[0] for item in batch]
        view2_list = [item[1] for item in batch]
        return self.process_batch_items(view1_list, is_first_view=True), self.process_batch_items(view2_list, is_first_view=False)
    
# ----------------------------------------------------------------------
# 4. Training Loop Implementation
# ----------------------------------------------------------------------

def train_simcse_from_db(    
    encoder: nn.Module,       
    projector: nn.Module,
    db_session, # DB Session 객체 주입 필요
    batch_size: int,
    epochs: int,
    lr: float
):
    print("🚀 Fetching data from DB...")
    stmt = select(
        ProductInferenceInput.product_id, 
        ProductInferenceInput.feature_data, 
        ProductInferenceInput.product_name 
    )
    result = db_session.execute(stmt).mappings().all()

    if not result:
        print("❌ [Error] No data found in DB.")
        return

    # load
    products_list = []
    
    for row in result:
        # DB의 원본 데이터 (수정 불가능하므로 dict로 복사)
        raw_feats = dict(row['feature_data'])
        
        # 'reinforced_feature'가 있다면 꺼내서 처리
        if 'reinforced_feature' in raw_feats:
            re_dict = raw_feats['reinforced_feature']
            # mat -> [mat] why? tsf encoder에는 상관없고, 혹시 bert embedding에서 더 잘 쓰일까봐.
            if isinstance(re_dict, dict):
                for key, val in re_dict.items():
   
                    if key.startswith("[") and key.endswith("]"):
                        vocab_key = key
                    else:
                        vocab_key = f"[{key}]"  # "MAT" -> "[MAT]"
                    

                    raw_feats[vocab_key] = val
                    
        # name tagging 
        base_name = row['product_name']
        product_type = raw_feats.get('product_type_name', "").strip() # 예: "Underwear Tights"
        
        final_name = ""

        if base_name:
            # Case A: 이름이 있는 경우 -> "원래이름 (Category: 타입명)"
            if product_type:
                final_name = f"{base_name} (Category: {product_type})"
            else:
                final_name = base_name
        else:
            # Case B: 이름이 없는 경우 (Fallback) -> "타입명 + 외형"
            appearance = raw_feats.get('graphical_appearance_name', "").strip()
            final_name = f"{product_type} {appearance}".strip()
            
            if not final_name:
                final_name = "Unknown Product"
                
                        
        item = TrainingItem(
                product_id=str(row['product_id']), 
                feature_data=raw_feats, 
                product_name=row['product_name'] if row['product_name'] else ""
            )
        products_list.append(item)
    print(f"✅ Loaded {len(products_list)} items.")
    # ▼▼▼ 디버깅 코드 
    print("\n🔎 [DEBUG] Raw DB Data Check (First Item):")
    first_row = result[0]
    print(f"   - Keys in feature_data: {list(first_row['feature_data'].keys())}")
    print(f"   - Full content: {first_row['feature_data']}")
    print("-" * 50 + "\n")
    # ▲▲▲ 여기까지 ▲▲▲
    
    
    # 1. Model Setup
    model = SimCSEModelWrapper(encoder, projector).to(DEVICE)
    model.train()
    # 🛠️ [AMP] Scaler 초기화 (GPU 사용 시)
    use_amp = (DEVICE == "cuda")
    scaler = GradScaler(enabled=use_amp)
    
    if use_amp:
        print("⚡ [AMP] Mixed Precision Training Enabled.")     
    bert_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue 
            

        if "bert_model" in name:
            bert_params.append(param)
        else:
            other_params.append(param)
    
    
    # 2. Optimization

    optimizer = AdamW([
        {
            'params': bert_params, 
            'lr': 1e-5  
        },
        {
            'params': other_params, 
            'lr': lr   
        }
    ])
    
    # 3. Dataset & DataLoader
    dataset = SimCSERecSysDataset(products_list, dropout_prob=0.4)
    collator = SimCSECollator() # Initialize Tokenizer once
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=collator, # Use the class instance
        drop_last=True,
        num_workers=0 # win - 멀티프로세싱 시 Tokenizer 이슈 주의
    )

    # 4. Scheduler
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # Loss Function (Contrastive)
    # in-batch negatives 활용
    from torch.nn import CrossEntropyLoss
    loss_func = CrossEntropyLoss()

    print("🔥 Starting Training Loop...")
    
    for epoch in range(epochs):
        total_loss = 0
        step = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        # Unpack Collator Outputs (5 tensors per view)
        for inputs_v1, inputs_v2 in progress:
            # Move to Device
            inputs_v1 = [t.to(DEVICE) for t in inputs_v1]
            inputs_v2 = [t.to(DEVICE) for t in inputs_v2]
            
            optimizer.zero_grad()
            
            
            with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    # Forward
                emb1 = model(*inputs_v1)
                emb2 = model(*inputs_v2)
                    
                    # Loss Calculation
                temperature = 0.05
                sim_matrix = torch.matmul(emb1, emb2.T) / temperature 
                    
                labels = torch.arange(emb1.size(0)).to(DEVICE)
                loss_1 = loss_func(sim_matrix, labels) 
                loss_2 = loss_func(sim_matrix.T, labels) 
                    
                loss = (loss_1 + loss_2) / 2
                
            scaler.scale(loss).backward()  # loss scaling
            scaler.step(optimizer)         # optimizer step with scaler
            scaler.update()                # update scaler factor
                    
            scheduler.step()
                    
            total_loss += loss.item()
            step += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
                
        if step > 0:
            avg_loss = total_loss / step
            print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}: No batches processed.")
        
        ckpt_name = f"encoder_ep{epoch+1:02d}_loss{avg_loss:.4f}.pth"
        save_path = os.path.join(MODEL_DIR, ckpt_name)
        
        # encoder only 
        torch.save(encoder.state_dict(), save_path)
        print(f"✅ Saved Checkpoint: {ckpt_name}")
        
  
        # torch.save(projector.state_dict(), os.path.join(MODEL_DIR, f"projector_ep{epoch+1:02d}.pth"))
        
        print("-" * 50)
        # =========================================================

    print("Training Finished.")