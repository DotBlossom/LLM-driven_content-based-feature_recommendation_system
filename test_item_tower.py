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

from utils import vocab



# --- Global Configuration ---
EMBED_DIM = 128
OUTPUT_DIM_ENCODER = 128       # Encoder(Representation) 출력
OUTPUT_DIM_PROJECTOR = 128     # Projector(SimCSE Loss용) 출력
FASHION_BERT_MODEL = "bert-base-uncased" # 학습하면서 같이 tune됨
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAD_ID = vocab.PAD_ID
UNK_ID = vocab.UNK_ID

# ----------------------------------------------------------------------
# 1. Models: Encoder + Projector + Wrapper
# ----------------------------------------------------------------------

# (A) Encoder: HM_HybridItemTower 
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
        # 256 -> 128 로 압축하여 최종 임베딩 생성
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
    

class HM_HybridItemTower(nn.Module):
    def __init__(self,
                 std_vocab_size: int,
                 num_std_fields: int,
                 embed_dim: int = 128,  # [Update] Default to 128
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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = DeepResidualHead(input_dim=embed_dim, output_dim=output_dim)

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

        # 3. Text (Product Name) -> Full BERT Context
        bert_out = self.bert_model(input_ids=text_input_ids, attention_mask=text_attn_mask)
        cls_token = bert_out.last_hidden_state[:, 0, :] # [CLS]
        text_vec = self.text_proj(cls_token).unsqueeze(1) 

        # 4. Fusion
        combined_seq = torch.cat([std_emb, re_vectors, text_vec], dim=1) 
        context_out = self.transformer(combined_seq) 
        final_vec = context_out.mean(dim=1) 

        return self.head(final_vec)
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
    def __init__(self, products: List[TrainingItem], dropout_prob: float = 0.2):
        self.products = products
        self.dropout_prob = dropout_prob
        
        # Dropout 대상이 되는 Key 그룹 정의
        self.std_keys = vocab.get_std_field_keys() # ["product_type_name", ...]
        self.re_keys = vocab.RE_FEATURE_KEYS       # ["[CAT]", "[MAT]", ...]

    def __len__(self):
        return len(self.products)

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
        
        # SimCSE: 같은 아이템에 서로 다른 Dropout을 적용하여 View 1, View 2 생성
        view1 = self._apply_dropout(item)
        view2 = self._apply_dropout(item)
        
        return view1, view2

# ----------------------------------------------------------------------
# 3. Collate Function (Tokenizer & Tensor Conversion)
# ----------------------------------------------------------------------

MAX_RE_LEN = 32  
MAX_TXT_LEN = 32

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
    def _serialize_feature_value(self, value: Any) -> str:
        """
        [LLM preprosess 이어받음] 리스트를 [SEP] 토큰으로 구분하여 '독립적 의미 단위'임을 명시
        Example: ["elasticated waist", "extra space"] 
              -> "elasticated waist [SEP] extra space"
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

    def process_batch_items(self, items: List[TrainingItem]):
        """Raw Items -> Model Input Tensors 변환"""
        
        batch_std = []
        batch_re_ids = []
        batch_re_masks = []
        batch_txt = [] 

        for item in items:
            # 1. STD (기존 동일)
            std_ids = [vocab.get_std_id(item.feature_data.get(k, "")) for k in self.std_keys]
            batch_std.append(std_ids)

            # 2. RE (Multiple Values Handling 적용)
            re_vals = [self._serialize_feature_value(item.feature_data.get(k)) for k in self.re_keys]
            
            curr_re_ids = []
            curr_re_masks = []
            
            for val in re_vals:
                # [수정 2] max_length=32 적용
                enc = self.tokenizer(
                    val, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=self.max_re_len, # 32
                    add_special_tokens=True # [CLS], [SEP] 써야됨
                )
                curr_re_ids.append(enc['input_ids'])
                curr_re_masks.append(enc['attention_mask'])
            
            batch_re_ids.append(curr_re_ids)    
            batch_re_masks.append(curr_re_masks) 

            # 3. Text Name
            batch_txt.append(item.product_name)

        # Tensor Stacking (기존 동일)
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
        
        return tensor_std, tensor_re_ids, tensor_re_mask, txt_enc['input_ids'], txt_enc['attention_mask']

    def __call__(self, batch):
        # batch: List of (view1, view2)
        view1_list = [item[0] for item in batch]
        view2_list = [item[1] for item in batch]

        # Process View 1
        v1_inputs = self.process_batch_items(view1_list)
        # Process View 2
        v2_inputs = self.process_batch_items(view2_list)

        # Return: (std1, re_id1, re_mask1, txt1, txt_mask1), (std2, ... )
        return v1_inputs, v2_inputs

# ----------------------------------------------------------------------
# 4. Training Loop Implementation
# ----------------------------------------------------------------------

def train_simcse_from_db(    
    encoder: nn.Module,       
    projector: nn.Module,
    db_session, # DB Session 객체 주입 필요
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 1e-4
):
    print("🚀 Fetching data from DB...")
    
    # [DB Fetch Logic Placeholder]
    # 실제 환경에서는 SQL Alchemy select 사용
    # stmt = select(ProductInferenceInput.product_id, ProductInferenceInput.feature_data, ProductInferenceInput.product_name)
    # result = db_session.execute(stmt).mappings().all()
    
    # Dummy Data for Code Validation
    result = []
    for i in range(100):
        result.append({
            "product_id": str(i),
            "feature_data": {
                "product_type_name": "T-shirt",
                "colour_group_name": "Black" if i % 2 == 0 else "White",
                "[CAT]": "Top",
                "[MAT]": "Cotton"
            },
            "product_name": f"Basic T-shirt {i}"
        })

    if not result:
        print("❌ No data found.")
        return

    # Convert to Pydantic List
    products_list = [
        TrainingItem(
            product_id=row['product_id'], 
            feature_data=row['feature_data'],
            product_name=row.get('product_name', "")
        ) 
        for row in result
    ]
        
    print(f"✅ Loaded {len(products_list)} items.")
    
    # 1. Model Setup
    model = SimCSEModelWrapper(encoder, projector).to(DEVICE)
    model.train()
    
    # 2. Optimization
    # Fashion-BERT 일부(Embeddings)를 사용하므로 params 확인 필요
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # 3. Dataset & DataLoader
    dataset = SimCSERecSysDataset(products_list, dropout_prob=0.2)
    collator = SimCSECollator() # Initialize Tokenizer once
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=collator, # Use the class instance
        drop_last=True,
        num_workers=0 # 멀티프로세싱 시 Tokenizer 이슈 주의
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
            
            # Forward (SimCSEWrapper takes 5 args)
            # inputs_v1 = (std, re_ids, re_mask, txt_ids, txt_mask)
            emb1 = model(*inputs_v1)
            emb2 = model(*inputs_v2)
            
            # --- SimCSE Loss Calculation ---
            # Cosine Similarity Matrix
            # emb1, emb2 are normalized in Projector
            # sim_matrix: (Batch, Batch)
            sim_matrix = torch.matmul(emb1, emb2.T) 
            
            # Temperature Scaling
            temperature = 0.05
            sim_matrix = sim_matrix / temperature
            
            # Labels: 대각선 요소(자기 자신)가 정답
            labels = torch.arange(batch_size).to(DEVICE)
            
            loss = loss_func(sim_matrix, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            step += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/step:.4f}")
        
    print("Training Finished.")
    print("💾 Models are ready to save.")
    # torch.save(encoder.state_dict(), ...)

# ----------------------------------------------------------------------
# 5. FastAPI Endpoint (Mock)
# ----------------------------------------------------------------------
# 실제 앱에서는 router.post 등으로 구현
if __name__ == "__main__":
    # Mock Dependency Injection
    # 1. Vocab Setup (Imported from utils.vocab)
    std_size = vocab.get_std_vocab_size()
    num_std = len(vocab.get_std_field_keys())
    
    # 2. Instantiate Models
    encoder = HM_HybridItemTower(std_size, num_std, embed_dim=EMBED_DIM)
    projector = OptimizedItemTower(input_dim=128, output_dim=128)
    
    # 3. Run Train
    train_simcse_from_db(encoder, projector, db_session=None, batch_size=4, epochs=2)