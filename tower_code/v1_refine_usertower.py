import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm





def dataset_peek(dataset, processor):
    """Dataset에서 1개 샘플을 꺼내 로직이 정합한지 검수"""
    print("\n🧐 [Data Peek] Checking Sequence Integrity...")
    sample = dataset[0]
    
    # 1. 시퀀스 Shift 확인
    ids = sample['item_ids'].tolist()
    targets = sample['target_ids'].tolist()
    
    # 0이 아닌 첫 번째 실제 데이터 인덱스 찾기
    first_idx = next((i for i, x in enumerate(ids) if x != 0), None)
    
    if first_idx is not None and first_idx < len(ids) - 1:
        print(f"   - Input Seq  (t):   ... {ids[first_idx:first_idx+3]}")
        print(f"   - Target Seq (t+1): ... {targets[first_idx:first_idx+3]}")
        if ids[first_idx+1] == targets[first_idx]:
            print("   ✅ Shift Logic: OK (Input[t+1] == Target[t])")
        else:
            print("   ❌ Shift Logic: ERROR! Target is not shifted correctly.")

    # 2. 유저 스태틱 피처 확인
    print(f"   - Age Bucket ID: {sample['age_bucket'].item()}")
    print(f"   - Cont Feats Shape: {sample['cont_feats'].shape}")



class FeatureProcessor:
    def __init__(self, user_path, item_path, seq_path):
        print("🚀 Loading preprocessed features...")
        self.users = pd.read_parquet(user_path).drop_duplicates(subset=['customer_id']).set_index('customer_id')
        self.items = pd.read_parquet(item_path).drop_duplicates(subset=['article_id']).set_index('article_id')
        self.seqs = pd.read_parquet(seq_path).set_index('customer_id')

        # 인덱스 타입 강제 (String)
        self.users.index = self.users.index.astype(str)
        self.items.index = self.items.index.astype(str)
        self.seqs.index = self.seqs.index.astype(str)

        # =================================================================
        # 1. ID Mappings (1-based, 0 is Padding)
        # =================================================================
        self.user_ids = self.seqs.index.tolist() # 시퀀스가 존재하는 유저만 대상
        self.user2id = {uid: i + 1 for i, uid in enumerate(self.users.index)}
        self.item_ids = self.items.index.tolist()
        self.item2id = {iid: i + 1 for i, iid in enumerate(self.item_ids)}
        
        self.num_items = len(self.item_ids)

        # =================================================================
        # 2. Fast Lookup Arrays for Dataset (__getitem__ 속도 최적화)
        # =================================================================
        print("⚡ Building fast lookup tables...")
        
        # [A] User Features (유저 ID 1~N으로 바로 접근할 수 있도록 배열화)
        num_users_total = len(self.users) + 1
        
        # Bucket / Categorical (LongTensor용)
        self.u_bucket_arr = np.zeros((num_users_total, 4), dtype=np.int64) 
        self.u_cat_arr = np.zeros((num_users_total, 5), dtype=np.int64)
        # Continuous (FloatTensor용)
        self.u_cont_arr = np.zeros((num_users_total, 4), dtype=np.float32)

        # 매핑 수행
        for uid, row in self.users.iterrows():
            if uid not in self.user2id: continue
            uidx = self.user2id[uid]
            
            # Buckets: age, price, cnt, recency
            self.u_bucket_arr[uidx] = [
                row['age_bucket'], row['user_avg_price_bucket'], 
                row['total_cnt_bucket'], row['recency_bucket']
            ]
            # Categoricals: channel, club, news, fn, active
            self.u_cat_arr[uidx] = [
                row['preferred_channel'], row['club_member_status_idx'],
                row['fashion_news_frequency_idx'], row['FN'], row['Active']
            ]
            # Continuous Scaled: price_std, last_diff, repurch, weekend
            self.u_cont_arr[uidx] = [
                row['price_std_scaled'], row['last_price_diff_scaled'],
                row['repurchase_ratio_scaled'], row['weekend_ratio_scaled']
            ]

        # [B] Item Side Info Lookup (아이템 ID 1~N으로 바로 접근)
        # 아이템 데이터 프레임에 type_id, color_id 등이 있다고 가정
        self.i_side_arr = np.zeros((self.num_items + 1, 4), dtype=np.int64)
        for iid, row in self.items.iterrows():
            if iid not in self.item2id: continue
            idx = self.item2id[iid]
            # 전처리된 아이템 피처에 맞춰 컬럼명 수정 필요
            self.i_side_arr[idx] = [
                row.get('type_id', 0), row.get('color_id', 0), 
                row.get('graphic_id', 0), row.get('section_id', 0)
            ]

    def get_logq_probs(self, device):
        """Negative Sampling이나 Loss 보정을 위한 아이템 등장 확률 Log 반환"""
        raw_probs = self.items['raw_probability'].reindex(self.item_ids).values
        eps = 1e-6
        sorted_probs = np.nan_to_num(raw_probs, nan=0.0) + eps
        sorted_probs /= sorted_probs.sum()
        
        log_q_values = np.log(sorted_probs).astype(np.float32)
        
        full_log_q = np.zeros(self.num_items + 1, dtype=np.float32)
        full_log_q[1:] = log_q_values 
        full_log_q[0] = -20.0 # Padding Index
    
        return torch.tensor(full_log_q, dtype=torch.float32).to(device)
    
class SASRecDataset(Dataset):
    def __init__(self, processor: FeatureProcessor, max_len=30, is_train=True):
        self.processor = processor
        self.max_len = max_len
        self.is_train = is_train
        self.user_ids = processor.user_ids

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        u_mapped_id = self.processor.user2id.get(user_id, 0)
        
        # 1. 시퀀스 로드 
        seq_raw = self.processor.seqs.loc[user_id, 'sequence_ids']
        
        # 1-1. time deltas : 1년전 동일계절에 구매했던건? 최근은? 등등을 매핑
        time_deltas_raw = self.processor.seqs.loc[user_id, 'sequence_deltas']
        bins = np.array([0, 3, 7, 14, 30, 60, 180, 330, 395])
        time_buckets = np.digitize(time_deltas_raw, bins, right=False).tolist()
        
        
        seq = [self.processor.item2id.get(item, 0) for item in seq_raw]
        
        # =========================================================
        # 2. Causality Split (SASRec Shift Logic)
        # =========================================================
        if self.is_train:
            # 학습 시: input과 target을 위해 max_len + 1 개를 가져옴
            seq = seq[-(self.max_len + 1):]
            time_buckets = time_buckets[-(self.max_len + 1):] # [신규 추가] 타임 버킷도 동일하게 슬라이싱
            if len(seq) > 1:
                input_seq = seq[:-1]  # t 시점까지의 입력
                target_seq = seq[1:]  # t+1 시점의 정답
                input_time = time_buckets[:-1] # [신규 추가] t 시점의 시간 간격
            else:
                input_seq = seq
                target_seq = seq # 방어 코드 (길이가 1인 경우)
                input_time = time_buckets
        else:
            # 추론/검증 시: 최신 max_len 개를 입력으로 사용 (다음 1개를 예측하기 위해)
            input_seq = seq[-self.max_len:]
            target_seq = [] # Test loop에서 정답을 별도로 처리
            input_time = time_buckets[-self.max_len:] # [신규 추가]

        # =========================================================
        # 3. Left Padding
        # =========================================================
        # 최근 행동이 배열의 끝에 오도록 Left Padding을 적용
        pad_len = self.max_len - len(input_seq)
        input_padded = [0] * pad_len + input_seq
        time_padded = [0] * pad_len + input_time
        if self.is_train:
            target_padded = [0] * pad_len + target_seq
        else:
            target_padded = [0] * self.max_len

        # =========================================================
        # 4. Item Side Info Lookup (Sequence)
        # =========================================================
        # padding(0)인 경우 Lookup 배열의 0번째 인덱스(0,0,0,0)를 가져옴
        item_side_info = self.processor.i_side_arr[input_padded]
        
        type_ids = item_side_info[:, 0]
        color_ids = item_side_info[:, 1]
        graphic_ids = item_side_info[:, 2]
        section_ids = item_side_info[:, 3]

        # Padding Mask (True면 Transformer에서 무시)
        padding_mask = [True] * pad_len + [False] * len(input_seq)

        # =========================================================
        # 5. User Features Lookup (Static)
        # =========================================================
        u_buckets = self.processor.u_bucket_arr[u_mapped_id]
        u_cats = self.processor.u_cat_arr[u_mapped_id]
        u_conts = self.processor.u_cont_arr[u_mapped_id]

        # =========================================================
        # 6. Return Tensors
        # =========================================================
        return {
            # Sequence
            'item_ids': torch.tensor(input_padded, dtype=torch.long),
            'target_ids': torch.tensor(target_padded, dtype=torch.long),
            'padding_mask': torch.tensor(padding_mask, dtype=torch.bool),
            'time_bucket_ids': torch.tensor(time_padded, dtype=torch.long),
            
            # Item Side Info
            'type_ids': torch.tensor(type_ids, dtype=torch.long),
            'color_ids': torch.tensor(color_ids, dtype=torch.long),
            'graphic_ids': torch.tensor(graphic_ids, dtype=torch.long),
            'section_ids': torch.tensor(section_ids, dtype=torch.long),
            
            # User Buckets
            'age_bucket': torch.tensor(u_buckets[0], dtype=torch.long),
            'price_bucket': torch.tensor(u_buckets[1], dtype=torch.long),
            'cnt_bucket': torch.tensor(u_buckets[2], dtype=torch.long),
            'recency_bucket': torch.tensor(u_buckets[3], dtype=torch.long),
            
            # User Categoricals
            'channel_ids': torch.tensor(u_cats[0], dtype=torch.long),
            'club_status_ids': torch.tensor(u_cats[1], dtype=torch.long),
            'news_freq_ids': torch.tensor(u_cats[2], dtype=torch.long),
            'fn_ids': torch.tensor(u_cats[3], dtype=torch.long),
            'active_ids': torch.tensor(u_cats[4], dtype=torch.long),
            
            # User Continuous
            'cont_feats': torch.tensor(u_conts, dtype=torch.float32)
        }
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class SASRecUserTower(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.max_len = args.max_len
        self.dropout_rate = args.dropout

        # ==================================================================
        # 1. Sequence Embeddings (Dynamic: Short-term Intent)
        # ==================================================================
        self.item_proj = nn.Linear(args.pretrained_dim, self.d_model)
        self.item_id_emb = nn.Embedding(args.num_items + 1, self.d_model, padding_idx=0)
        
        self.type_emb = nn.Embedding(args.num_prod_types + 1, self.d_model, padding_idx=0)
        self.color_emb = nn.Embedding(args.num_colors + 1, self.d_model, padding_idx=0)
        self.graphic_emb = nn.Embedding(args.num_graphics + 1, self.d_model, padding_idx=0)
        self.section_emb = nn.Embedding(args.num_sections + 1, self.d_model, padding_idx=0)

        self.pos_emb = nn.Embedding(self.max_len, self.d_model)
        
        # [업데이트] Time-Aware 버킷 임베딩
        num_time_buckets = 12 
        self.time_emb = nn.Embedding(num_time_buckets, self.d_model, padding_idx=0)
        
        self.emb_ln = nn.LayerNorm(self.d_model)
        self.emb_dropout = nn.Dropout(self.dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=args.nhead,
            dim_feedforward=self.d_model * 2,
            dropout=self.dropout_rate,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)

        # ==================================================================
        # 2. Static Embeddings (Global: Long-term Preference)
        # ==================================================================
        #  (A) Categorical Embeddings (Cardinality에 따른 효율적 차원 할당)
        
        # 10구간 Bucket 피처들 (상대적으로 정보량이 많음) -> 16차원
        mid_dim = 16
        self.age_emb = nn.Embedding(11, mid_dim, padding_idx=0)      
        self.price_emb = nn.Embedding(11, mid_dim, padding_idx=0)    
        self.cnt_emb = nn.Embedding(11, mid_dim, padding_idx=0)      
        self.recency_emb = nn.Embedding(11, mid_dim, padding_idx=0)  

        # Binary 및 Low-Cardinality 피처들 -> 4차원
        low_dim = 4
        self.channel_emb = nn.Embedding(4, low_dim, padding_idx=0)   
        self.club_status_emb = nn.Embedding(4, low_dim, padding_idx=0) 
        self.news_freq_emb = nn.Embedding(3, low_dim, padding_idx=0)   
        self.fn_emb = nn.Embedding(3, low_dim, padding_idx=0)        
        self.active_emb = nn.Embedding(3, low_dim, padding_idx=0)    

        # (B) Continuous Features Projection
        # 4차원의 연속형 데이터를 16차원으로 키워 임베딩과 볼륨을 맞춤
        self.num_cont_feats = 4
        cont_proj_dim = 16
        self.cont_proj = nn.Linear(self.num_cont_feats, cont_proj_dim)

        # 모든 Static Feature의 Concat 후 총 차원 계산
        # (16 * 4) + (4 * 5) + 16 = 64 + 20 + 16 = 100
        total_static_input_dim = (mid_dim * 4) + (low_dim * 5) + cont_proj_dim
        
        self.static_mlp = nn.Sequential(
            nn.Linear(total_static_input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )

        # ==================================================================
        # 3. Final Fusion & Output
        # ==================================================================
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    def get_causal_mask(self, seq_len, device):
        # float('-inf') 대신 dtype=torch.bool을 사용하여 True/False 행렬로 생성
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    
    def forward(self, 
                # Sequence Inputs (Batch, Seq)
                pretrained_vecs, item_ids, 
                time_bucket_ids, 
                type_ids, color_ids, graphic_ids, section_ids,
                # Static Categorical Inputs (Batch, )
                age_bucket, price_bucket, cnt_bucket, recency_bucket,
                channel_ids, club_status_ids, news_freq_ids, fn_ids, active_ids,
                # Static Continuous Inputs (Batch, 4)
                cont_feats, 
                padding_mask=None,
                training_mode=True
                ):
        
        device = item_ids.device
        seq_len = item_ids.size(1)

        # -----------------------------------------------------------
        # Phase 1: Sequence Encoding (Short-term)
        # -----------------------------------------------------------
        seq_emb = self.item_proj(pretrained_vecs) 
        seq_emb += self.item_id_emb(item_ids)
        seq_emb += self.time_emb(time_bucket_ids) # Time Aware
        seq_emb += self.type_emb(type_ids)
        seq_emb += self.color_emb(color_ids)
        seq_emb += self.graphic_emb(graphic_ids)
        seq_emb += self.section_emb(section_ids)
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        seq_emb += self.pos_emb(positions)
        
        seq_emb = self.emb_ln(seq_emb)
        seq_emb = self.emb_dropout(seq_emb)

        causal_mask = self.get_causal_mask(seq_len, device)
        
        output = self.transformer_encoder(
            seq_emb, 
            mask=causal_mask, 
            src_key_padding_mask=padding_mask
        )

        # -----------------------------------------------------------
        # Phase 2: Static Encoding (Long-term)
        # -----------------------------------------------------------
        #  Dataset에서 전달받은 모든 피처들을 개별 임베딩
        emb_age = self.age_emb(age_bucket)
        emb_price = self.price_emb(price_bucket)
        emb_cnt = self.cnt_emb(cnt_bucket)
        emb_rec = self.recency_emb(recency_bucket)
        
        emb_chan = self.channel_emb(channel_ids)
        emb_club = self.club_status_emb(club_status_ids)
        emb_news = self.news_freq_emb(news_freq_ids)
        emb_fn = self.fn_emb(fn_ids)
        emb_act = self.active_emb(active_ids)
        
        # 연속형 변수 차원 확대
        cont_proj_vec = F.relu(self.cont_proj(cont_feats)) 
        
        # Concat All Static Features
        static_input = torch.cat([
            emb_age, emb_price, emb_cnt, emb_rec,
            emb_chan, emb_club, emb_news, emb_fn, emb_act,
            cont_proj_vec
        ], dim=1)
        
        # MLP Processing
        user_profile_vec = self.static_mlp(static_input) # (Batch, d_model)

        # -----------------------------------------------------------
        # Phase 3: Late Fusion
        # -----------------------------------------------------------
        if training_mode:
            user_profile_expanded = user_profile_vec.unsqueeze(1).expand(-1, seq_len, -1)
            final_vec = torch.cat([output, user_profile_expanded], dim=-1)
            final_vec = self.output_proj(final_vec)
            
            return F.normalize(final_vec, p=2, dim=-1)
        else:
            user_intent_vec = output[:, -1, :] 
            final_vec = torch.cat([user_intent_vec, user_profile_vec], dim=-1)
            final_vec = self.output_proj(final_vec)
            
            return F.normalize(final_vec, p=2, dim=-1)
        # -----------------------------------------------------------
        # SEQ + pretrained vec -> Transformer -> User Intent Vector late fusion
        # -----------------------------------------------------------
    
    
    

# ==========================================
# 1. Loss Functions (Flatten 지원 수정)
# ==========================================
# ==========================================
# 1. Loss Functions (In-Batch Negative + LogQ)
# ==========================================
def inbatch_corrected_logq_loss(user_emb, item_tower_emb, target_ids, log_q_tensor, temperature=0.1, lambda_logq=1.0):
    """
    In-Batch Negative Sampling과 LogQ 보정이 적용된 효율적인 CrossEntropy Loss
    
    Args:
        user_emb: (N, Dim) - Batch 단위 유저 벡터 (Flatten 적용됨)
        item_tower_emb: (Num_Items, Dim) - 전체 아이템 임베딩
        target_ids: (N, ) - 정답 아이템 ID (Flatten 적용됨)
        log_q_tensor: (Num_Items, ) - 전체 아이템의 등장 확률(Log)
        temperature: (float) - Softmax Temperature
        lambda_logq: (float) - 편향 제어 강도 (보통 1.0)
    """
    N = user_emb.size(0)
    
    # 1. 배치 내 등장한 정답 아이템들의 임베딩만 추출 (N, Dim)
    # 전체 47,062개가 아닌 배치 내 N개만 사용하여 메모리를 극도로 절약합니다.
    batch_item_emb = item_tower_emb[target_ids]
    
    # 2. In-Batch Logits 계산 (N, N)
    # i번째 유저 벡터와 j번째 아이템 벡터의 내적 (대각선 원소가 정답)
    logits = torch.matmul(user_emb, batch_item_emb.T)
    logits.div_(temperature)

    # 3. LogQ 편향 보정 (Sampling Bias Correction)
    if lambda_logq > 0.0:
        # 배치 내 등장한 아이템들의 LogQ 값 추출 (N,)
        batch_log_q = log_q_tensor[target_ids]
        
        # Google RecSys 논문 수식: s^c(x, y) = s(x, y) - log(P(y))
        # 정답이든 오답이든 해당 아이템의 인기도(LogQ)만큼 로짓을 깎아줌
        # Broadcasting: (N, N) 행렬의 각 열(Column)에서 해당 아이템의 LogQ를 뺌
        logits = logits - (batch_log_q.view(1, -1) * lambda_logq)

    # 4. 정답 Label 생성 (대각선 인덱스: 0, 1, 2, ..., N-1)
    # i번째 유저의 정답은 배치 내 i번째 아이템임
    labels = torch.arange(N, device=user_emb.device)
    
    # 5. 최종 CrossEntropyLoss 계산
    return F.cross_entropy(logits, labels)


def duorec_loss_refined(user_emb_1, user_emb_2, target_ids, temperature=0.1, lambda_sup=0.1):
    """
    Supervised Contrastive Learning (SupCon) + NaN 방지 및 패딩 처리 완료
    """
    batch_size = user_emb_1.size(0)
    device = user_emb_1.device
    
    # 1. 벡터 정규화
    z_i = F.normalize(user_emb_1, dim=1)
    z_j = F.normalize(user_emb_2, dim=1)
    
    # 2. Unsupervised Loss (InfoNCE)
    logits_unsup = torch.matmul(z_i, z_j.T) / temperature
    labels = torch.arange(batch_size, device=device)
    loss_unsup = F.cross_entropy(logits_unsup, labels)
    
    # 3. Supervised Loss
    loss_sup = torch.tensor(0.0, device=device)
    
    if lambda_sup > 0:
        targets = target_ids.view(-1, 1)
        
        # 같은 타겟을 공유하는 유저 Mask (Batch, Batch)
        mask = torch.eq(targets, targets.T).float()
        
        # [Fix 1: Padding 오인 방지] 타겟이 0(Padding)인 유저들은 전부 마스크 0으로 초기화
        pad_mask = (targets == 0).float()
        mask = mask * (1 - pad_mask) 
        
        # 자기 자신 제외
        mask.fill_diagonal_(0)
        
        if mask.sum() > 0:
            logits_sup = torch.matmul(z_i, z_i.T) / temperature
            diag_mask = torch.eye(batch_size, device=device).bool()
            
            # 대각선을 -inf로 마스킹 (자기 자신 제외)
            logits_sup.masked_fill_(diag_mask, float('-inf'))
            
            # Log Softmax 계산
            log_prob = F.log_softmax(logits_sup, dim=1)
            
            # [Fix 2: NaN 폭탄 방지] 대각선의 -inf가 mask(0)와 곱해져 NaN이 되는 것을 막기 위해 0.0으로 덮어씀
            log_prob = log_prob.masked_fill(diag_mask, 0.0)
            
            # Positive Sample이 존재하는 유저만 필터링
            valid_rows = mask.sum(1) > 0
            if valid_rows.sum() > 0:
                loss_sup_batch = -(mask[valid_rows] * log_prob[valid_rows]).sum(1) / mask[valid_rows].sum(1)
                loss_sup = loss_sup_batch.mean()
                
    return loss_unsup + (lambda_sup * loss_sup)

# ==========================================
# 2. Main Training Logic
# ==========================================

def train_model(dataloader, item_tower, args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SASRecUserTower(args).to(device)
    
    # Item Tower의 임베딩을 가져옴 (Freeze 가정)
    item_tower.eval()
    with torch.no_grad():
        # 전체 아이템 임베딩 테이블 (Num_Items, Dim) - 0번은 Padding
        full_item_embeddings = item_tower.get_all_embeddings().to(device) 
        # LogQ (Popularity Correction)
        log_q_tensor = item_tower.get_log_q().to(device)

    # Optimizer (Transformer에 적합한 AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    model.train()
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # -------------------------------------------------------
        # 1. Data Unpacking (Dictionary to Device)
        # -------------------------------------------------------
        
        pretrained_vecs = batch.get('pretrained_vecs', None)
        if pretrained_vecs is not None:
            pretrained_vecs = pretrained_vecs.to(device)
            
        item_ids = batch['item_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        time_bucket_ids = batch['time_bucket_ids'].to(device)
        
        type_ids = batch['type_ids'].to(device)
        color_ids = batch['color_ids'].to(device)
        graphic_ids = batch['graphic_ids'].to(device)
        section_ids = batch['section_ids'].to(device)
        
        age_bucket = batch['age_bucket'].to(device)
        price_bucket = batch['price_bucket'].to(device)
        cnt_bucket = batch['cnt_bucket'].to(device)
        recency_bucket = batch['recency_bucket'].to(device)
        
        channel_ids = batch['channel_ids'].to(device)
        club_status_ids = batch['club_status_ids'].to(device)
        news_freq_ids = batch['news_freq_ids'].to(device)
        fn_ids = batch['fn_ids'].to(device)
        active_ids = batch['active_ids'].to(device)
        
        cont_feats = batch['cont_feats'].to(device)

        # -------------------------------------------------------
        # 2. Forward Pass with AMP
        # -------------------------------------------------------
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            
            # Forward Arguments Dictionary 
            forward_kwargs = {
                'pretrained_vecs': pretrained_vecs,
                'item_ids': item_ids,
                'time_bucket_ids': time_bucket_ids,
                'type_ids': type_ids,
                'color_ids': color_ids,
                'graphic_ids': graphic_ids,
                'section_ids': section_ids,
                'age_bucket': age_bucket,
                'price_bucket': price_bucket,
                'cnt_bucket': cnt_bucket,
                'recency_bucket': recency_bucket,
                'channel_ids': channel_ids,
                'club_status_ids': club_status_ids,
                'news_freq_ids': news_freq_ids,
                'fn_ids': fn_ids,
                'active_ids': active_ids,
                'cont_feats': cont_feats,
                'padding_mask': padding_mask,
                'training_mode': True
            }

            # A. First View (Main Task + DuoRec View 1)
            output_1 = model(**forward_kwargs)

            # B. Second View (DuoRec View 2)
            # Dropout 마스크가 다르게 적용되어 다른 벡터가 생성됨
            output_2 = model(**forward_kwargs)

            # -------------------------------------------------------
            # 3. Loss Calculation
            # -------------------------------------------------------
            
            # (1) Main Loss (All Time Steps)
            # Padding 부분(True)을 제외하고 Flatten
            valid_mask = ~padding_mask.view(-1)
            flat_output = output_1.view(-1, args.d_model)[valid_mask]
            flat_targets = target_ids.view(-1)[valid_mask]
            
            main_loss = inbatch_corrected_logq_loss(
                user_emb=flat_output,
                item_tower_emb=full_item_embeddings,
                target_ids=flat_targets,
                log_q_tensor=log_q_tensor,
                lambda_logq=args.lambda_logq
            )

            # (2) DuoRec Loss (Last Time Step Only)
            # Left Padding 구조이므로 유효한 마지막 아이템은 항상 배열의 맨 끝(인덱스 -1)에 위치함
            last_output_1 = output_1[:, -1, :] 
            last_output_2 = output_2[:, -1, :]
            last_targets = target_ids[:, -1]

            cl_loss = duorec_loss_refined(
                user_emb_1=last_output_1,
                user_emb_2=last_output_2,
                target_ids=last_targets,
                lambda_sup=args.lambda_sup
            )

            total_loss = main_loss + (args.lambda_cl * cl_loss)

        # -------------------------------------------------------
        # 4. Backward & Step (AMP)
        # -------------------------------------------------------
        scaler.scale(total_loss).backward()
        
        # Gradient Clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss {total_loss.item():.4f} (Main: {main_loss.item():.4f}, CL: {cl_loss.item():.4f})")
            
            
            
            
            
            
            
            
            
            
import os
import torch
import torch.nn as nn
import numpy as np
import random
from dataclasses import dataclass


# =====================================================================
# [Config] 파이프라인 설정 
# =====================================================================
@dataclass
class PipelineConfig:
    # Paths
    base_dir: str = r"D:\trainDataset\localprops"
    model_dir: str = r"C:\Users\candyform\Desktop\inferenceCode\models"
    
    # Hyperparameters
    batch_size: int = 896
    lr: float = 5e-5
    weight_decay: float = 1e-4
    epochs: int = 5
    
    # Model Args (SASRecUserTower용)
    d_model: int = 128
    max_len: int = 50
    dropout: float = 0.3
    pretrained_dim: int = 128 # 사전학습 아이템 벡터 차원 
    nhead: int = 4
    num_layers: int = 2
    
    # Loss Penalties
    lambda_logq: float = 0.1
    lambda_sup: float = 0.1
    lambda_cl: float = 0.1

    # 자동 할당될 메타데이터 크기
    num_items: int = 0
    num_prod_types: int = 0
    num_colors: int = 0
    num_graphics: int = 0
    num_sections: int = 0
    num_age_groups: int = 10

# =====================================================================
# Phase 1: Environment Setup
# =====================================================================
def setup_environment(seed: int = 42):
    """난수 고정 및 디바이스 설정 (Airflow Task 독립성 보장)"""
    print("\n⚙️ [Phase 1] Setting up environment...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device set to: {device}")
    return device

# =====================================================================
# Phase 2: Data Preparation
# =====================================================================
def prepare_features(cfg: PipelineConfig):
    """FeatureProcessor 초기화 및 메타데이터 업데이트"""
    print("\n📊 [Phase 2] Loading Processors...")
    
    # 경로 설정
    user_path = os.path.join(cfg.base_dir, "features_user_w_meta.parquet") 
    item_path = os.path.join(cfg.base_dir, "features_item.parquet")
    seq_path = os.path.join(cfg.base_dir, "features_sequence_cleaned.parquet")
    
    # Processor 초기화 
    processor = FeatureProcessor(user_path, item_path, seq_path)
    
    # Config에 임베딩 레이어 생성을 위한 메타데이터 업데이트
    cfg.num_items = processor.num_items
    
    ####### 실제 item metadata id랑 묶인상태로 가져와야하고 연결 필요 #######

    cfg.num_prod_types = int(processor.items['type_id'].max()) if 'type_id' in processor.items else 50
    cfg.num_colors = int(processor.items['color_id'].max()) if 'color_id' in processor.items else 50
    cfg.num_graphics = int(processor.items['graphic_id'].max()) if 'graphic_id' in processor.items else 50
    cfg.num_sections = int(processor.items['section_id'].max()) if 'section_id' in processor.items else 50

    print(f"✅ Features Loaded. Total Items: {cfg.num_items}")
    return processor, cfg

# =====================================================================
# Phase 3: Embedding Alignment & DataLoader
# =====================================================================
def load_aligned_pretrained_embeddings(processor, model_dir, pretrained_dim):
    """Dataset에서 사용할 수 있도록 정렬된 사전학습 벡터(N+1, Dim) 생성"""
    print(f"\n🔄 [Phase 3-1] Aligning Pretrained Item Embeddings...")
    emb_path = os.path.join(model_dir, "pretrained_item_matrix.pt")
    ids_path = os.path.join(model_dir, "item_ids.pt")

    num_embeddings = processor.num_items + 1 
    aligned_weight = torch.randn(num_embeddings, pretrained_dim) * 0.01 
    aligned_weight[0] = 0.0 # Padding
    
    try:
        pretrained_emb = torch.load(emb_path, map_location='cpu')
        if isinstance(pretrained_emb, dict):
            pretrained_emb = pretrained_emb.get('weight', pretrained_emb.get('item_content_emb.weight'))
        pretrained_ids = torch.load(ids_path, map_location='cpu')
        
        pretrained_map = {str(iid.item()) if isinstance(iid, torch.Tensor) else str(iid): pretrained_emb[idx] 
                          for idx, iid in enumerate(pretrained_ids)}
        
        matched = 0
        for i, current_id_str in enumerate(processor.item_ids):
            if current_id_str in pretrained_map:
                aligned_weight[i + 1] = pretrained_map[current_id_str]
                matched += 1
                
        print(f"✅ Matched: {matched}/{len(processor.item_ids)}")
    except Exception as e:
        print(f"⚠️ [Warning] Failed to load Pretrained files: {e}. Using random init.")
        
    return aligned_weight

def create_dataloaders(processor, cfg: PipelineConfig, aligned_pretrained_vecs=None):
    """Dataset 및 DataLoader 인스턴스화"""
    print("\n📦 [Phase 3-2] Creating DataLoaders...")
    
    # SASRecDataset 내부에서 aligned_pretrained_vecs를 참조하게끔 
    
    train_dataset = SASRecDataset(processor, max_len=cfg.max_len, is_train=True)
    
    # Dataset 인스턴스에 정렬된 pretrained vector 룩업 테이블 주입 (동적 바인딩)
    train_dataset.pretrained_lookup = aligned_pretrained_vecs 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=0, 
        pin_memory=True,
        drop_last=True
    )
    
    print(f"✅ Train Loader Ready: {len(train_loader)} batches/epoch")
    return train_loader


import hashlib
import json

def get_hash_id(text, hash_size):
    """문자열을 일관된 정수 ID(1 ~ hash_size)로 해싱 (0은 Padding)"""
    if not text or str(text).lower() in ['unknown', 'nan', 'none']:
        return 0
    # MD5를 사용하여 파이썬 세션이 바뀌어도 항상 동일한 해시값 보장
    hash_obj = hashlib.md5(str(text).strip().lower().encode('utf-8'))
    # 16진수를 정수로 변환 후 hash_size로 나눈 나머지 + 1
    return (int(hash_obj.hexdigest(), 16) % hash_size) + 1

def load_item_metadata_hashed(processor, base_dir, hash_size=1000):
    """JSON 파일을 읽어 정렬된 메타데이터 해시 텐서(N+1, 4)를 생성"""
    print("\n🏷️ [Phase 3-2] Loading and Hashing Item Metadata...")
    json_path = os.path.join(base_dir, "filtered_data_reinforced.json")
    
    num_items = processor.num_items + 1
    # 0번 인덱스는 패딩을 위해 0으로 유지 (N+1, 4차원 배열)
    item_side_arr = np.zeros((num_items, 4), dtype=np.int64)
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            item_data = json.load(f)
    except Exception as e:
        print(f"❌ [Error] Failed to load JSON: {e}")
        return torch.tensor(item_side_arr, dtype=torch.long)
    
    # 빠른 검색을 위해 O(1) Lookup Dictionary 생성
    # int형 article_id를 string으로 변환하여 매핑
    metadata_dict = {str(item.get('article_id', '')): item for item in item_data}
    
    matched = 0
    for i, current_id_str in enumerate(processor.item_ids):
        idx = i + 1 # 1-based indexing
        
        if current_id_str in metadata_dict:
            meta = metadata_dict[current_id_str]
            
            # 카테고리 매핑 및 해싱 (해당 키가 없으면 빈 문자열 반환)
            type_val = meta.get("product_type_name", "")
            color_val = meta.get("colour_group_name", "")
            graphic_val = meta.get("graphical_appearance_name", "")
            section_val = meta.get("section_name", "")
            
            item_side_arr[idx, 0] = get_hash_id(type_val, hash_size)
            item_side_arr[idx, 1] = get_hash_id(color_val, hash_size)
            item_side_arr[idx, 2] = get_hash_id(graphic_val, hash_size)
            item_side_arr[idx, 3] = get_hash_id(section_val, hash_size)
            
            matched += 1

    print(f"✅ Metadata Matched & Hashed: {matched}/{len(processor.item_ids)} (Hash Size: {hash_size})")
    
    return torch.tensor(item_side_arr, dtype=torch.long)
# =====================================================================
# Phase 4: Model Setup
# =====================================================================
class DummyItemTower(nn.Module):
    """실행 테스트용 더미 아이템 타워"""
    def __init__(self, num_items, dim):
        super().__init__()
        self.emb = nn.Embedding(num_items + 1, dim)
        self.log_q = nn.Parameter(torch.zeros(num_items + 1), requires_grad=False)
    def get_all_embeddings(self): return self.emb.weight
    def get_log_q(self): return self.log_q

def setup_models(cfg: PipelineConfig, device):
    """User Tower 초기"""
    print("\n🧠 [Phase 4] Initializing Models...")
    
    user_tower = SASRecUserTower(cfg).to(device)
    


    print("✅ Models initialized and moved to device.")
    return user_tower

# =====================================================================
# Phase 5: Training Loop (1 Epoch Runner)
# =====================================================================
def train_one_epoch(epoch, model, full_item_embeddings, log_q_tensor, dataloader, optimizer, scaler, cfg, device):
    """단일 에포크 훈련 함수 (실제 Loss 계산 및 로그 모니터링 적용)"""
    model.train()
    total_loss_accum = 0.0
    main_loss_accum = 0.0
    cl_loss_accum = 0.0
    
        
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()

        # -------------------------------------------------------
        # 1. Data Unpacking (Dictionary to Device)
        # -------------------------------------------------------
        item_ids = batch['item_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        time_bucket_ids = batch['time_bucket_ids'].to(device)
        
        type_ids = batch['type_ids'].to(device)
        color_ids = batch['color_ids'].to(device)
        graphic_ids = batch['graphic_ids'].to(device)
        section_ids = batch['section_ids'].to(device)
        
        age_bucket = batch['age_bucket'].to(device)
        price_bucket = batch['price_bucket'].to(device)
        cnt_bucket = batch['cnt_bucket'].to(device)
        recency_bucket = batch['recency_bucket'].to(device)
        
        channel_ids = batch['channel_ids'].to(device)
        club_status_ids = batch['club_status_ids'].to(device)
        news_freq_ids = batch['news_freq_ids'].to(device)
        fn_ids = batch['fn_ids'].to(device)
        active_ids = batch['active_ids'].to(device)
        
        cont_feats = batch['cont_feats'].to(device)
        
        # Pretrained Vector 룩업 처리
        if 'pretrained_vecs' in batch:
            pretrained_vecs = batch['pretrained_vecs'].to(device)
        else:
            pretrained_vecs = dataloader.dataset.pretrained_lookup[item_ids.cpu()].to(device)
            
        forward_kwargs = {
            'pretrained_vecs': pretrained_vecs,
            'item_ids': item_ids,
            'time_bucket_ids': time_bucket_ids,
            'type_ids': type_ids,
            'color_ids': color_ids,
            'graphic_ids': graphic_ids,
            'section_ids': section_ids,
            'age_bucket': age_bucket,
            'price_bucket': price_bucket,
            'cnt_bucket': cnt_bucket,
            'recency_bucket': recency_bucket,
            'channel_ids': channel_ids,
            'club_status_ids': club_status_ids,
            'news_freq_ids': news_freq_ids,
            'fn_ids': fn_ids,
            'active_ids': active_ids,
            'cont_feats': cont_feats,
            'padding_mask': padding_mask,
            'training_mode': True
        }

        # =======================================================
        # [모니터링 로그] 첫 배치에서만 데이터 상태 점검
        # =======================================================
        if batch_idx == 0:
            print(f"\n📦 [Batch 0 Monitor]")
            print(f"   - Item IDs: Shape {item_ids.shape} | Min {item_ids.min()} | Max {item_ids.max()}")
            print(f"   - Time Buckets: Min {time_bucket_ids.min()} | Max {time_bucket_ids.max()}")
            pad_ratio = (padding_mask.sum().item() / padding_mask.numel()) * 100
            print(f"   - Padding Ratio: {pad_ratio:.1f}%")
            print(f"   - Cont Feats Mean: {cont_feats.mean().item():.3f} | Std: {cont_feats.std().item():.3f}")
            
            print("\n🎯 [First User Data State Check]")
            print("-" * 50)
            print(f"👤 [User Profile]")
            print(f"   - Age Bucket ID:    {age_bucket[0].item()} (Target Age Group)")
            print(f"   - Price Bucket ID:  {price_bucket[0].item()} (Spending Power)")
            print(f"   - News Freq ID:     {news_freq_ids[0].item()} (Marketing Sensitivity)")
            
            valid_indices = torch.where(~padding_mask[0])[0]
            if len(valid_indices) > 0:
                print(f"\n🛍️ [Item History - Last 3 Items]")
                sample_indices = valid_indices[-3:] 
                sample_types = type_ids[0][sample_indices].tolist()
                sample_times = time_bucket_ids[0][sample_indices].tolist()
                for i, (t_id, time_id) in enumerate(zip(sample_types, sample_times)):
                    print(f"   - Item {i+1}: Type Hash ID [{t_id}] | Time Bucket ID [{time_id}]")
            else:
                print("\n⚠️ [Warning] This user has NO valid sequence (All Padded).")
            print("-" * 50)

        # -------------------------------------------------------
        # 2. Forward & Real Loss Calculation (AMP)
        # -------------------------------------------------------
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            # A. First View
            output_1 = model(**forward_kwargs)
            # B. Second View (Dropout 마스크가 달라짐)
            output_2 = model(**forward_kwargs)

            # (1) Main Loss (All Time Steps)
            valid_mask = ~padding_mask.view(-1)
            flat_output = output_1.view(-1, cfg.d_model)[valid_mask]
            flat_targets = target_ids.view(-1)[valid_mask]
            
            

            main_loss = inbatch_corrected_logq_loss(
                user_emb=flat_output,
                item_tower_emb=full_item_embeddings,
                target_ids=flat_targets,
                log_q_tensor=log_q_tensor,
                lambda_logq=cfg.lambda_logq
            )
            
            # (2) DuoRec Loss (Last Time Step Only)
            last_output_1 = output_1[:, -1, :] 
            last_output_2 = output_2[:, -1, :]
            last_targets = target_ids[:, -1]

            cl_loss = duorec_loss_refined(
                user_emb_1=last_output_1,
                user_emb_2=last_output_2,
                target_ids=last_targets,
                lambda_sup=cfg.lambda_sup
            )

            # 최종 Loss 조합
            total_loss = main_loss + (cfg.lambda_cl * cl_loss)

        # -------------------------------------------------------
        # 3. Backward & Optimizer Step
        # -------------------------------------------------------
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        # 기울기 폭발 방지를 위한 정규화 (5.0은 트랜스포머에서 많이 쓰이는 여유있는 값)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        # 누적
        total_loss_accum += total_loss.item()
        main_loss_accum += main_loss.item()
        cl_loss_accum += cl_loss.item()
        
        pbar.set_postfix({
            'Loss': f"{total_loss.item():.4f}",
            'Main': f"{main_loss.item():.4f}",
            'CL': f"{cl_loss.item():.4f}"
        })
        
        # 100배치마다 로깅
        if batch_idx % 100 == 0:
            print(f"   [Epoch {epoch}] Batch {batch_idx:04d}/{len(dataloader)} | Total Loss: {total_loss.item():.4f} (Main: {main_loss.item():.4f}, CL: {cl_loss.item():.4f})")

    avg_loss = total_loss_accum / len(dataloader)
    avg_main = main_loss_accum / len(dataloader)
    avg_cl = cl_loss_accum / len(dataloader)
    
    print(f"🏁 Epoch {epoch} Completed | Avg Total: {avg_loss:.4f} (Main: {avg_main:.4f}, CL: {avg_cl:.4f})")
    return avg_loss
# =====================================================================
# Main Execution Pipeline
# =====================================================================
def run_pipeline():
    """Airflow DAG나 MLflow Run에서 직접 호출하는 엔트리 포인트"""
    print("🚀 Starting User Tower Training Pipeline...")
    
    # 1. Config & Env
    cfg = PipelineConfig()
    device = setup_environment()
    processor, cfg = prepare_features(cfg)
    # item metadata cfg
    HASH_SIZE = 1000
    cfg.num_prod_types = HASH_SIZE
    cfg.num_colors = HASH_SIZE
    cfg.num_graphics = HASH_SIZE
    cfg.num_sections = HASH_SIZE
    
    # 2. Data

    aligned_vecs = load_aligned_pretrained_embeddings(processor, cfg.model_dir, cfg.pretrained_dim)
    
    full_item_embeddings = aligned_vecs.to(device)
    log_q_tensor = processor.get_logq_probs(device)
    
    
    item_metadata_tensor = load_item_metadata_hashed(processor, cfg.base_dir, hash_size=HASH_SIZE)
    processor.i_side_arr = item_metadata_tensor.numpy()
    train_loader = create_dataloaders(processor, cfg, aligned_vecs)
    dataset_peek(train_loader.dataset, processor)
    
    # 3. Models & Optimizer
    user_tower = setup_models(cfg, device)
    optimizer = torch.optim.AdamW(user_tower.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # 4. Training Loop (Phase 5)
    # mlflow.start_run() 블록용
    for epoch in range(1, cfg.epochs + 1):
        avg_loss = train_one_epoch(
            epoch=epoch,
            model=user_tower,
            full_item_embeddings=full_item_embeddings,
            log_q_tensor=log_q_tensor,

            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            device=device
        )
        # mlflow.log_metric("train_loss", avg_loss, step=epoch)
        
    print("🎉 Pipeline Execution Finished Successfully!")

if __name__ == "__main__":
    run_pipeline()