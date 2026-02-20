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

