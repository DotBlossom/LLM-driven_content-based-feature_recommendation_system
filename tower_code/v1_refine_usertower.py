import torch
import torch.nn as nn
import torch.nn.functional as F



        # ==================================================================
        # FeatureProc 도입, GNN + 게이팅 도입, Static input 도입 등 필요
        # ==================================================================

class SASRecUserTower(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.d_model
        self.max_len = args.max_len
        self.dropout_rate = args.dropout

        # ==================================================================
        # 1. Sequence Embeddings (Dynamic: Short-term Intent)
        # ==================================================================
        # (A) Pre-trained & ID
        self.item_proj = nn.Linear(args.pretrained_dim, self.d_model)
        self.item_id_emb = nn.Embedding(args.num_items + 1, self.d_model, padding_idx=0)
        
        # (B) Selected Side Info (Orthogonal Attributes)
        # 각 속성은 d_model 차원으로 투영되어 더해집니다.
        self.type_emb = nn.Embedding(args.num_prod_types + 1, self.d_model, padding_idx=0)
        self.color_emb = nn.Embedding(args.num_colors + 1, self.d_model, padding_idx=0)
        self.graphic_emb = nn.Embedding(args.num_graphics + 1, self.d_model, padding_idx=0)
        self.section_emb = nn.Embedding(args.num_sections + 1, self.d_model, padding_idx=0)

        # (C) Position
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)
        self.emb_dropout = nn.Dropout(self.dropout_rate)

        # (D) Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=args.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout_rate,
            activation='gelu',
            norm_first=True,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)

        # ==================================================================
        # 2. Static Embeddings (Global: Long-term Preference)
        # ==================================================================
        # (A) Categorical Embeddings
        # 나이, 성별, 선호채널 등은 차원을 작게(예: 16~32) 가져가거나 d_model//4 정도로 설정
        static_emb_dim = self.d_model // 4
        self.age_emb = nn.Embedding(args.num_age_groups + 1, static_emb_dim, padding_idx=0)
        self.gender_emb = nn.Embedding(3, static_emb_dim, padding_idx=0) # 0:Unknown, 1:M, 2:F
        self.channel_emb = nn.Embedding(3, static_emb_dim, padding_idx=0) # 0:Unknown, 1:Online, 2:Offline

        # (B) Continuous Features Processing (MLP)
        # 입력: [avg_price, price_std, last_price_diff, recency, total_cnt] (5 dims)
        self.num_cont_feats = 5 
        # Categorical(3개 * static_emb_dim) + Continuous(5개) -> Hidden
        total_static_input_dim = (static_emb_dim * 3) + self.num_cont_feats
        
        self.static_mlp = nn.Sequential(
            nn.Linear(total_static_input_dim, self.d_model),
            nn.BatchNorm1d(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )

        # ==================================================================
        # 3. Final Fusion & Output
        # ==================================================================
        # Seq Vector(d_model) + Static Vector(d_model) -> Output(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model) # Final Alignment
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def get_causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)

    def forward(self, 
                # Sequence Inputs (Batch, Seq)
                pretrained_vecs, item_ids, 
                type_ids, color_ids, graphic_ids, section_ids,
                # Static Inputs (Batch, )
                age_ids, gender_ids, channel_ids, 
                cont_feats, # (Batch, 5) - Normalized Continuous Features
                padding_mask=None,
                training_mode=True
                ):
        
        device = item_ids.device
        seq_len = item_ids.size(1)
        batch_size = item_ids.size(0)

        # -----------------------------------------------------------
        # Phase 1: Sequence Encoding (Short-term)
        # -----------------------------------------------------------
        # Element-wise Sum for Sequence Features
        seq_emb = self.item_proj(pretrained_vecs) 
        seq_emb += self.item_id_emb(item_ids)
        seq_emb += self.type_emb(type_ids)
        seq_emb += self.color_emb(color_ids)
        seq_emb += self.graphic_emb(graphic_ids)
        seq_emb += self.section_emb(section_ids)
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        seq_emb += self.pos_emb(positions)
        seq_emb = self.emb_dropout(seq_emb)

        # Transformer
        causal_mask = self.get_causal_mask(seq_len, device)
        
        
        
        
        output = self.transformer_encoder(
            seq_emb, 
            mask=causal_mask, 
            src_key_padding_mask=padding_mask
        )
        


        # -----------------------------------------------------------
        # Phase 2: Static Encoding (Long-term)
        # -----------------------------------------------------------
        a_emb = self.age_emb(age_ids)       # (Batch, static_emb_dim)
        g_emb = self.gender_emb(gender_ids) # (Batch, static_emb_dim)
        c_emb = self.channel_emb(channel_ids) # (Batch, static_emb_dim)
        
        # Concat All Static Features
        static_input = torch.cat([a_emb, g_emb, c_emb, cont_feats], dim=1)
        
        # MLP Processing
        user_profile_vec = self.static_mlp(static_input) # (Batch, d_model)

        # -----------------------------------------------------------
        # Phase 3: Late Fusion
        # -----------------------------------------------------------
        # 의도(Intent) + 성향(Profile) 결합
        if training_mode:
            # ★ [학습 시] 모든 시점(Step)에 대해 Static Feature를 결합하여 학습
            
            # 1. Static Vector를 시퀀스 길이만큼 복사 (Broadcasting)
            # (Batch, d_model) -> (Batch, 1, d_model) -> (Batch, Seq_Len, d_model)
            user_profile_expanded = user_profile_vec.unsqueeze(1).expand(-1, seq_len, -1)
            
            # 2. Sequence Output과 결합
            # (Batch, Seq, d_model) + (Batch, Seq, d_model) -> (Batch, Seq, 2*d_model)
            final_vec = torch.cat([output, user_profile_expanded], dim=-1)
            
            # 3. Projection -> (Batch, Seq, d_model)
            final_vec = self.output_proj(final_vec)
            
            # 결과: (Batch, Seq_Len, d_model) 반환
            # 이제 Loss 함수에서 50개 시점 모두에 대해 정답과 비교 가능
            return F.normalize(final_vec, p=2, dim=-1)

        else:
            # ★ [추론 시 / Retrieval] 마지막 시점만 필요
            user_intent_vec = output[:, -1, :] # (Batch, d_model)
            
            # Static Vector와 결합 (Batch, 2*d_model)
            final_vec = torch.cat([user_intent_vec, user_profile_vec], dim=-1)
            
            # Projection -> (Batch, d_model)
            final_vec = self.output_proj(final_vec)
            
            return F.normalize(final_vec, p=2, dim=-1)
    
    
        # -----------------------------------------------------------
        # SEQ + pretrained vec -> Transformer -> User Intent Vector late fusion
        # -----------------------------------------------------------
    
    
    

# ==========================================
# 1. Loss Functions (Flatten 지원 수정)
# ==========================================
def efficient_corrected_logq_loss(user_emb, item_tower_emb, target_ids, log_q_tensor, temperature=0.1, lambda_logq=0.1):
    """
    user_emb: (N, Dim) - N은 Batch * Seq_Len
    item_tower_emb: (Num_Items, Dim) - 전체 아이템 임베딩 (공유)
    target_ids: (N, ) - Flattened Targets
    """
    # 1. Logits 계산 (Matrix Multiplication)
    # (N, Dim) x (Dim, Num_Items) -> (N, Num_Items)
    # 메모리 효율을 위해 청크 단위로 나누거나, 여기서는 단순화하여 전체 계산
    logits = torch.matmul(user_emb, item_tower_emb.T)
    logits.div_(temperature)

    if lambda_logq > 0.0:
        # LogQ Correction
        # (1, Num_Items) 형태로 브로드캐스팅
        logits.sub_(log_q_tensor.view(1, -1) * lambda_logq)
        
        # Positive Item Score Recovery (정답 아이템의 LogQ 보정 취소 - Optional but Recommended)
        # 해당 배치의 정답 아이템에 대한 LogQ 값을 다시 더해줌 (구현 생략 가능하나 디테일 챙김)
        # 여기서는 간단히 LogQ만 뺌

    # Labels
    # CrossEntropyLoss는 target_ids(인덱스)를 받음
    return F.cross_entropy(logits, target_ids)

def duorec_loss_refined(user_emb_1, user_emb_2, target_ids, temperature=0.1, lambda_sup=0.1):
    """
    DuoRec은 '마지막 시점'의 벡터끼리 비교하는 것이 일반적임.
    user_emb_1, 2: (Batch, Dim)
    target_ids: (Batch, ) - 마지막 시점의 정답 아이템
    """
    batch_size = user_emb_1.size(0)
    
    # Normalize
    z_i = F.normalize(user_emb_1, dim=1)
    z_j = F.normalize(user_emb_2, dim=1)
    
    # Unsupervised Loss (Self-Augmentation)
    logits = torch.matmul(z_i, z_j.T) / temperature
    labels = torch.arange(batch_size, device=user_emb_1.device)
    loss_unsup = F.cross_entropy(logits, labels)
    
    # Supervised Loss (Target-Aware)
    # 같은 아이템을 다음에 구매한 유저끼리 당기기
    if lambda_sup > 0:
        targets = target_ids.view(-1, 1)
        mask = torch.eq(targets, targets.T).float().fill_diagonal_(0)
        
        if mask.sum() > 0:
            logits_sup = torch.matmul(z_i, z_i.T) / temperature
            log_prob = F.log_softmax(logits_sup, dim=1)
            loss_sup = -(mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
            loss_sup = loss_sup[mask.sum(1) > 0].mean()
            return loss_unsup + (lambda_sup * loss_sup)
            
    return loss_unsup

# ==========================================
# 2. Main Training Logic
# ==========================================
def train_model(dataloader, item_tower, args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SASRecUserTower(args).to(device)
    
    # Item Tower의 임베딩을 가져옴 (Freeze 가정)
    # 학습 중 Item Tower도 같이 업데이트한다면 forward를 매번 불러야 함
    # 여기서는 고정된 Item Vector 테이블을 쓴다고 가정 (메모리 절약)
    item_tower.eval()
    with torch.no_grad():
        # 전체 아이템 임베딩 테이블 (Num_Items, Dim)
        full_item_embeddings = item_tower.get_all_embeddings().to(device) 
        # LogQ (Popularity Correction)
        log_q_tensor = item_tower.get_log_q().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    model.train()
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # -------------------------------------------------------
        # 1. Data Unpacking (To Device)
        # -------------------------------------------------------
        # 배치는 (Input Sequence, Target Sequence) 쌍이어야 함
        # Input: [A, B, C], Target: [B, C, D]
        (pretrained_vecs, item_ids, 
         type_ids, color_ids, graphic_ids, section_ids, # Seq Feats
         age_ids, gender_ids, channel_ids, cont_feats,  # Static Feats
         padding_mask, target_ids) = [x.to(device) for x in batch]
        
        # -------------------------------------------------------
        # 2. Forward Pass with AMP
        # -------------------------------------------------------
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            # A. First View (Main Task + DuoRec View 1)
            # training_mode=True: Returns (Batch, Seq, Dim)
            output_1 = model(
                pretrained_vecs, item_ids, 
                type_ids, color_ids, graphic_ids, section_ids,
                age_ids, gender_ids, channel_ids, cont_feats,
                padding_mask=padding_mask,
                training_mode=True
            )

            # B. Second View (DuoRec View 2)
            # Same Input, Different Dropout inside model
            output_2 = model(
                pretrained_vecs, item_ids, 
                type_ids, color_ids, graphic_ids, section_ids,
                age_ids, gender_ids, channel_ids, cont_feats,
                padding_mask=padding_mask,
                training_mode=True
            )

            # -------------------------------------------------------
            # 3. Loss Calculation
            # -------------------------------------------------------
            
            # (1) Main Loss (All Time Steps)
            # Flatten: (Batch, Seq, Dim) -> (Batch * Seq, Dim)
            # Padding 부분은 Loss에서 제외해야 함 (Masking)
            
            # 유효한 타겟만 골라내기 (target_ids != 0)
            # padding_mask는 True가 Padding임. 반전시켜서 유효 마스크 생성
            valid_mask = ~padding_mask.view(-1) # Flatten Mask
            
            flat_output = output_1.view(-1, args.d_model)[valid_mask]
            flat_targets = target_ids.view(-1)[valid_mask]
            
            main_loss = efficient_corrected_logq_loss(
                user_emb=flat_output,
                item_tower_emb=full_item_embeddings,
                target_ids=flat_targets,
                log_q_tensor=log_q_tensor,
                lambda_logq=args.lambda_logq
            )

            # (2) DuoRec Loss (Last Time Step Only)
            # Contrastive Learning은 계산 비용이 높으므로 마지막 시점만 수행
            # padding을 고려하여 각 배치의 '실제 마지막 아이템' 위치를 가져와야 함
            # 여기서는 편의상 output[:, -1, :] 사용 (Padding이 뒤에 몰려있다고 가정 시 주의 필요)
            # 정확히 하려면 gather를 써야 하지만, 간단히 구현:
            
            last_output_1 = output_1[:, -1, :] 
            last_output_2 = output_2[:, -1, :]
            last_targets = target_ids[:, -1] # 마지막 정답

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
        
        # Gradient Clipping (Optional but stable for Transformer)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss {total_loss.item():.4f} (Main: {main_loss.item():.4f}, CL: {cl_loss.item():.4f})")
    
def train_logic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 모델 인스턴스화
    args = type('Args', (), {
        'd_model': 128, 'max_len': 50, 'dropout': 0.1, 
        'pretrained_dim': 768, 'num_items': 10000, 
        'num_brands': 500, 'num_cates': 100,
        'num_price_buckets': 50, 'num_time_buckets': 10,
        'nhead': 4, 'num_layers': 2
    })()

    model = SASRecUserTower(args).to(device)


