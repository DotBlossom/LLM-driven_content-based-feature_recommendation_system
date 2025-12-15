


"""

class ResidualBlock(nn.Module):

    def __init__(self, dim, dropout=0.2):
        super().__init__()
        # 블록 내에서 차원을 유지하는 2개의 Linear Layer (Skip Connection 전 처리)
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.GELU = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        # x + block(x) -> 잔차 연결 (핵심!)
        return self.GELU(residual + out)

# --- Deep Residual Head (Pyramid Funnel) ---
class DeepResidualHead(nn.Module):

    def __init__(self, input_dim, output_dim=OUTPUT_DIM_ITEM_TOWER):
        super().__init__()
        
        # 1. 내부 확장 (Expansion): 표현력을 위해 4배 확장은 유지 (64 -> 256)
        hidden_dim = input_dim * 4 
        
        self.expand = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
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
        if not self.training:
            # (B, 128)
            final_sample = x[0, :6].detach().cpu().numpy()
            print(f"[Head DEBUG] D. Final Output (B, {x.shape[1]}): {final_sample}")
        return x









        # [STD_1, STD_2, ..., RE_1, RE_2, ...]
        combined_seq = torch.cat([std_token, re_token], dim=1) # (B, 2*F, D)
        
        # --- [Logic 3] Transformer & Pooling ---
        # PAD Masking: 여기서는 간단히 생략 (SimCLR 특성상 Noise도 정보가 됨)
        # 정교하게 하려면 src_key_padding_mask 추가 가능
        
        context_out = self.transformer(combined_seq) # (B, 2*F, D)
        
        # Mean Pooling (Flatten 대신 사용 -> 필드 수 변화에 강인함)
        pooled = context_out.mean(dim=1) # (B, D)
    
        if not self.training:
            # 첫 번째 샘플의 처음 6개 값만 출력
            pooled_sample = pooled[0, :6].detach().cpu().numpy()
            print(f"DEBUG: Pooled Vector (h) Sample (1st 6 values): {pooled_sample}")
        
        return self.head(pooled) # (B, 128)
    


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

        #Args:
        #    history_ids (Tensor): (B, L_hist) - 아이템 ID 시퀀스
        #    concept_input (Tensor): (B, L_txt) - 장바구니 컨셉 텍스트 토큰
        #    profile_features (Tensor): (B, 2) - [Normalized Height, Normalized Weight]
        #                              예: [[1.2, -0.5], [0.0, 0.8], ...] (Z-score 권장)

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

class CoarseToFineItemTower(nn.Module):

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
    








class RichAttributeDataset(Dataset):
    
    #Stage 2 학습을 위한 데이터셋. 입력은 Stage 1의 128 pre-vector
    
def __init__(self, product_list):
        # 1. 임시 저장소 (가벼운 Python List 사용)
        temp_vectors = []
        temp_ids = []  # <--- ID 저장을 위한 리스트
        self.fine_labels = []
        self.coarse_labels = []
        self.label_to_id = {}
        
        # 2. Loop: 데이터 정제 및 라벨링 (여기서는 Tensor 변환 금지!)
        for item in product_list:
            # --- Vector 처리 ---
            vec = item.get('vector')
            p_id = item.get('id') # ID 추출
            
            if vec is None:
                # Mock vector (리스트 형태)
                vec = np.random.randn(512).astype(np.float32).tolist()
            
            # 여기서 torch.tensor(vec)를 하지 않고, 그냥 리스트(혹은 numpy) 상태로 둡니다.
            temp_vectors.append(vec)
            temp_ids.append(p_id)
            
            # --- Label 처리 ---, DTO 주의
            full_cat = item['clothes']['category'][0]
            coarse_cat = full_cat[:2] # 예: "top" from "top/tee"
            
            if full_cat not in self.label_to_id:
                self.label_to_id[full_cat] = len(self.label_to_id)
            
            self.fine_labels.append(self.label_to_id[full_cat])
            self.coarse_labels.append(coarse_cat)

        # 3. [핵심 최적화] Bulk Conversion (한방에 텐서화)
        # 리스트의 리스트 -> 하나의 거대한 FloatTensor 
        # 이 방식이 메모리를 훨씬 적게 쓰고, 연산 속도도 빠릅니다.
        self.data = torch.tensor(temp_vectors, dtype=torch.float32)
        self.ids = torch.tensor(temp_ids, dtype=torch.long) 
        
        # 라벨도 마찬가지로 LongTensor로 한 번에 변환
        self.fine_labels = torch.tensor(self.fine_labels, dtype=torch.long)

        print(f"[Dataset] Created tensors. Shape: {self.data.shape}")
        
        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # 인덱싱 속도도 훨씬 빠름 (텐서 슬라이싱)
            return self.data[idx], self.fine_labels[idx], self.ids[idx]

class HierarchicalBatchSampler(Sampler):
    # ... (Sampler 클래스 정의 유지) ...
    def __init__(self, dataset, batch_size, samples_per_class=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.structure = defaultdict(lambda: defaultdict(list))
        self.all_indices = []
        
        for idx, (fine_id, coarse_name) in enumerate(zip(dataset.fine_labels, dataset.coarse_labels)):
            self.structure[coarse_name][fine_id].append(idx)
            self.all_indices.append(idx)
            
        self.coarse_keys = list(self.structure.keys())

    def __iter__(self):
        num_batches = len(self.dataset) // self.batch_size
        
        for _ in range(num_batches):
            batch_indices = []
            target_coarse = random.choice(self.coarse_keys)
            fine_dict = self.structure[target_coarse]
            available_fine_labels = list(fine_dict.keys())
            num_classes_needed = self.batch_size // self.samples_per_class
            
            # A. 충분한 소분류가 있는 경우 (Hard Negative Mode)
            if len(available_fine_labels) >= num_classes_needed:
                selected_fines = random.sample(available_fine_labels, k=num_classes_needed)
                
                for f_label in selected_fines:
                    indices = fine_dict[f_label]
                    selected_indices = random.choices(indices, k=self.samples_per_class)
                    batch_indices.extend(selected_indices)
            
            # B. 소분류가 부족한 경우 (Fallback: Noise Mixing Mode)
            else:
                for f_label in available_fine_labels:
                    indices = fine_dict[f_label]
                    selected_indices = random.choices(indices, k=self.samples_per_class)
                    batch_indices.extend(selected_indices)
                
                remaining_slots = self.batch_size - len(batch_indices)
                if remaining_slots > 0:
                    noise_indices = random.choices(self.all_indices, k=remaining_slots)
                    batch_indices.extend(noise_indices)
            
            yield batch_indices

    def __len__(self):
        return len(self.dataset) // self.batch_size

def train_model(product_list, epochs=5, batch_size=32, save_path="models/final_optimized_adapter.pth"):
    # ... (train_model 함수 로직 유지) ...
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = RichAttributeDataset(product_list)
    
    if len(dataset) < batch_size:
        dataloader = DataLoader(dataset, batch_size=len(dataset))
    else:
        sampler = HierarchicalBatchSampler(dataset, batch_size)
        dataloader = DataLoader(dataset, batch_sampler=sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptimizedItemTower().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    ###### Triplet Logic
    
    #distance = distances.CosineSimilarity()
    #loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance)
    #mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")

    #####
    
    
    # infoNCE / Temperature (0.07 ~ 0.1 추천): 낮을수록 Hard Negative에 집중, 높을수록 부드럽게 학습
    loss_func = losses.NTXentLoss(temperature=0.093) 
    
    model.train()
    history = []
    
    for epoch in range(epochs):
        total_loss = 0
        triplets_count = 0
        
        for batch_vecs, batch_labels, _ in dataloader:
            batch_vecs = batch_vecs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            embeddings = model(batch_vecs)
            #indices_tuple = mining_func(embeddings, batch_labels)
            loss = loss_func(embeddings, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        log = f"Epoch {epoch+1}/{epochs}: Avg Loss={avg_loss:.4f}"
        print(log)
        history.append(log)

    torch.save(model.state_dict(), save_path)
    return history
    
    
    
    
    # vector_triplet에 저장, real cos
def load_and_infer(input_vector, model_path="models/final_optimized_adapter.pth"):
    print("="*60)
    print(f"[Inference Step 1] Initializing Inference Pipeline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference Step 2] Target Device: {device}")
    
    # 모델 초기화
    model = OptimizedItemTower().to(device)
    
    # 모델 가중치 로드
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found at: {model_path}")
        raise FileNotFoundError("Model file not found. Train first.")
    
    print(f"[Inference Step 3] Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 평가 모드 전환 (Dropout, Batchnorm 고정)
    model.eval()
    print(f"[Inference Step 4] Model set to EVAL mode.")
    
    with torch.no_grad():
        # 입력 텐서 변환
        tensor_in = torch.tensor([input_vector], dtype=torch.float32).to(device)
        print(f"[Inference Step 5] Input Tensor created on {device}. Shape: {tensor_in.shape}")
        
        # 모델 Forward 실행 (여기서 위에서 정의한 [Model Internal] 로그가 찍힘)
        print("-" * 30 + " Model Forward Start " + "-" * 30)
        output = model(tensor_in)
        print("-" * 30 + " Model Forward End " + "-" * 30)
        
    # 결과 반환
    result_vector = output.cpu().numpy().tolist()[0]
    print(f"[Inference Step 6] Final Output Vector generated. Length: {len(result_vector)}")
    print("="*60)
    
    return result_vector
    """
