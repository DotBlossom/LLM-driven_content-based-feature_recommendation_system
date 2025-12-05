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
import vocab




# ItemTowerEmbedding(S1) * N -> save..DB -> stage2 (optimizer pass -> triplet)  





# --- Global Configuration (전체 시스템이 참조하는 공통 차원) ---
EMBED_DIM_CAT = 64 # Feature의 임베딩 차원 (Transformer d_model)
OUTPUT_DIM_TRIPLET = 128 # Stage 2 최종 압축 차원
OUTPUT_DIM_ITEM_TOWER = 512 # Stage 1 최종 출력 차원 (Triplet Tower Input)
RE_MAX_CAPACITY = 50000 # <<<<<<<<<<<< RE 토큰의 최대 개수를 미리 할당
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
    Categorical Vector(64d)를 받아 512d로 확장 및 정제
    """
    def __init__(self, input_dim, output_dim=OUTPUT_DIM_ITEM_TOWER):
        super().__init__()
        
        # 1. 확장 차원 정의: 입력 64d -> 4배 확장 (256d)
        hidden_dim = input_dim * 4  # 64 * 4 = 256
        
        # 1. Expansion (확장)
        self.expand = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # 64 -> 256
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 2. Compression & Residuals (압축 및 심화)
        self.deep_layers = nn.Sequential(
            # Layer 1: 차원 축소 시작 (256 -> 128)
            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            
            # Layer 2 & 3: Residual Blocks (128차원 유지하며 깊이 추가)
            ResidualBlock(hidden_dim // 2),
            ResidualBlock(hidden_dim // 2), 
            
            # Layer 4: 최종 출력 차원 맞추기 (128 -> 512)
            nn.Linear(hidden_dim // 2, output_dim) 
        )
        
    def forward(self, x):
        x = self.expand(x)
        x = self.deep_layers(x)
        return x

# ----------------------------------------------------------------------
# 3. Main Model: CoarseToFineItemTower (Stage 1)
# ----------------------------------------------------------------------
class CoarseToFineItemTower(nn.Module):
    """
    [Item Tower]: Standard/Reinforced 피쳐를 융합하고 512차원 벡터 생성.
    vocab.py의 이중 어휘 구조와 호환되도록 수정되었습니다.
    """
    def __init__(self, embed_dim=EMBED_DIM_CAT, nhead=4, output_dim=OUTPUT_DIM_ITEM_TOWER):
        super().__init__()
        
        # 1. vocab.py에서 STD와 RE의 분리된 어휘 크기를 가져옵니다.
        std_vocab_size, re_vocab_size = vocab.get_vocab_sizes()
        
        # A. Dual Embedding (64d)
        # 단일 임베딩 대신, 분리된 어휘 크기를 사용합니다.
        self.std_embedding = nn.Embedding(std_vocab_size, embed_dim, padding_idx=vocab.PAD_ID)
        self.re_embedding = nn.Embedding(RE_MAX_CAPACITY, embed_dim, padding_idx=vocab.PAD_ID)
        # B. Self-Attention Encoders (d_model=64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.std_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.re_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # C. Cross-Attention (d_model=64, nhead=4)
        # 이 레이어는 Q=STD, K/V=RE로 사용될 것입니다.
        # (수정됨) Shape Vector (128d)가 제거되어 입력은 64d가 됨.
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # D. Deep Residual Head (입력 차원: embed_dim = 64)
        head_input_dim = embed_dim
        self.head = DeepResidualHead(input_dim=head_input_dim, output_dim=output_dim)

    def forward(self, std_input: torch.Tensor, re_input: torch.Tensor) -> torch.Tensor:
        # 1. 임베딩 (STD와 RE 분리 처리)
        std_embed = self.std_embedding(std_input)
        re_embed = self.re_embedding(re_input)
        
        # 2. Self-Attention Encoders
        std_output = self.std_encoder(std_embed) # Shape: (B, L_std, D)
        re_output = self.re_encoder(re_embed)   # Shape: (B, L_re, D)
        
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
        # Deep Head Pass (64 -> 512)
        final_vector = self.head(pooled_output)
        
        return final_vector
    

# ----------------------------------------------------------------------
# 4. OptimizedItemTower (Stage 2 Adapter - Triplet Training)
# ----------------------------------------------------------------------

class OptimizedItemTower(nn.Module):
    """
    [Optimization Tower]: Stage 1의 512차원 벡터를 받아 Triplet Loss로 128차원으로 압축.
    """
    def __init__(self, input_dim=OUTPUT_DIM_ITEM_TOWER, output_dim=OUTPUT_DIM_TRIPLET):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )
        
    def forward(self, x):
        # [Log 1] 입력 데이터 확인
        if not self.training: # 추론(eval) 모드일 때만 로그 출력 (학습 땐 너무 많음)
            print(f"\n  [Model Internal] Input Vector Shape: {x.shape}")
            print(f"  [Model Internal] Input Sample (First 5): {x[0, :5].detach().cpu().numpy()}")

        # 레이어 통과
        x = self.layer(x)
        
        # [Log 2] 압축 후 데이터 확인
        if not self.training:
            print(f"  [Model Internal] After Linear Layer Shape: {x.shape}")

        # 정규화 (L2 Normalization)
        x = F.normalize(x, p=2, dim=1)
        
        # [Log 3] 정규화 확인 (Norm이 1.0에 가까운지)
        if not self.training:
            norm_check = torch.norm(x, p=2, dim=1).mean().item()
            print(f"  [Model Internal] Output Normalized Shape: {x.shape} | Avg Norm: {norm_check:.4f} (Expected ~1.0)")
            
        return x
    
    
# ----------------------------------------------------------------------
# 5. Dataset & Sampler & Training Function (Stage 2 Logic) / first INPUT from DB
# ----------------------------------------------------------------------

class RichAttributeDataset(Dataset):
    """
    Stage 2 학습을 위한 데이터셋. 입력은 Stage 1의 최종 출력(512d) 벡터입니다.
    """
def __init__(self, product_list):
        # 1. 임시 저장소 (가벼운 Python List 사용)
        temp_vectors = []
        self.fine_labels = []
        self.coarse_labels = []
        self.label_to_id = {}
        
        # 2. Loop: 데이터 정제 및 라벨링 (여기서는 Tensor 변환 금지!)
        for item in product_list:
            # --- Vector 처리 ---
            vec = item.get('vector')
            if vec is None:
                # Mock vector (리스트 형태)
                vec = np.random.randn(512).astype(np.float32).tolist()
            
            # 여기서 torch.tensor(vec)를 하지 않고, 그냥 리스트(혹은 numpy) 상태로 둡니다.
            temp_vectors.append(vec)
            
            # --- Label 처리 ---
            full_cat = item['clothes']['category'][0]
            coarse_cat = full_cat[:2] # 예: "top" from "top/tee"
            
            if full_cat not in self.label_to_id:
                self.label_to_id[full_cat] = len(self.label_to_id)
            
            self.fine_labels.append(self.label_to_id[full_cat])
            self.coarse_labels.append(coarse_cat)

        # 3. [핵심 최적화] Bulk Conversion (한방에 텐서화)
        # 리스트의 리스트 -> 하나의 거대한 FloatTensor (N, 512)
        # 이 방식이 메모리를 훨씬 적게 쓰고, 연산 속도도 빠릅니다.
        self.data = torch.tensor(temp_vectors, dtype=torch.float32)
        
        # 라벨도 마찬가지로 LongTensor로 한 번에 변환
        self.fine_labels = torch.tensor(self.fine_labels, dtype=torch.long)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # 인덱싱 속도도 훨씬 빠름 (텐서 슬라이싱)
            return self.data[idx], self.fine_labels[idx]


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
    
    distance = distances.CosineSimilarity()
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance)
    mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")

    model.train()
    history = []
    
    for epoch in range(epochs):
        total_loss = 0
        triplets_count = 0
        
        for batch_vecs, batch_labels in dataloader:
            batch_vecs = batch_vecs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            embeddings = model(batch_vecs)
            indices_tuple = mining_func(embeddings, batch_labels)
            loss = loss_func(embeddings, batch_labels, indices_tuple)
            
            if loss > 0:
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                triplets_count += mining_func.num_triplets
            
        log = f"Epoch {epoch+1}: Loss={total_loss:.4f}, Valid Triplets={triplets_count}"
        print(log)
        history.append(log)

    torch.save(model.state_dict(), save_path)
    return history


# serving APUI nneeded
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