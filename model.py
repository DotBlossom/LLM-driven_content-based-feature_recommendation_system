# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from pytorch_metric_learning import losses, miners, distances
from collections import defaultdict
import random
import os

# --- 1. Dataset & Sampler ---
class RichAttributeDataset(Dataset):
    def __init__(self, product_list):
        self.data = []
        self.fine_labels = []
        self.coarse_labels = []
        self.label_to_id = {}
        
        for item in product_list:
            # 실제 서비스에선 item['vector']가 리스트 형태로 들어온다고 가정
            # 여기선 테스트를 위해 vector가 없으면 랜덤 생성
            vec = item.get('vector')
            if vec is None:
                vec = torch.randn(512)
            else:
                vec = torch.tensor(vec, dtype=torch.float32)
                
            self.data.append(vec)
            
            full_cat = item['clothes']['category'][0]
            coarse_cat = full_cat.split('_')[0]
            
            if full_cat not in self.label_to_id:
                self.label_to_id[full_cat] = len(self.label_to_id)
            
            self.fine_labels.append(self.label_to_id[full_cat])
            self.coarse_labels.append(coarse_cat)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.fine_labels[idx]

class HierarchicalBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, samples_per_class=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.structure = defaultdict(lambda: defaultdict(list))
        
        for idx, (fine_id, coarse_name) in enumerate(zip(dataset.fine_labels, dataset.coarse_labels)):
            self.structure[coarse_name][fine_id].append(idx)
        self.coarse_keys = list(self.structure.keys())

    def __iter__(self):
        num_batches = len(self.dataset) // self.batch_size
        for _ in range(num_batches):
            batch_indices = []
            target_coarse = random.choice(self.coarse_keys)
            fine_dict = self.structure[target_coarse]
            available_fine_labels = list(fine_dict.keys())
            
            if len(available_fine_labels) < (self.batch_size // self.samples_per_class):
                continue # 데이터 부족시 스킵 (단순화)

            num_classes_needed = self.batch_size // self.samples_per_class
            selected_fines = random.choices(available_fine_labels, k=num_classes_needed)
            
            for f_label in selected_fines:
                indices = fine_dict[f_label]
                selected_indices = random.choices(indices, k=self.samples_per_class)
                batch_indices.extend(selected_indices)
            
            yield batch_indices

    def __len__(self):
        return len(self.dataset) // self.batch_size

# --- 2. Model Definition ---
class OptimizedItemTower(nn.Module):
    def __init__(self, input_dim=512, output_dim=128):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
        )
        
    def forward(self, x):
        x = self.layer(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)

# --- 3. Training Function ---
def train_model(product_list, epochs=5, batch_size=32, save_path="models/final_optimized_adapter.pth"):
    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = RichAttributeDataset(product_list)
    
    # 데이터셋이 너무 작으면 에러 방지를 위해 기본 DataLoader 사용
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

# --- 4. Inference Helper ---
def load_and_infer(input_vector, model_path="models/final_optimized_adapter.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptimizedItemTower().to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found. Train first.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        tensor_in = torch.tensor([input_vector], dtype=torch.float32).to(device)
        output = model(tensor_in)
        
    return output.cpu().numpy().tolist()[0]



# --- ItemTowerBlock ---


class ResidualBlock(nn.Module):
    """
    유튜브 논문의 ReLU 층을 현대적으로 해석한 ResNet 블록
    Input 차원과 Output 차원이 같을 때 사용
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x + block(x) -> 잔차 연결 (정보 보존 + 기울기 소실 방지)
        return x + self.block(x)

class DeepPyramidTower(nn.Module):
    """
    깔때기(Funnel) 모양으로 차원을 줄여나가는 Deep MLP
    """
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        
        # 1. 확장 단계 (Expansion): 정보를 풍부하게 펼침
        # 유튜브 논문에서도 첫 레이어는 충분히 넓게 잡는 것을 권장함
        hidden_dim = input_dim * 4  # 예: 64 -> 256
        
        self.expansion = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 2. 압축 단계 (Compression with Residuals)
        # 차원을 점진적으로 줄여나감 (256 -> 128)
        # 여기서는 간단히 한 번 줄이고 ResBlock을 통과시키는 구조 예시
        
        self.deep_layers = nn.Sequential(
            # Layer 1: 차원 축소
            nn.Linear(hidden_dim, hidden_dim // 2), # 256 -> 128
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            
            # Layer 2: Residual Block (깊이감 추가)
            # 차원이 유지되므로 Skip Connection 가능
            ResidualBlock(hidden_dim // 2),
            ResidualBlock(hidden_dim // 2), # 층을 더 깊게 쌓고 싶으면 추가
            
            # Layer 3: 최종 출력 차원 맞추기 (만약 위에서 안 맞았다면)
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        x = self.expansion(x)
        x = self.deep_layers(x)
        return x






class CoarseToFineItemTower(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, nhead=4, output_dim=512):
        super().__init__()
        
        # A. Shared Embedding
        # 0번은 PAD, 나머지 토큰들은 공유 임베딩 사용
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # B. Self-Attention Encoders (Context 파악)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.std_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.re_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # C. Cross-Attention (Refinement)
        # Query: Standard(뼈대), Key/Value: Reinforced(살)
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # D. Deep Residual Head (고도화된 부분!)
        # self.head = DeepResidualHead(input_dim=embed_dim, output_dim=output_dim)
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim) 
        )

    def forward(self, std_ids, re_ids):
        """
        std_ids: (Batch, Seq_Len1)
        re_ids: (Batch, Seq_Len2)
        """
        # 1. Embedding & Self-Encode
        v_std = self.embedding(std_ids) # (B, S1, E)
        v_re = self.embedding(re_ids)   # (B, S2, E)
        
        v_std = self.std_encoder(v_std)
        v_re = self.re_encoder(v_re)
        
        # 2. Cross-Attention (Standard가 Reinforced의 정보를 흡수)
        # attn_output: (B, S1, E) - v_std와 같은 shape
        attn_output, _ = self.cross_attn(query=v_std, key=v_re, value=v_re)
        
        # 3. Residual & Norm (안전장치)
        # Reinforced 정보가 이상해도 원본 Standard 정보는 보존됨
        v_refined = self.layer_norm(v_std + attn_output)
        
        # 4. Pooling (Sequence -> Vector)
        # 평균을 내서 하나의 상품 벡터로 압축
        v_final = v_refined.mean(dim=1) # (B, E)
        
        # 5. Deep Head Pass
        # 64차원 -> 256차원(ResBlock) -> 512차원
        output = self.head(v_final)
        
        # 6. L2 Normalization Check 

        return F.normalize(output, p=2, dim=1)


