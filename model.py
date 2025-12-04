# model.py
import torch
import torch.nn as nn
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