
from typing import Any, Dict, List
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from requests import Session
from sqlalchemy import select
import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from database import ProductInferenceInput, SessionLocal, get_db
from utils.dependencies import get_global_batch_size, get_global_encoder, get_global_projector
from model import CoarseToFineItemTower, FinalUserTower, OptimizedItemTower, SimCSEModelWrapper, SimCSERecSysDataset
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader, Dataset
import os
#from model import SymmetricUserTower 
import torch.optim as optim



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_router = APIRouter()
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)



# ------------------------------------------------------
# Item Tower Training Task
# ------------------------------------------------------

class TrainingItem(BaseModel):

    product_id: int
    feature_data: Dict[str, Any]

def collate_simcse(batch):
    from APIController.serving_controller import preprocess_batch_input
    """(View1, View2) 리스트 -> Tensor 변환"""
    view1_list = [item[0] for item in batch]
    view2_list = [item[1] for item in batch]
    
    t_std1, t_re1 = preprocess_batch_input(view1_list)
    t_std2, t_re2 = preprocess_batch_input(view2_list)
    
    return t_std1, t_re1, t_std2, t_re2


## 메모리 최적화: db_session.query(Model).all() 대신 select(...).mappings().all()을 사용하여 딕셔너리로 데이터를 로드하세요


def train_simcse_from_db(    
    encoder: nn.Module,       
    projector: nn.Module,
    batch_size: int = Depends(get_global_batch_size),
    epochs: int = 20,
    lr: float = 1e-4
):
    print("🚀 Fetching data from DB...")
    
    # 혹시 모를 taskbackground떄문에 일단.
    db_session = SessionLocal()
    
    
    stmt = select(ProductInferenceInput.product_id, ProductInferenceInput.feature_data)
    result = db_session.execute(stmt).mappings().all()
    
    if not result:
        print("❌ No data found.")
        return

    # [수정 2] Dictionary -> Pydantic 변환
    products_list = []
    for row in result:
        # row['feature_data'] 접근
        f_data = row['feature_data']
        p_input = TrainingItem(
            product_id=row['product_id'],
            feature_data=f_data
        )
        products_list.append(p_input)
        
    print(f"✅ Loaded {len(products_list)} items.")
    
    # 3. 모델 설정
    model = SimCSEModelWrapper(encoder, projector).to(DEVICE)
    model.train() 
    
    # Optimizer는 두 모델의 파라미터를 모두 학습해야 함
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    

    
    # Loss Function (Contrastive Learning)
    loss_func = losses.NTXentLoss(temperature=0.07)
    
    dataset = SimCSERecSysDataset(products_list, dropout_prob=0.2)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=collate_simcse,
        drop_last=True
    )

    # [추가] 스케줄러 설정 (Warmup 10%)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    
    print("🔥 Starting Training Loop...")
    
    # 5. Training Loop
    for epoch in range(epochs):
        total_loss = 0
        step = 0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for t_std1, t_re1, t_std2, t_re2 in progress:

            t_std1, t_re1 = t_std1.to(DEVICE), t_re1.to(DEVICE)
            t_std2, t_re2 = t_std2.to(DEVICE), t_re2.to(DEVICE)
            
            optimizer.zero_grad()
            
            
            # Forward 
            emb1 = model(t_std1, t_re1)
            emb2 = model(t_std2, t_re2)
            
            # Contrastive Loss Calculation
            embeddings = torch.cat([emb1, emb2], dim=0)
            
            # Label generation
            # 배치 사이즈만큼 0~N 라벨을 만들고 두 번 반복
            batch_curr = emb1.size(0)
            labels = torch.arange(batch_curr).to(DEVICE)
            labels = torch.cat([labels, labels], dim=0)
            
            loss = loss_func(embeddings, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            step += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
        if epochs % 10 == 0:
            print(f"Epoch {epoch+1} Avg Loss: {total_loss/step:.4f}")
        
    print("Training Finished.")
    

    
    print("💾 Saving models...")
    torch.save(encoder.state_dict(), os.path.join(MODEL_DIR, "encoder_stage1.pth"))
    torch.save(projector.state_dict(), os.path.join(MODEL_DIR, "projector_stage2.pth"))
    
    # torch.save(model.state_dict(), "final_simcse_model.pth")    



#DB에 있는 Item load -> positives(dropout) item 증강 -> collate가서 피쳐 토크나이저 하고 텐서화
#이후 텐서 아이템타워가서 trnsf-> std, re cross att 하고 진행
#ProductItem(DB) -> ItemTower(1차아이템텐서) -> opt tensor 학습 (1차학습)  
@train_router.post("/run")
def test_line(

    encoder_instance: CoarseToFineItemTower = Depends(get_global_encoder), 
    projector_instance: OptimizedItemTower = Depends(get_global_projector)
):
    
    train_simcse_from_db(
        encoder=encoder_instance,
        projector=projector_instance
    )
    
    return {"message": "SimCSE training task initiated and completed."}




# ------------------------------------------------------
# User Tower Training Task
# ------------------------------------------------------


class UserTowerTrainDataset(Dataset):
    def __init__(self, 
                 user_data_list: List[dict], 
                 product_id_map: Dict[int, int],
                 max_seq_len: int = 50):
        """
        user_data_list: [
            {'history': [101, 102], 'target': 103, 'gender': 1, 'age': 2}, ...
        ]
        """
        self.data = user_data_list
        self.max_seq_len = max_seq_len
        self.product_id_map = product_id_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        
        # 1. History ID Mapping & Padding
        raw_history = row['history']
        mapped_history = [self.product_id_map.get(pid, 0) for pid in raw_history] # 없으면 0(PAD)
        
        # 시퀀스 길이 맞추기 (Truncate or Pad)
        seq_len = len(mapped_history)
        if seq_len > self.max_seq_len:
            mapped_history = mapped_history[-self.max_seq_len:] # 최근 것만 유지
        else:
            mapped_history = mapped_history + [0] * (self.max_seq_len - seq_len) # 뒤에 0 채움

        # 2. Target Item Mapping
        target_db_id = row['target_idx'] 
        target_idx = self.product_id_map.get(target_db_id, 0)
        
        # 3. Profile Data
        gender = row.get('gender', 0)
        age = row.get('age', 0)
        season = row.get('season', 0)
        

        return {
            "history": torch.tensor(mapped_history, dtype=torch.long),
            "target_idx": torch.tensor(target_idx, dtype=torch.long), # 정답 아이템의 Model Index
            "gender": torch.tensor(gender, dtype=torch.long),
            "age": torch.tensor(age, dtype=torch.long),
            "season": torch.tensor(season, dtype=torch.long)
        }


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. Dataset 정의
# ==========================================
class UserSessionDataset(Dataset):
    def __init__(self, 
                 user_sessions: list,   # [{'history':[], 'season':0, 'gender':0, 'target_item_id':10}, ...]
                 max_len: int = 50):
        self.data = user_sessions
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        
        # History Padding (Pre-padding or Post-padding)
        # 보통 Transformer는 Post-padding + Masking을 쓰지만, 여기서는 0이 Pad ID라고 가정
        history = row['history']
        if len(history) > self.max_len:
            history = history[-self.max_len:] # 최근거만
        else:
            history = history + [0] * (self.max_len - len(history))
            
        return {
            'history': torch.tensor(history, dtype=torch.long),
            'season': torch.tensor(row['season'], dtype=torch.long),
            'gender': torch.tensor(row['gender'], dtype=torch.long),
            'target_item_id': torch.tensor(row['target_item_id'], dtype=torch.long),
            # 만약 Item Tower가 Feature를 입력받아야 한다면 여기에 item_features도 포함되어야 함
            # 여기서는 편의상 ID로 Item Tower에서 벡터를 룩업한다고 가정
        }

# ==========================================
# 2. In-batch Negative Loss (Contrastive)
# ==========================================
class InfoNCELoss(nn.Module):
    """
    배치 내의 다른 샘플들을 Negative로 활용하는 효율적인 Loss
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, user_vectors, item_vectors):
        """
        user_vectors: (Batch, Dim)
        item_vectors: (Batch, Dim) - Positive Pairs
        """
        # Similarity Matrix: (Batch, Batch)
        # (B, D) @ (D, B) -> (B, B)
        scores = torch.matmul(user_vectors, item_vectors.T)
        
        # Scaling
        scores = scores / self.temperature
        
        # Labels: 대각선(Diagonal)이 정답 (0번째 유저는 0번째 아이템이 정답)
        batch_size = user_vectors.size(0)
        labels = torch.arange(batch_size).to(user_vectors.device)
        
        loss = self.criterion(scores, labels)
  
        return loss
    
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader





DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_final_user_tower(
    user_tower: FinalUserTower,
    pretrained_item_matrix: torch.Tensor, # Loss 계산용 (Target/Teacher)
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-4,
):
    # 1. Setup
    user_tower.to(DEVICE)
    pretrained_item_matrix = pretrained_item_matrix.to(DEVICE)
    
    optimizer = optim.AdamW(user_tower.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = InfoNCELoss(temperature=0.07).to(DEVICE)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, 
        steps_per_epoch=len(train_loader), epochs=epochs
    )

    print(f"🚀 Start Training FinalUserTower on {DEVICE}...")
    user_tower.train()
    
    for epoch in range(epochs):
        total_loss = 0
        step = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            # 2. Input Data Preparation
            history = batch['history'].to(DEVICE)       # (Batch, Seq)
            season = batch['season'].to(DEVICE)         # (Batch, )
            gender = batch['gender'].to(DEVICE)         # (Batch, )
            
            target_idx = batch['target_idx'].to(DEVICE) # (Batch, ) - 정답 아이템 Index
            
            # -----------------------------------------------------------
            # A. Ground Truth (Target Item Vectors)
            # -----------------------------------------------------------
            # 미리 계산된 아이템 행렬에서 정답 벡터를 직접 가져옴 (Teacher)
            # pretrained_item_matrix: (Total_Items, 128)
            with torch.no_grad():
                target_item_vectors = pretrained_item_matrix[target_idx]
                # 타겟 벡터도 정규화되어 있는지 확인 (Model이 Normalize를 쓴다면 여기도 해야 함)
                target_item_vectors = F.normalize(target_item_vectors, p=2, dim=1)

            # -----------------------------------------------------------
            # B. User Representation (Student)
            # -----------------------------------------------------------
            # [Call] FinalUserTower.forward(history_ids, season_idx, gender_idx)
            user_vectors = user_tower(history, season, gender)
            
            # -----------------------------------------------------------
            # C. Contrastive Loss
            # -----------------------------------------------------------
            loss = loss_fn(user_vectors, target_item_vectors)
            
            # D. Optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(user_tower.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            step += 1
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        print(f"📊 Epoch {epoch+1} Avg Loss: {total_loss / step:.4f}")
        
    print("✅ Training Finished.")
    return user_tower

