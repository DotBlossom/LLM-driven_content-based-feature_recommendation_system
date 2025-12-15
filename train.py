
from typing import Any, Dict, List
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from requests import Session
from sqlalchemy import select
import torch
from tqdm import tqdm
from utils.util import fetch_training_data_from_db, load_pretrained_vectors_from_db
from database import ProductInferenceInput, SessionLocal, get_db
from utils.dependencies import get_global_encoder, get_global_projector
from model import CoarseToFineItemTower, OptimizedItemTower, SimCSEModelWrapper, SimCSERecSysDataset
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader, Dataset
import os
from model import SymmetricUserTower 
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
    batch_size: int = 4,
    epochs: int = 5,
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
            
            
            # Forward (Cross-Attention)
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
            
            total_loss += loss.item()
            step += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
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
        self.product_id_map = product_id_map # DB ID -> Model Index 변환기
        self.max_seq_len = max_seq_len

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
        target_db_id = row['target']
        target_idx = self.product_id_map.get(target_db_id, 0)
        
        # 3. Profile Data
        gender = row.get('gender', 0)
        age = row.get('age', 0)

        return {
            "history": torch.tensor(mapped_history, dtype=torch.long),
            "target_idx": torch.tensor(target_idx, dtype=torch.long), # 정답 아이템의 Model Index
            "gender": torch.tensor(gender, dtype=torch.long),
            "age": torch.tensor(age, dtype=torch.long)
        }


def train_user_tower_task(
    db_session: Session = Depends(get_db), 
    epochs: int = 5, 
    batch_size: int = 4, 
    lr: float = 1e-4,
    temperature: float = 0.075 # Loss dx 낮음 : low , Loss div : High
):
    print("\n🚀 [Task Started] User Tower Training...")
    
    # 1. Pre-trained Vector 로드 (Lookup Table 준비)
    pretrained_matrix, product_id_map = load_pretrained_vectors_from_db(db_session)
    num_total_products = len(product_id_map)
    
    # 2. 모델 초기화
    model = SymmetricUserTower(
        num_total_products=num_total_products,
        max_seq_len=50,
        input_dim=128
    )
    
    # ⭐ 핵심: 학습된 아이템 벡터 주입 및 동결
    model.load_pretrained_weights(pretrained_matrix, freeze=True)
    model.to(DEVICE)
    model.train() # 학습 모드
    
    # 3. 데이터셋 준비 (Dummy Logic - 실제로는 DB User Log 테이블에서 쿼리해야 함)
    # TODO: 실제 DB에서 유저 로그(UserInteraction)를 긁어오는 로직으로 대체 필요
    print("📊 Fetching user interaction data...")
    
    train_data = fetch_training_data_from_db(db_session, min_interactions=2)
    print(f" 데이터 개수 확인: {len(train_data)}개")
    # 데이터가 너무 적으면 학습 중단 (Safety Check)
    if len(train_data) < batch_size:
        print("⚠️ Warning: Not enough data to train. At least one batch needed.")
       
    
    
    dataset = UserTowerTrainDataset(train_data, product_id_map)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
        )
    
    # 4. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # CrossEntropyLoss를 사용 (In-batch Negative 방식)
    # 정답 라벨은 항상 대각선(0, 1, 2...)이 됨
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    print(f"🔥 Start Training for {epochs} epochs (Temp={temperature})...")
    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress:
            history = batch['history'].to(DEVICE)     # (B, L)
            target_idx = batch['target_idx'].to(DEVICE) # (B,)
            gender = batch['gender'].to(DEVICE)
            age = batch['age'].to(DEVICE)
            
            profile_data = {'gender': gender, 'age': age}
            
            optimizer.zero_grad()
            
            # -----------------------------------------------------------
            # (A) User Vector 생성 (B, 128)
            # -> 이미 모델 내부에서 F.normalize 되어서 나옴 (길이=1)
            # -----------------------------------------------------------
            user_vectors = model(history, profile_data)
            
            # -----------------------------------------------------------
            # (B) Target Item Vector 조회 (B, 128)
            # -> DB에서 온 벡터이므로 이미 정규화 되어 있음 (길이=1)
            # -----------------------------------------------------------
            target_item_vectors = model.item_embedding(target_idx)
            
            # -----------------------------------------------------------
            # (C) Similarity (Logits) Calculation & Scaling [핵심!]
            # -----------------------------------------------------------
            # 내적(Dot Product) 수행 -> 정규화된 벡터끼리의 내적이므로 코사인 유사도임 (-1.0 ~ 1.0)
            # (B, 128) x (128, B) = (B, B) Matrix
            sim_matrix = torch.matmul(user_vectors, target_item_vectors.T)
            
            # [Temperature Scaling]
            # 값의 범위를 -1~1에서 -10~10 (temp=0.1 기준)으로 뻥튀기해줌.
            # 그래야 Softmax가 뾰족해지고(Sharpening), Gradient가 잘 흐름.
            logits = sim_matrix / temperature 
            
            # -----------------------------------------------------------
            # (D) Labeling (In-batch Negative)
            # -----------------------------------------------------------
            # i번째 유저는 i번째 아이템(대각선)이 정답.
            # 나머지는 전부 Negative Sample로 간주.
            labels = torch.arange(batch_size).to(DEVICE)
            
            # -----------------------------------------------------------
            # (E) Loss & Update
            # -----------------------------------------------------------
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
            # 진행바에 현재 Loss 표시
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / steps if steps > 0 else 0
        print(f"   Epoch {epoch+1} Summary | Avg Loss: {avg_loss:.4f}")

    # 5. Save Model
    save_path = os.path.join(MODEL_DIR, "user_tower_symmetric_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training Complete. Model saved to {save_path}")
    
    
    
# ------------------------------------------------------
# Ranker (DCN V2)Task
# ------------------------------------------------------

from torch.utils.data import Dataset, DataLoader

class RankingDataset(Dataset):
    def __init__(self, 
                 interaction_logs,       # [user_id, item_id, label, context] 리스트
                 user_vector_store,      # {user_id: vector(128)} (Redis/Memory)
                 item_vector_store):     # {item_id: vector(128)} (DB/Memory)
        self.logs = interaction_logs
        self.u_store = user_vector_store
        self.i_store = item_vector_store
        
        # OOV(Out of Vocabulary) 방지용 랜덤 벡터 (혹은 평균 벡터)
        self.default_vec = torch.zeros(128) 

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        uid, iid, label, context = self.logs[idx]
        
        # 1. Feature Fetching 
        u_vec = self.u_store.get(uid, self.default_vec)
        i_vec = self.i_store.get(iid, self.default_vec)
        
        # Tensor 변환
        u_tensor = torch.tensor(u_vec, dtype=torch.float32)
        i_tensor = torch.tensor(i_vec, dtype=torch.float32)
        c_tensor = torch.tensor(context, dtype=torch.float32) # 예: 시간 정보 등
        
        # Label (0 or 1)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return u_tensor, i_tensor, c_tensor, label_tensor

def train_ranking_model(
    ranking_model, 
    train_loader, 
    epochs=5, 
    lr=0.001
):
    # 1. Setup
    ranking_model.to(DEVICE)
    optimizer = torch.optim.AdamW(ranking_model.parameters(), lr=lr)
    criterion = nn.BCELoss() # Binary Cross Entropy (출력이 Sigmoid여야 함)
    
    print("🔥 Start Ranking Model Training...")
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total_samples = 0
        
        for u_emb, i_emb, c_emb, labels in tqdm(train_loader):
            u_emb = u_emb.to(DEVICE)
            i_emb = i_emb.to(DEVICE)
            c_emb = c_emb.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1) # (B,) -> (B, 1) 차원 맞춤
            
            # 2. Forward
            optimizer.zero_grad()
            
            # Ranking Model은 (0~1) 사이의 확률값을 뱉음
            preds = ranking_model(u_emb, i_emb, c_emb) 
            
            # 3. Loss Calculation
            loss = criterion(preds, labels)
            
            # 4. Backward
            loss.backward()
            optimizer.step()
            
            # 5. Accuracy Check (0.5 기준)
            total_loss += loss.item()
            predicted_labels = (preds > 0.5).float()
            correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
            
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total_samples
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {accuracy*100:.2f}%")
        
    print("✅ Ranking Training Finished.")