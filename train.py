
from typing import Any, Dict
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
import torch
from tqdm import tqdm
from database import ProductInput, SessionLocal
from dependencies import get_global_encoder, get_global_projector
from model import CoarseToFineItemTower, OptimizedItemTower, SimCSEModelWrapper, SimCSERecSysDataset
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_router = APIRouter()

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
    batch_size: int = 16, 
    epochs: int = 5,
    lr: float = 1e-4
):
    print("🚀 Fetching data from DB...")
    
    # 혹시 모를 taskbackground떄문에 일단.
    db_session = SessionLocal()
    
    
    stmt = select(ProductInput.product_id, ProductInput.feature_data)
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
    model.train() # Dropout ON (필수)
    
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
    torch.save(encoder.state_dict(), "encoder_stage1.pth")
    torch.save(projector.state_dict(), "projector_stage2.pth")
    
    # torch.save(model.state_dict(), "final_simcse_model.pth")    


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