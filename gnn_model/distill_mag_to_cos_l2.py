import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MagnitudeEncoder(nn.Module):
    def __init__(self, input_dim=64, output_dim=64, hidden_dim=128):
        """
        Args:
            input_dim: LightGCL 임베딩 차원 (예: 64)
            output_dim: 변환할 차원 (FAISS 등에 맞춤, 보통 같거나 큼)
            hidden_dim: 정보를 섞기 위해 잠시 차원을 늘림
        """
        super(MagnitudeEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # 1. 차원 확장 & 비선형성 추가
            # 크기(Length) 정보를 좌표(Coordinate) 정보로 '접어서' 넣기 위함
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2), # ReLU보다 정보 손실이 적은 Leaky 권장
            nn.Dropout(0.1),
            
            # 2. 다시 원래 차원(또는 타겟 차원)으로 압축
            nn.Linear(hidden_dim, output_dim)
        )
        
        # [중요] Cosine Similarity의 한계(-1~1)를 극복하기 위한 학습 가능한 스케일 파라미터
        # Dot Product는 10, 20까지 가는데 Cosine은 1이 최대라서,
        # 이 logit_scale을 곱해서 범위를 맞춰줌.
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x):
        x = self.encoder(x)
        # 3. 무조건 크기를 1로 고정 (L2 Normalize)
        # 이제 모든 정보는 '방향(Angle)'에만 남아있음
        return F.normalize(x, p=2, dim=1)
    
    
    
    
def train_projector(lightgcl_model, dataloader, device, epochs=5):
    """
    LightGCL(Teacher) -> Projector(Student) 지식 증류 학습
    """
    # 1. 모델 설정
    lightgcl_model.eval() # 선생님은 고정 (평가 모드)
    
    # User용, Item용 Projector를 따로 만들거나 하나를 공유해도 됨 (여기선 공유)
    projector = MagnitudeEncoder(input_dim=64, output_dim=64).to(device)
    optimizer = torch.optim.Adam(projector.parameters(), lr=0.001)
    
    # Teacher의 User/Item 임베딩 가져오기 (고정된 텐서)
    # 메모리 절약을 위해 detach()
    src_user_emb = lightgcl_model.embedding_user.weight.detach()
    src_item_emb = lightgcl_model.embedding_item.weight.detach()

    print("🚀 Start Projector Distillation...")
    
    for epoch in range(epochs):
        projector.train()
        total_loss = 0
        
        # tqdm 등 사용 가능
        for batch_users, batch_pos_items, _ in dataloader:
            batch_users = batch_users.to(device)
            batch_pos_items = batch_pos_items.to(device)
            
            # -----------------------------------------------------------
            # A. Teacher (LightGCL) - 정답지 생성
            # -----------------------------------------------------------
            # 내적(Dot Product) 사용 -> 크기(Magnitude) 정보가 점수에 반영됨
            # 값이 -inf ~ +inf 범위 (예: 12.5)
            with torch.no_grad():
                u_tea = src_user_emb[batch_users]
                i_tea = src_item_emb[batch_pos_items]
                # (Batch,)
                scores_teacher = torch.sum(u_tea * i_tea, dim=1)

            # -----------------------------------------------------------
            # B. Student (Projector) - 따라하기
            # -----------------------------------------------------------
            # Projector 통과 -> 크기가 1로 바뀜 (L2 Norm)
            # 코사인 유사도와 동일해짐
            u_stu = projector(u_tea) # 입력은 원본 벡터
            i_stu = projector(i_tea)
            
            # (Batch,) 값은 -1.0 ~ 1.0
            cosine_scores = torch.sum(u_stu * i_stu, dim=1)
            
            # [핵심] 스케일 보정
            # Cosine(-1~1)에 큰 값을 곱해서 Teacher(-10~10)와 비슷하게 만듦
            scores_student = cosine_scores * projector.logit_scale.exp()
            
            # -----------------------------------------------------------
            # C. Loss 계산 (Distillation)
            # -----------------------------------------------------------
            # 두 점수 분포의 차이를 줄임 (MSE가 가장 직관적이고 빠름)
            loss = F.mse_loss(scores_student, scores_teacher)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Distillation Loss: {total_loss / len(dataloader):.4f}")

    return projector




'''

# 1. 학습 완료 후 변환
projector.eval()
with torch.no_grad():
    # LightGCL의 원본 임베딩(크기 제각각)을 넣음
    raw_user_emb = lightgcl_model.embedding_user.weight
    raw_item_emb = lightgcl_model.embedding_item.weight
    
    # Projector 통과 -> 크기가 1이면서 인기도 정보가 각도에 반영된 벡터 탄생
    final_user_emb = projector(raw_user_emb).cpu().numpy()
    final_item_emb = projector(raw_item_emb).cpu().numpy()

# 2. 이제 이 final_item_emb는 Norm=1 이므로
#    기존 시스템(FAISS 등)에 바로 넣어도 "인기도"가 반영된 추천이 나옵니다.
print(f"New Norm: {np.linalg.norm(final_user_emb, axis=1).mean():.4f}") # 1.0000 출력
'''