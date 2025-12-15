import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossNet(nn.Module):
    """
    [Cross Network]
    피처 간의 명시적인 상호작용(Interaction)을 학습합니다.
    수식: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
    """
    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        # 각 층마다 가중치와 편향을 가짐
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(input_dim, 1))) 
            for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) 
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x_0 = x # 원본 입력 유지
        x_l = x
        
        for i in range(self.num_layers):
            # 1. Linear Projection (W * x)
            # (B, D) @ (D, 1) -> (B, 1) : 스칼라 값 생성
            linear_proj = torch.matmul(x_l, self.kernels[i]) + self.biases[i] 
            
            # 2. Feature Crossing (x_0 * Scalar)
            # 원본 입력 x_0에 스칼라를 곱해 모든 피처 간의 교차 효과를 냄
            # + x_l (Residual Connection)
            x_l = x_0 * linear_proj + x_l
            
        return x_l

class RankingModel(nn.Module):
    """
    [DCN-V2 Re-ranker]
    Retrieval 모델이 뽑은 후보군(Top-N)을 정밀 채점
    """
    def __init__(self, user_dim=128, item_dim=128, context_dim=20):
        super().__init__()
        
        total_input_dim = user_dim + item_dim + context_dim
        
        # 1. Cross Network (Explicit Interaction)
        self.cross_net = CrossNet(total_input_dim, num_layers=3)
        
        # 2. Deep Network (Implicit Interaction)
        self.deep_net = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        
        # 3. Final Prediction Head
        # Cross 출력(D) + Deep 출력(128) -> Score(1)
        self.final_head = nn.Linear(total_input_dim + 128, 1)
        
        
    def forward(self, user_emb, item_emb, context_emb=None):
        # (B, 128), (B, 128), (B, 20)
        
        # 1. Feature Concatenation (Early Interaction)
        if context_emb is not None:
            x = torch.cat([user_emb, item_emb, context_emb], dim=1)
        else:
            x = torch.cat([user_emb, item_emb], dim=1)
            
        # 2. Dual Path Processing
        cross_out = self.cross_net(x) # (B, Total_D)
        deep_out = self.deep_net(x)   # (B, 128)
        
        # 3. Stack & Predict
        stacked = torch.cat([cross_out, deep_out], dim=1)
        logits = self.final_head(stacked)
        
        # 4. Score (0~1 Probability)
        return torch.sigmoid(logits)
    
    
    @torch.no_grad() # 추론 전용이므로 Gradient 계산 끔
    def predict_for_user(self, user_vec, item_vecs, context_vec=None):
        """
        [Inference Helper]
        1명의 유저 벡터를 N개의 아이템 벡터에 맞춰서 확장(Broadcasting)하고 점수를 계산합니다.
        
        Args:
            user_vec: (1, 128) 또는 (128,) - 유저 1명의 벡터
            item_vecs: (N, 128) - 후보 아이템 N개의 벡터
            context_vec: (1, 20) - 현재 컨텍스트 (선택)
        """
        # 1. 차원 정리 (1차원이면 2차원으로)
        if user_vec.dim() == 1:
            user_vec = user_vec.unsqueeze(0) # (128,) -> (1, 128)
            
        # 2. 개수 확인 (N)
        num_candidates = item_vecs.size(0)
        
        # 3. 확장 (Broadcasting)
        # (1, 128) -> (N, 128) 로 복사
        user_batch = user_vec.expand(num_candidates, -1)
        
        # 컨텍스트가 있다면 동일하게 확장
        context_batch = None
        if context_vec is not None:
            if context_vec.dim() == 1:
                context_vec = context_vec.unsqueeze(0)
            context_batch = context_vec.expand(num_candidates, -1)
            
        # 4. Forward 호출
        # 이제 user_batch와 item_vecs의 크기가 (N, 128)로 같으므로 forward 사용 가능
        scores = self.forward(user_batch, item_vecs, context_batch)
        
        return scores.squeeze() # (N, 1) -> (N,) 형태로 반환