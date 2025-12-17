from catboost import CatBoostRanker, Pool
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GBDTRankingModel:
    """
    [CatBoost Re-ranker]
    유저 벡터와 아이템 벡터를 결합하여 클릭 확률(Rank)을 예측
    """
    def __init__(self, model_path="catboost_ranker.cbm"):
        self.model_path = model_path
        self.model = CatBoostRanker(
            iterations=1000,          # 트리 개수 (학습량)
            learning_rate=0.03,       # 학습률
            depth=6,                  # 트리의 깊이 (피처 크로싱 복잡도)
            loss_function='YetiRank', # 랭킹 전용 손실함수 (NDCG 최적화)
            eval_metric='NDCG',
            verbose=100,
            task_type="GPU"           # GPU가 있다면 "GPU"로 변경 가능
        )
        self.is_fitted = False

    def train(self, user_vectors, item_vectors, labels, group_ids):
        """
        Args:
            user_vectors: (N, 128) numpy array
            item_vectors: (N, 128) numpy array
            labels: (N,) 0 or 1 (클릭 여부)
            group_ids: (N,) 유저 ID (쿼리 단위 그룹핑을 위해 필수)
        """
        # 1. Feature Engineering
        # 유저 벡터와 아이템 벡터를 옆으로 붙입니다. (Concatenation)
        # 추가로 '내적값(유사도)'을 피처로 
        dot_product = np.sum(user_vectors * item_vectors, axis=1, keepdims=True)
        X = np.hstack([user_vectors, item_vectors, dot_product])
        
        # 2. CatBoost Pool 생성
        train_pool = Pool(
            data=X,
            label=labels,
            group_id=group_ids # "이 유저 안에서 순서를 맞춰라"라는 뜻
        )
        
        # 3. 학습
        print("🌲 Start Training CatBoost Ranker...")
        self.model.fit(train_pool)
        self.is_fitted = True
        
        # 4. 저장
        self.model.save_model(self.model_path)
        print(f"✅ Model saved to {self.model_path}")

    def predict(self, user_vec, item_vecs):
        """
        [Inference]
        user_vec: (128,)
        item_vecs: (K, 128) - 후보 아이템 K개
        Returns: (K,) scores
        """
        if not self.is_fitted:
            # 모델 파일이 있으면 로드
            try:
                self.model.load_model(self.model_path)
                self.is_fitted = True
            except:
                # 학습된 적 없으면 랜덤 점수 반환 (Cold Start 방어)
                return np.random.rand(len(item_vecs))

        # 1. User Vector 확장 (Broadcasting)
        # (128,) -> (K, 128)
        K = len(item_vecs)
        user_batch = np.tile(user_vec, (K, 1))
        
        # 2. Feature 생성 (Train과 동일해야 함)
        dot_product = np.sum(user_batch * item_vecs, axis=1, keepdims=True)
        X_test = np.hstack([user_batch, item_vecs, dot_product])
        
        # 3. 예측
        return self.model.predict(X_test)





'''
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
        
        return scores.squeeze() # (N, 1) -> (N,) 형태로 
        
'''