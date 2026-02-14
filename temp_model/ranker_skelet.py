import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from catboost import CatBoostClassifier, Pool
from typing import List, Dict, Any
import joblib
import os

# ==========================================
# 1. Feature Engineering (Data Preparation)
# ==========================================
class FeatureEngineer:
    """
    투타워 모델의 결과(Vectors, Scores)와 Raw Metadata를 결합하여
    CatBoost가 학습할 수 있는 Tabular 데이터로 변환합니다.
    """
    def __init__(self):
        # 범주형 변수 목록 (CatBoost는 이를 자동으로 처리함)
        self.cat_features = [
            'season', 
            'gender', 
            'item_category', 
            'item_material', 
            'item_fit',
            'time_of_day' # (오전/오후/저녁 등)
        ]
        
    def create_features(self, 
                        user_meta: Dict, 
                        item_meta: Dict, 
                        two_tower_score: float,
                        user_vector: np.ndarray,
                        item_vector: np.ndarray) -> Dict:
        """
        단일 샘플에 대한 피처 생성
        """
        features = {}
        
        # --- [Group A] Two-Tower Signals (가장 중요) ---
        features['two_tower_score'] = two_tower_score  # Cosine Similarity
        
        # (Advanced) 벡터 간의 상호작용 (Top-5 차원만 사용하거나, 전체 내적 등)
        # 벡터의 Element-wise 곱의 통계량 (Hybrid 전략)
        vec_interaction = user_vector * item_vector
        features['vec_prod_mean'] = np.mean(vec_interaction)
        features['vec_prod_max'] = np.max(vec_interaction)
        features['vec_prod_std'] = np.std(vec_interaction)
        
        # --- [Group B] User Metadata ---
        features['season'] = user_meta.get('season', 'unknown')
        features['gender'] = user_meta.get('gender', 'unknown')
        features['user_avg_price'] = user_meta.get('avg_price', 0.0) # 유저의 평균 구매 가격대
        
        # --- [Group C] Item Metadata ---
        features['item_category'] = item_meta.get('category', 'unknown')
        features['item_material'] = item_meta.get('material', 'unknown')
        features['item_fit'] = item_meta.get('fit', 'unknown')
        features['item_price'] = item_meta.get('price', 0)
        
        # --- [Group D] Context & Cross Features ---
        # 예: 유저 평균 가격대와 아이템 가격의 차이 (Price Sensitivity)
        if features['user_avg_price'] > 0:
            features['price_diff_ratio'] = (features['item_price'] - features['user_avg_price']) / features['user_avg_price']
        else:
            features['price_diff_ratio'] = 0.0
            
        return features

    def prepare_batch(self, batch_data: List[Dict]) -> pd.DataFrame:
        """
        배치 단위 변환 (학습/추론용)
        batch_data: [{'user_meta':..., 'item_meta':..., 'score':...}, ...]
        """
        rows = []
        for data in batch_data:
            row = self.create_features(
                data['user_meta'],
                data['item_meta'],
                data['two_tower_score'],
                data['user_vector'],
                data['item_vector']
            )
            # 학습용이면 target 추가
            if 'label' in data:
                row['target'] = data['label']
            rows.append(row)
            
        return pd.DataFrame(rows)


# ==========================================
# 2. CatBoost Model Wrapper
# ==========================================
class RecommendationRanker:
    def __init__(self, model_path: str = None):
        self.engineer = FeatureEngineer()
        self.model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss', # CTR 예측 (0~1 확률)
            eval_metric='AUC',
            verbose=100,
            early_stopping_rounds=50,
            cat_features=self.engineer.cat_features,
            random_seed=42
        )
        self.is_fitted = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(self, train_data: List[Dict], val_data: List[Dict] = None):
        """
        train_data: FeatureEngineer.prepare_batch에 들어갈 리스트 + 'label' (1=구매, 0=비구매)
        """
        print("🛠️ Preprocessing Training Data...")
        df_train = self.engineer.prepare_batch(train_data)
        X_train = df_train.drop(columns=['target'])
        y_train = df_train['target']
        
        eval_set = None
        if val_data:
            df_val = self.engineer.prepare_batch(val_data)
            X_val = df_val.drop(columns=['target'])
            y_val = df_val['target']
            eval_set = (X_val, y_val)
            
        print(f"🚀 Training CatBoost with {len(X_train)} samples...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            use_best_model=True
        )
        self.is_fitted = True
        print("✅ Training Finished.")

    def save_model(self, path: str):
        self.model.save_model(path)
        print(f"💾 Model saved to {path}")

    def load_model(self, path: str):
        self.model.load_model(path)
        self.is_fitted = True
        print(f"📂 Model loaded from {path}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1] # Class 1(구매)일 확률


# ==========================================
# 3. Inference Pipeline (Two-Tower -> Ranker)
# ==========================================
class ReRankingSystem:
    def __init__(self, 
                 user_tower: torch.nn.Module, 
                 item_tower: torch.nn.Module, 
                 ranker: RecommendationRanker,
                 item_db_metadata: Dict[int, Dict], # {product_id: {meta...}}
                 item_db_vectors: torch.Tensor):    # Pre-computed vectors
        
        self.user_tower = user_tower
        self.item_tower = item_tower
        self.ranker = ranker
        self.item_metadata = item_db_metadata
        self.item_vectors = item_db_vectors
        self.device = next(user_tower.parameters()).device

    def recommend(self, 
                  user_history_ids: List[int], 
                  season_idx: int, 
                  gender_idx: int, 
                  user_meta_raw: Dict,
                  top_k_retrieval: int = 100, 
                  final_k: int = 10):
        """
        [1. Retrieval] User Tower -> Top-100 Candidates
        [2. Re-ranking] CatBoost -> Top-10 Final
        """
        # --- Step 1: User Vector Generation ---
        self.user_tower.eval()
        with torch.no_grad():
            hist_tensor = torch.tensor([user_history_ids], device=self.device)
            season_tensor = torch.tensor([season_idx], device=self.device)
            gender_tensor = torch.tensor([gender_idx], device=self.device)
            
            # (1, 128)
            user_vector_emb = self.user_tower(hist_tensor, season_tensor, gender_tensor)
            
        # --- Step 2: Retrieval (Dot Product) ---
        # (1, 128) @ (N, 128).T -> (1, N)
        scores = torch.matmul(user_vector_emb, self.item_vectors.T).squeeze(0)
        
        # Top-K indices 추출
        top_scores, top_indices = torch.topk(scores, k=top_k_retrieval)
        
        top_indices_cpu = top_indices.cpu().numpy()
        top_scores_cpu = top_scores.cpu().numpy()
        user_vector_cpu = user_vector_emb.cpu().numpy().flatten()
        
        # --- Step 3: Prepare Data for Ranker ---
        ranker_input_list = []
        
        for i, item_idx in enumerate(top_indices_cpu):
            # item_db_vectors의 인덱스와 실제 product_id 매핑 필요 (여기선 인덱스=ID 가정)
            product_id = int(item_idx) 
            item_meta = self.item_metadata.get(product_id, {})
            item_vec = self.item_vectors[item_idx].cpu().numpy()
            
            ranker_input_list.append({
                'user_meta': user_meta_raw,
                'item_meta': item_meta,
                'two_tower_score': float(top_scores_cpu[i]),
                'user_vector': user_vector_cpu,
                'item_vector': item_vec,
                'product_id': product_id # 나중에 식별용
            })
            
        # --- Step 4: Re-ranking Prediction ---
        
        df_candidates = self.ranker.engineer.prepare_batch(ranker_input_list)
        
        # 구매 확률 예측
        probs = self.ranker.predict_proba(df_candidates)
        
        # --- Step 5: Sort & Final Select ---
        # 확률 기준으로 내림차순 정렬
        final_indices = np.argsort(probs)[::-1][:final_k]
        
        recommendations = []
        for idx in final_indices:
            rec_item = ranker_input_list[idx]
            rec_item['final_score'] = probs[idx]
            recommendations.append(rec_item)
            
        return recommendations
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