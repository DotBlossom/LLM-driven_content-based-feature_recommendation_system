# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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