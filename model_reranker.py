# model_reranker.py
import torch
import torch.nn as nn
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from deepctr_torch.models import DeepFM
from utils import vocab  # vocab.py 임포트

def build_reranker_model(
    simcse_encoder: nn.Module,   
    total_item_count: int,       
    re_max_capacity: int,        
    embedding_dim: int = 64,
    device: str = 'cuda'
):
    std_size, _ = vocab.get_vocab_sizes()
    
    # 1. Feature Columns 정의
    feature_columns = [
        DenseFeat("user_height", 1),
        DenseFeat("user_weight", 1),
        
        # SimCSE STD 임베딩 공유
        *[SparseFeat(key, vocabulary_size=std_size, embedding_dim=embedding_dim,
                     embedding_name="pretrained_std_embedding") 
          for key in vocab.STD_VOCAB_CONFIG.keys()],
          
        # SimCSE RE 임베딩 공유
        VarLenSparseFeat(
            SparseFeat("re_attributes", vocabulary_size=re_max_capacity, 
                       embedding_dim=embedding_dim, embedding_name="pretrained_re_embedding"),
            maxlen=10, combiner='mean'
        ),
        
        # User History (Reranker 독자 학습)
        VarLenSparseFeat(
            SparseFeat("history_item_id", vocabulary_size=total_item_count, 
                       embedding_dim=embedding_dim, embedding_name="item_id_embedding"),
            maxlen=50, combiner='mean'
        )
    ]

    # 2. 모델 생성
    model = DeepFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                   task='binary', device=device)
    
    # 3. 가중치 이식 (SimCSE -> DeepFM)
    if simcse_encoder is not None:
        print("🔄 Transferring SimCSE weights...")
        with torch.no_grad():
            if "pretrained_std_embedding" in model.embedding_dict:
                src = simcse_encoder.std_embedding.weight.data
                model.embedding_dict["pretrained_std_embedding"].weight.data.copy_(src)
                
            if "pretrained_re_embedding" in model.embedding_dict:
                src = simcse_encoder.re_embedding.weight.data
                tgt = model.embedding_dict["pretrained_re_embedding"].weight.data
                min_len = min(src.shape[0], tgt.shape[0])
                tgt[:min_len] = src[:min_len]
                
    return model