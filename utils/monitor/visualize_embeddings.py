import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# 혹은 더 빠른 UMAP 추천: pip install umap-learn
# import umap.umap_ as umap 
import torch
def visualize_embeddings(model, item_map):
    model.eval()
    with torch.no_grad():
        # 학습된 최종 임베딩 가져오기
        _, final_item_embeddings = model(perturbed=False)
        embeddings = final_item_embeddings.cpu().numpy()
        
    # t-SNE로 2차원 축소 (아이템 1000개만 샘플링해서 찍어보기)
    # 데이터가 많으면 시간이 오래 걸리므로 UMAP 추천
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_emb = tsne.fit_transform(embeddings[:1000]) # 1000개만
    
    # 시각화
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], alpha=0.6, s=10)
    plt.title("Item Embeddings Visualization (SimGCL)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.show()
