import json
import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ==========================================
# 1. 설정 및 경로
# ==========================================
BASE_DIR = r"C:\Users\candyform\Desktop\inferenceCode\localprops"
DATA_DIR = os.path.join(BASE_DIR, "results")

# 순서가 동일하다고 가정하는 두 파일
FILE_A_PATH = os.path.join(BASE_DIR, "articles_detail_desc.json")      # 원본 문장
FILE_B_PATH = os.path.join(DATA_DIR, "final_ordered_result.json")     # 정렬된 결과값

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


BATCH_SIZE = 512 

# ==========================================
# 2. 데이터 로딩 및 전처리
# ==========================================
def load_and_extract_text():
    print("Step 1: 파일 로딩 및 텍스트 추출 중...")
    
    # File A (Sentences) 로드
    with open(FILE_A_PATH, 'r', encoding='utf-8') as f:
        data_a = json.load(f)
    
    # 텍스트 추출 (dict면 'text' or 'caption', 아니면 str)
    list_a = []
    for item in data_a:
        if isinstance(item, dict):
            list_a.append(item.get("text", "") or item.get("caption", ""))
        else:
            list_a.append(str(item))

    # File B (Results) 로드
    with open(FILE_B_PATH, 'r', encoding='utf-8') as f:
        data_b = json.load(f)
        
    # 텍스트 추출 (속성값들을 하나의 문자열로 결합)
    list_b = []
    for item in data_b:
        # text, score 같은 메타데이터 제외하고 속성값만 모음
        tokens = []
        for k, v in item.items():
            if k not in ['text', 'similarity_score', 'key_correct']:
                if isinstance(v, list):
                    tokens.extend([str(x) for x in v])
        list_b.append(" ".join(tokens))

    # 개수 체크
    if len(list_a) != len(list_b):
        print(f"⚠️ 경고: 개수 불일치! (A: {len(list_a)}, B: {len(list_b)})")
        min_len = min(len(list_a), len(list_b))
        list_a = list_a[:min_len]
        list_b = list_b[:min_len]
        print(f"   -> {min_len}개 기준으로 1:1 비교를 진행합니다.")
    else:
        print(f"✅ 개수 일치 확인: {len(list_a)}건")
        
    return list_a, list_b

# ==========================================
# 3. 고속 배치 비교 로직
# ==========================================
def run_fast_verification():
    # 데이터 준비
    texts_a, texts_b = load_and_extract_text()
    
    print(f"Step 2: 모델 로딩 ({DEVICE})...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    
    all_scores = []
    total_len = len(texts_a)
    
    print(f"Step 3: 배치 단위 GPU 검증 시작 (Batch: {BATCH_SIZE})...")
    
    # tqdm으로 진행률 표시
    for i in tqdm(range(0, total_len, BATCH_SIZE), desc="Verifying"):
        end_i = min(i + BATCH_SIZE, total_len)
        
        batch_a = texts_a[i:end_i]
        batch_b = texts_b[i:end_i]
        
        # 1. 임베딩 (Encoding) - GPU로 바로 텐서 생성
        emb_a = model.encode(batch_a, convert_to_tensor=True, show_progress_bar=False)
        emb_b = model.encode(batch_b, convert_to_tensor=True, show_progress_bar=False)
        
        # 2. 1:1 코사인 유사도 계산 (Pairwise Cosine Similarity)
        # pairwise_cos_sim
        scores = util.pairwise_cos_sim(emb_a, emb_b)
        
        # 3. 결과 수집 (GPU -> CPU)
        all_scores.extend(scores.cpu().tolist())

    # ==========================================
    # 4. 결과 리포트
    # ==========================================
    scores_np = np.array(all_scores)
    
    print("\n" + "="*50)
    print("🚀 FAST VERIFICATION REPORT")
    print("="*50)
    print(f"Total Pairs : {len(scores_np):,}")
    print(f"Average Sim : {np.mean(scores_np):.4f}")
    print(f"Median  Sim : {np.median(scores_np):.4f}")
    print(f"Min Score   : {np.min(scores_np):.4f}")
    print("-" * 50)
    
    # 점수대별 분포
    count_high = np.sum(scores_np >= 0.7)
    count_mid = np.sum((scores_np >= 0.5) & (scores_np < 0.7))
    count_low = np.sum(scores_np < 0.5)
    
    print(f"✅ High Match (>= 0.7) : {count_high:,} ({count_high/len(scores_np)*100:.1f}%)")
    print(f"⚠️ Mid Match  (0.5~0.7): {count_mid:,} ({count_mid/len(scores_np)*100:.1f}%)")
    print(f"❌ Low Match  (< 0.5)  : {count_low:,} ({count_low/len(scores_np)*100:.1f}%)")
    print("="*50)

    # (옵션) 문제가 되는 인덱스, lowerbound 직접 눈으로 보고 판단
    if count_low > 0:
        print("\n🔍 불일치 의심 상위 3개 (Low Score Examples):")
        worst_indices = np.argsort(scores_np)[:3]
        for idx in worst_indices:
            
            print(f"[Row {idx}] Score: {scores_np[idx]:.4f}")
            print(f"  A: {texts_a[idx][:100]}...") # 너무 길면 자름
            print(f"  B: {texts_b[idx][:100]}...")
            print("-" * 30)

if __name__ == "__main__":
    
    run_fast_verification()