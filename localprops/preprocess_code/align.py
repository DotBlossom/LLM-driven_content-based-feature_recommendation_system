import json
import os
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ==========================================
# 1. 설정 및 경로
# ==========================================
BASE_DIR = r"C:\Users\candyform\Desktop\inferenceCode\localprops"
DATA_DIR = os.path.join(BASE_DIR, "results")

# 기준이 되는 문장 파일 (이 파일의 순서를 따름)
SENTENCE_FILE = os.path.join(BASE_DIR, "articles_detail_desc.json")


# 검색 대상이 될 타겟 데이터 
TARGET_FILES = [
    "desc_tokenizer_merged.json",
    "desc_tokenizer_17_merged.json",
    "desc_tokenizer_31_merged.json",
    "desc_tokenizer_41_merged.json",
    "desc_tokenizer_51_merged.json"
]

OUTPUT_FILE = os.path.join(DATA_DIR, "final_ordered_result.json")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'all-MiniLM-L6-v2'
BATCH_SIZE = 256 # 2070 Super 기준 넉넉

# ==========================================
# 2. 유틸리티 함수
# ==========================================
def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def flatten_target_data(file_list):
    """모든 타겟 데이터를 하나의 리스트로 합칩니다."""
    all_items = []
    print("Loading target data files (Corpus)...")
    for fname in file_list:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            data = load_json(path)
            if isinstance(data, list):
                all_items.extend(data)
            else:
                all_items.append(data)
    print(f"Total Target Candidates: {len(all_items)}")
    return all_items

def create_string_representation(data_item):
    """JSON 객체를 검색용 문자열로 변환 (Value들만 공백 연결)"""
    if not isinstance(data_item, dict):
        return ""
    tokens = []
    for val in data_item.values():
        if isinstance(val, list):
            tokens.extend([str(v) for v in val])
    return " ".join(tokens)

# ==========================================
# 3. 메인 로직
# ==========================================
def run_sentence_ordered_alignment():
    print(f"Initializing Model on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    # --- Step 1: Corpus 생성 ---

    target_items = flatten_target_data(TARGET_FILES)
    target_strings = [create_string_representation(item) for item in target_items]
    
    print(f"[1/3] Embedding {len(target_strings)} target items into GPU memory...")
    # 타겟 데이터 전체를 GPU 텐서로 변환 (Corpus)
    corpus_embeddings = model.encode(
        target_strings, 
        batch_size=BATCH_SIZE, 
        convert_to_tensor=True, 
        show_progress_bar=True
    )

    # --- Step 2: 문장 데이터 로드 ---
    print(f"\n[2/3] Loading Sentences from {SENTENCE_FILE}...")
    try:
        sentences_raw = load_json(SENTENCE_FILE)
    except FileNotFoundError:
        print(f"❌ Error: Cannot find {SENTENCE_FILE}")
        return

    # 문장 리스트 정제
    sentences = []
    for s in sentences_raw:
        if isinstance(s, dict):
            # text 키가 없으면 caption, 그것도 없으면 빈 문자열
            sentences.append(s.get("text", "") or s.get("caption", ""))
        else:
            sentences.append(str(s))
            
    total_sentences = len(sentences)
    print(f"Total Sentences to process: {total_sentences}")

    # --- Step 3: 배치 단위 검색 및 저장 ---
    print(f"\n[3/3] Searching Best Targets for each Sentence & Saving...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        f_out.write("[\n")
        first_line = True
        
        for start_idx in tqdm(range(0, total_sentences, BATCH_SIZE)):
            end_idx = min(start_idx + BATCH_SIZE, total_sentences)
            batch_sentences = sentences[start_idx:end_idx]
            
            # 1. 문장 배치 임베딩 (Query Embedding)
            query_embeddings = model.encode(
                batch_sentences, 
                convert_to_tensor=True, 
                show_progress_bar=False
            )
            
            # 2. 시맨틱 서치 (이 문장들과 가장 유사한 타겟을 Corpus에서 찾음)
            # top_k=1 : 가장 유사한 것 1개만
            hits = util.semantic_search(
                query_embeddings, 
                corpus_embeddings, 
                top_k=1,
                query_chunk_size=BATCH_SIZE
            )
            
            # 3. 결과 매핑 및 파일 쓰기
            for i, hit in enumerate(hits):
                # hit은 리스트 형태 [{'corpus_id': 123, 'score': 0.9}]
                best_match_idx = hit[0]['corpus_id']
                score = hit[0]['score']
                
                # 원본 문장
                original_sentence = batch_sentences[i]
                
                # 찾아낸 가장 유사한 타겟 데이터 (복사해서 사용)
                matched_target = target_items[best_match_idx].copy()
                
                # 결과 데이터 구성
                matched_target['text'] = original_sentence
                matched_target['similarity_score'] = float(score)
                
   
                matched_target['key_correct'] = True 
                
                if not first_line:
                    f_out.write(",\n")
                json.dump(matched_target, f_out, ensure_ascii=False)
                first_line = False

        f_out.write("\n]")

    print(f"\n✅ All Done! Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    run_sentence_ordered_alignment()