import json
import os
import re

# 1. 경로 및 파일 목록 설정
BASE_DIR = r"C:\Users\candyform\Desktop\inferenceCode\localprops"
SENT_DIR = os.path.join(BASE_DIR, "split_files")
DATA_DIR = os.path.join(BASE_DIR, "results")

sentence_files = [os.path.join(SENT_DIR, f"file_{i}.json") for i in range(1, 60)]

target_data_filenames = [
    "desc_tokenizer_merged.json",
    "desc_tokenizer_17_merged.json",
    "desc_tokenizer_31_merged.json", 
    "desc_tokenizer_41_merged.json",
    "desc_tokenizer_51_merged.json"
]
data_files = [os.path.join(DATA_DIR, f) for f in target_data_filenames]
output_file_path = os.path.join(DATA_DIR, "processed_result.json")

# --- [핵심 로직] 불용어 및 구두점 정의 ---
# 프롬프트 규칙에 있는 제거 대상 단어들
STOPWORDS = {'in', 'with', 'at', 'for', 'that', 'and', 'the', 'a', 'an'}

def normalize_text(text):
    """
    텍스트를 비교하기 좋게 정규화합니다.
    1. 소문자 변환
    2. 문장 부호 제거 (하이픈, 콤마, 마침표 등)
    3. 불용어(Stopwords) 제거
    4. 모든 공백 제거 (압축)
    """
    if not text:
        return ""
    
    # 1. 소문자 변환
    text = str(text).lower()
    
    # 2. 문장 부호 제거 (특수문자를 공백으로 치환)
    # [^\w\s] : 문자나 공백이 아닌 것(구두점 등)을 찾아서 제거
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 3. 불용어 제거 (단어 단위로 정확히 일치하는 것만 제거)
    # \b는 단어의 경계(boundary)를 의미합니다. 예: "sand"에서 "and"는 안 지워짐.
    tokens = text.split()
    filtered_tokens = [t for t in tokens if t not in STOPWORDS]
    
    # 4. 모든 공백 제거 후 합치기 (예: "soft fine knit" -> "softfineknit")
    return "".join(filtered_tokens)

# ----------------------------------------

def stream_json_items(file_list):
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"Skipping missing file: {file_path}")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        yield item
                else:
                    yield data
            except json.JSONDecodeError:
                print(f"Error decoding JSON in {file_path}")


def process_and_save():
    print(f"Sentences Source: {SENT_DIR}")
    print(f"Data Source: {DATA_DIR}")
    print("Processing started (Advanced Normalization Mode)...\n")

    sent_stream = stream_json_items(sentence_files)
    data_stream = stream_json_items(data_files)
    
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write("[\n") 
        
        first_line = True
        processed_count = 0
        
        for sentence_item, data_item in zip(sent_stream, data_stream):
            
            # 1. 문장 데이터 추출
            if sentence_item is None:
                sentence_raw = ""
            elif isinstance(sentence_item, str):
                sentence_raw = sentence_item
            elif isinstance(sentence_item, dict):
                sentence_raw = sentence_item.get("text", "")
            else:
                sentence_raw = str(sentence_item)
            
            # [변경] 문장을 강력하게 정규화 (불용어/구두점/공백 제거)
            sentence_norm = normalize_text(sentence_raw)
            
            # 2. 데이터 아이템 방어
            if not isinstance(data_item, dict):
                data_item = {"key_correct": False} 

            # 3. 타겟 키워드 추출 및 정규화
            targets_norm = []
            if data_item:
                for vals in data_item.values():
                    if isinstance(vals, list):
                        # [변경] 타겟 값들도 똑같은 방식으로 정규화해서 리스트에 추가
                        for v in vals:
                            targets_norm.append(normalize_text(v))
            
            total_keywords = len(targets_norm)
            is_correct = False

            # 4. 비교 로직 (정규화된 문자열끼리 비교)
            if total_keywords > 0 and sentence_norm:
                # sentence_norm 안에 target_norm 문자열이 포함되어 있는지 확인
                match_count = sum(1 for t in targets_norm if t in sentence_norm)

                if total_keywords < 3:
                    # 3개 미만: 100% 일치해야 True
                    is_correct = (match_count == total_keywords)
                else:
                    # 3개 이상: 3개 이상만 일치하면 True
                    is_correct = (match_count >= 3)
            
            data_item["key_correct"] = is_correct

            if not first_line:
                f_out.write(",\n")
            json.dump(data_item, f_out, ensure_ascii=False)
            
            first_line = False
            processed_count += 1
            
            if processed_count % 1000 == 0:
                print(f"Processed {processed_count} items...", end='\r')

        f_out.write("\n]") 

    print(f"\nDone! Total {processed_count} items saved to:")
    print(f"-> {output_file_path}")

if __name__ == "__main__":
    process_and_save()