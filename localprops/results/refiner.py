import json
import os

BASE_DIR = r"C:\Users\candyform\Desktop\inferenceCode\localprops\results"
target_files = [
    "desc_tokenizer.json",
    "desc_tokenizer_17.json",
    "desc_tokenizer_31.json", # 특이한 확장자(.json_31)도 이름_merged.json_31 형태로 보존됩니다.
    "desc_tokenizer_41.json",
    "desc_tokenizer_51.json"
]

# ---------------------------------------------------------
# [핵심 로직] 중복된 키 병합 (Merge Duplicates)
# ---------------------------------------------------------
def merge_duplicates_hook(pairs):
    d = {}
    for key, val in pairs:
        if key in d:
            if isinstance(d[key], list) and isinstance(val, list):
                d[key].extend(val)
            else:
                if not isinstance(d[key], list):
                    d[key] = [d[key]]
                if isinstance(val, list):
                    d[key].extend(val)
                else:
                    d[key].append(val)
        else:
            d[key] = val
    return d

def salvage_and_merge_json():
    # 중복 키 병합 기능이 탑재된 디코더
    decoder = json.JSONDecoder(object_pairs_hook=merge_duplicates_hook)
    
    print(f"작업 경로: {BASE_DIR}\n")

    for filename in target_files:
        input_path = os.path.join(BASE_DIR, filename)
        
        # --- [파일명 수정 로직] ---
        # 확장자를 분리하여 사이에 _merged 삽입
        # 예: desc_tokenizer.json -> root="desc_tokenizer", ext=".json"
        # 결과: desc_tokenizer_merged.json
        file_root, file_ext = os.path.splitext(filename)
        output_filename = f"{file_root}_merged{file_ext}"
        output_path = os.path.join(BASE_DIR, output_filename)

        if not os.path.exists(input_path):
            print(f"❌ 파일 없음: {filename}")
            continue

        print(f"🔨 복구 및 병합 시작: {filename} ...")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        recovered_items = []
        idx = 0
        content_len = len(content)
        
        while idx < content_len:
            if content[idx] != '{':
                idx += 1
                continue
            
            try:
                obj, end_idx = decoder.raw_decode(content, idx)
                recovered_items.append(obj)
                idx = end_idx
            except json.JSONDecodeError:
                idx += 1

        if recovered_items:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(recovered_items, f_out, indent=2, ensure_ascii=False)
            
            print(f"✅ 완료: {len(recovered_items)}개 객체 저장됨.")
            print(f"   -> 저장 파일명: {output_filename}")
        else:
            print(f"⚠️ 실패: 데이터를 찾지 못했습니다.")

if __name__ == "__main__":
    salvage_and_merge_json()