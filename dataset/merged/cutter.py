import json
import os
import glob
from typing import Dict, Any, List

# ==========================================
# 1. 설정: 경로 및 제거 키워드
# ==========================================
# 이전 단계의 output 폴더를 입력으로 사용
INPUT_FOLDER = r"C:\Users\candyform\Desktop\inferenceCode\dataset\merged"
# 처리된 파일을 저장할 새로운 폴더
OUTPUT_FOLDER = r"C:\Users\candyform\Desktop\inferenceCode\dataset\merged\cutter"
# 제거할 Key의 뒷부분 단어들 (소문자 기준)
# "lentgth"는 오타 같아서 제외하고 표준인 "length"를 넣었습니다. 필요시 추가하세요.
REMOVE_SUFFIXES = ("size", "length", "width") 

# ==========================================
# 2. 헬퍼 함수: 값이 숫자인지 확인
# ==========================================
def is_number(s):
    """
    값이 int, float이거나, 문자열이라도 숫자("67.0")로 변환 가능한지 확인
    """
    if isinstance(s, (int, float)):
        return True
    if isinstance(s, str):
        try:
            float(s)
            return True
        except ValueError:
            return False
    return False

# ==========================================
# 3. 데이터 정제 함수
# ==========================================
def clean_measurements(item: Dict[str, Any]) -> tuple[Dict[str, Any], int]:
    clothes = item.get("feature_data", {}).get("clothes", {})
    removed_count = 0
    keys_to_remove = []

    for key, value in clothes.items():
        # 1. Key가 특정 단어로 끝나는지 확인 (대소문자 무시)
        if key.lower().endswith(REMOVE_SUFFIXES):
            # 2. Value가 실제로 숫자인지 확인
            if is_number(value):
                keys_to_remove.append(key)
    
    # 찾은 키 삭제
    for key in keys_to_remove:
        del clothes[key]
        removed_count += 1
    
    return item, removed_count

# ==========================================
# 4. 메인 실행 로직
# ==========================================
def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    json_files = glob.glob(os.path.join(INPUT_FOLDER, "*.json"))
    print(f"📂 '{INPUT_FOLDER}' 에서 {len(json_files)}개의 파일을 발견했습니다.\n")

    total_removed_fields = 0

    for file_path in json_files:
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)
            
            cleaned_data_list = []
            file_removed_count = 0

            for item in data_list:
                cleaned_item, count = clean_measurements(item)
                cleaned_data_list.append(cleaned_item)
                file_removed_count += count

            # 저장
            save_path = os.path.join(OUTPUT_FOLDER, filename)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_data_list, f, indent=4, ensure_ascii=False)
            
            print(f" ✅ {filename} 완료 (제거된 필드: {file_removed_count}개)")
            total_removed_fields += file_removed_count

        except Exception as e:
            print(f" ❌ 에러 발생 ({filename}): {e}")

    print("\n" + "="*50)
    print(f"🎉 모든 작업 완료!")
    print(f"💾 저장 경로: {OUTPUT_FOLDER}")
    print(f"🗑️ 총 제거된 수치 필드 수: {total_removed_fields}개")
    print("="*50)

if __name__ == "__main__":
    main()