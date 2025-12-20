import json
import os

# ==========================================
# 1. 설정: 경로, 카테고리, 제거 키워드
# ==========================================
BASE_STD_DIR = r"C:\Users\candyform\Desktop\inferenceCode\dataset\sampler_std" 
BASE_RE_DIR = r"C:\Users\candyform\Desktop\inferenceCode\dataset\top_re" 
BASE_OUTPUT_DIR = r"C:\Users\candyform\Desktop\inferenceCode\dataset\merged" # 결과 저장 경로

TARGET_CATEGORIES = [
    #"blouse", "cardigan", "coat", "jacket", 
    #"jumper", "shirt", "sweater", "vest",
    "t-shirt", 
    #"pants" , "skirt", "dress", "jumpsuit"
]

# 제거할 Key의 뒷부분 단어들 (소문자 기준)
REMOVE_SUFFIXES = ("size", "length", "width") 

# ==========================================
# 2. 헬퍼 함수: 데이터 정제 및 타입 체크
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

def clean_measurements_in_place(item: dict) -> int:
    """
    item 내부의 feature_data.clothes 에서 특정 접미사를 가진 숫자 필드를 제거합니다.
    (Dictionary는 Mutable이므로 직접 수정됨)
    Returns: 제거된 필드 수
    """
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
    
    return removed_count

# ==========================================
# 3. 메인 실행 로직
# ==========================================
def main():
    # 결과 폴더 생성
    if not os.path.exists(BASE_OUTPUT_DIR):
        os.makedirs(BASE_OUTPUT_DIR)

    print(f"🚀 총 {len(TARGET_CATEGORIES)}개 카테고리 [병합 + 정제] 작업을 시작합니다.\n")

    for category in TARGET_CATEGORIES:
        # 파일 경로 설정
        std_filename = f"{category}_sampled_half.json"
        std_path = os.path.join(BASE_STD_DIR, std_filename)
        
        re_filename = f"{category}_half.json"
        re_path = os.path.join(BASE_RE_DIR, re_filename)
        
        output_filename = f"{category}_merged_half.json"
        output_path = os.path.join(BASE_OUTPUT_DIR, output_filename)

        # 파일 존재 여부 확인
        if not os.path.exists(std_path):
            print(f"⚠️ [SKIP] STD 파일 없음: {std_filename}")
            continue
        if not os.path.exists(re_path):
            print(f"⚠️ [SKIP] RE 파일 없음: {re_filename}")
            continue

        print(f"🔹 [{category}] 처리 중...")

        try:
            # 1. RE 데이터 로드 (매핑 테이블 생성)
            with open(re_path, 'r', encoding='utf-8') as f:
                re_data_list = json.load(f)
            
            re_map = {}
            for item in re_data_list:
                p_id = str(item.get("product_id"))
                re_val = item.get("reinforced_feature_value", {})
                re_map[p_id] = re_val
            
            # 2. STD 데이터 로드
            with open(std_path, 'r', encoding='utf-8') as f:
                std_data_list = json.load(f)
            
            final_processed_list = []
            matched_count = 0
            total_removed_fields = 0
            
            # 3. 병합 및 정제 루프
            for item in std_data_list:
                p_id = str(item.get("product_id"))

                # (A) 병합 조건 확인: RE 파일에 ID가 존재하는가?
                if p_id in re_map:
                    # (B) 병합 수행: reinforced_feature_value 주입
                    if "feature_data" not in item:
                        item["feature_data"] = {}
                    
                    item["feature_data"]["reinforced_feature_value"] = re_map[p_id]
                    
                    # (C) 정제 수행: size, length, width 숫자 필드 제거
                    removed_in_item = clean_measurements_in_place(item)
                    total_removed_fields += removed_in_item
                    
                    # (D) 최종 리스트 추가
                    final_processed_list.append(item)
                    matched_count += 1
            
            # 4. 결과 저장
            if final_processed_list:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(final_processed_list, f, indent=4, ensure_ascii=False)
                
                print(f"   ✅ 완료: {output_filename}")
                print(f"   👉 매칭 성공: {matched_count}개 / 삭제된 필드 합계: {total_removed_fields}개")
            else:
                print(f"   ⚠️ 매칭된 데이터가 없어 파일을 저장하지 않았습니다.")

        except Exception as e:
            print(f"❌ [ERROR] 처리 중 오류 발생 ({category}): {e}")

    print("\n🏁 모든 작업이 완료되었습니다.")
    print(f"💾 최종 저장 경로: {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()