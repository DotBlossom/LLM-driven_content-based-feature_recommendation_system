import json
import os
import glob
from typing import Dict, Any, List

# ==========================================
# 1. 설정: 경로 및 기준값
# ==========================================
INPUT_FOLDER = r"C:\Users\candyform\Desktop\008.의류 통합 데이터(착용 이미지, 치수 및 원단 정보)\01-1.정식개방데이터\Training\02.라벨링데이터_bottom"
OUTPUT_FOLDER = r"C:\Users\candyform\Desktop\008.의류 통합 데이터(착용 이미지, 치수 및 원단 정보)\01-1.정식개방데이터\Training\output_bottom"

TARGET_CATEGORIES = {
    "blouse", "cardigan", "coat", "jacket", 
    "jumper", "shirt", "sweater", "t-shirt", "vest",
    "pants" , "skirt", "dress", "jumpsuit"  # 원피스, 점프슈트 추가
}

# 카테고리별 최대 허용 개수
MAX_PER_CATEGORY = 2000

# 데이터 내부 제외할 필드
EXCLUDE_FIELDS = {
    "washing_method", "bleach", "ironing", 
    "drycleaning", "wringing", "drying"
}

# ==========================================
# 2. 데이터 변환 함수 (기존 동일)
# ==========================================
def transform_single_json(item: Dict[str, Any]) -> Dict[str, Any]:
    dataset_info = item.get("dataset", {})
    product_id = dataset_info.get("dataset.id")
    clothes_raw = item.get("metadata.clothes", {})
    
    transformed_clothes = {}

    for key, value in clothes_raw.items():
        if value is None or value == "null":
            continue

        clean_key = key.replace("metadata.clothes.", "")

        if clean_key in EXCLUDE_FIELDS:
            continue

        if clean_key == "type":
            clean_key = "category"

        transformed_clothes[clean_key] = value

    return {
        "product_id": product_id,
        "feature_data": {
            "clothes": transformed_clothes,
            "reinforced_feature_value": {}
        }
    }

# ==========================================
# 3. 메인 실행 로직
# ==========================================
def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    json_files = glob.glob(os.path.join(INPUT_FOLDER, "*.json"))
    print(f"📂 '{INPUT_FOLDER}' 에서 {len(json_files)}개의 파일을 발견했습니다.\n")

    category_data_store: Dict[str, List] = {cat: [] for cat in TARGET_CATEGORIES}
    processed_ids = set()

    processed_count = 0
    skipped_full_count = 0 
    skipped_dup_count = 0   
    skipped_etc_count = 0   

    for file_path in json_files:
        try:
            filename = os.path.basename(file_path)
            parts = filename.split('_')

            # 1. 파일명 길이 체크
            if len(parts) <= 6:
                skipped_etc_count += 1
                continue

            target_id = parts[2]      # ID (3번째)
            raw_category = parts[6]   # 카테고리 (7번째)
            bottom_category = parts[5]
            # === [수정됨] 카테고리 매핑 로직 ===
            # 원피스/점프슈트 특수 케이스 처리
            if "03-1onepiece" in bottom_category:
                current_category = "dress"
            elif "03-2onepiece" in bottom_category:
                current_category = "jumpsuit"
            else:
                # 일반적인 상의 (02t-shirt -> t-shirt) : 앞 2글자 자르기
                current_category = raw_category[2:]

            # 2. 타겟 확인
            if current_category not in TARGET_CATEGORIES:
                skipped_etc_count += 1
                continue

            # 3. 수량 체크 (2000개)
            if len(category_data_store[current_category]) >= MAX_PER_CATEGORY:
                skipped_full_count += 1
                continue

            # 4. 중복 ID 체크
            if target_id in processed_ids:
                skipped_dup_count += 1
                continue
            
            # --- 처리 진행 ---
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                result = transform_single_json(data)
                
                category_data_store[current_category].append(result)
                processed_ids.add(target_id)
                processed_count += 1

                if processed_count % 500 == 0:
                    print(f"   ...현재 {processed_count}개 파일 처리 완료")

        except Exception as e:
            print(f"❌ 에러 발생 ({os.path.basename(file_path)}): {e}")

    # ==========================================
    # 4. 결과 저장
    # ==========================================
    print("\n" + "="*50)
    print("💾 결과 저장 시작...")
    
    for category, data_list in category_data_store.items():
        if not data_list:
            continue

        output_filename = f"{category}.json"
        save_path = os.path.join(OUTPUT_FOLDER, output_filename)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        
        print(f"   👉 [{category}] 저장 완료: {len(data_list)}개 -> {output_filename}")

    print("="*50)
    print(f"✅ 최종 리포트")
    print(f"   - 총 처리 성공: {processed_count}")
    print(f"   - 스킵 (수량 초과): {skipped_full_count}")
    print(f"   - 스킵 (ID 중복): {skipped_dup_count}")
    print(f"   - 스킵 (대상 아님): {skipped_etc_count}")
    print("="*50)

if __name__ == "__main__":
    main()