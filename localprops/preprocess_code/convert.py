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
    "pants" , "skirt", "dress", "jumpsuit"
}

# 총 목표 수량
TOTAL_TARGET = 2000

# [중요] 남성 자리를 보장하기 위해 각각의 한계(Limit)를 설정합니다.
MAX_MAN = 800
MAX_WOMAN = TOTAL_TARGET - MAX_MAN  # 1200개

# 데이터 내부 제외할 필드
EXCLUDE_FIELDS = {
    "washing_method", "bleach", "ironing", 
    "drycleaning", "wringing", "drying"
}

# ==========================================
# 2. 데이터 변환 함수
# ==========================================
def transform_single_json(item: Dict[str, Any], gender: str) -> Dict[str, Any]:
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
    
    transformed_clothes["gender"] = gender
    
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
    print(f"🎯 목표 설정: 카테고리별 총 {TOTAL_TARGET}개 (Man: {MAX_MAN} / Woman: {MAX_WOMAN})")

    category_data_store: Dict[str, List] = {cat: [] for cat in TARGET_CATEGORIES}
    
    # 성별 카운트 추적용
    category_gender_count = {cat: {"man": 0, "woman": 0} for cat in TARGET_CATEGORIES}
    
    processed_ids = set()
    processed_count = 0
    
    # 스킵 카운터
    skipped_full_man = 0       # 남성 800 초과
    skipped_full_woman = 0     # 여성 1200 초과
    skipped_dup_count = 0      
    skipped_etc_count = 0      

    for file_path in json_files:
        try:
            filename = os.path.basename(file_path)
            parts = filename.split('_')

            if len(parts) <= 7:
                gender_props = parts[6].split('.')[0]
            else:
                gender_props = parts[7].split('.')[0]

            target_id = parts[2]
            raw_category = parts[6]
            bottom_category = parts[5]
            
            # 카테고리 매핑
            if "03-1onepiece" in bottom_category:
                current_category = "dress"
            elif "03-2onepiece" in bottom_category:
                current_category = "jumpsuit"
            else:
                current_category = raw_category[2:]

            # 타겟 아님
            if current_category not in TARGET_CATEGORIES:
                skipped_etc_count += 1
                continue

            # -----------------------------------------------------------
            # [핵심 수정] 성별별 쿼터제 적용 (남성 자리를 위해 여성을 제한)
            # -----------------------------------------------------------
            current_man_count = category_gender_count[current_category]["man"]
            current_woman_count = category_gender_count[current_category]["woman"]

            if gender_props == "man":
                if current_man_count >= MAX_MAN:
                    skipped_full_man += 1
                    continue
            elif gender_props == "woman":
                if current_woman_count >= MAX_WOMAN:
                    skipped_full_woman += 1
                    continue
            
            # (옵션) 혹시 모르니 전체 합계 안전장치
            if len(category_data_store[current_category]) >= TOTAL_TARGET:
                continue

            # 중복 ID 체크
            if target_id in processed_ids:
                skipped_dup_count += 1
                continue
            
            # --- 데이터 로드 및 저장 ---
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                result = transform_single_json(data, gender_props)
                
                category_data_store[current_category].append(result)
                
                if gender_props in ["man", "woman"]:
                    category_gender_count[current_category][gender_props] += 1
                
                processed_ids.add(target_id)
                processed_count += 1

                if processed_count % 1000 == 0:
                    print(f"   ...현재 {processed_count}개 파일 처리 완료")

        except Exception as e:
            print(f"❌ 에러 발생 ({os.path.basename(file_path)}): {e}")

    # ==========================================
    # 4. 결과 저장
    # ==========================================
    print("\n" + "="*60)
    print("💾 결과 저장 시작...")
    
    for category, data_list in category_data_store.items():
        if not data_list:
            continue

        output_filename = f"{category}.json"
        save_path = os.path.join(OUTPUT_FOLDER, output_filename)

        m_count = category_gender_count[category]['man']
        w_count = category_gender_count[category]['woman']
        total = len(data_list)
        
        # 상태 메시지 생성
        status_msgs = []
        if m_count < MAX_MAN:
            status_msgs.append(f"⚠️Man부족({m_count})")
        else:
            status_msgs.append("Man완료")
            
        if w_count < MAX_WOMAN:
            status_msgs.append(f"⚠️Woman부족({w_count})")
        else:
            status_msgs.append("Woman완료")

        status_str = " / ".join(status_msgs)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        
        print(f"   👉 [{category:<10}] 저장: {total:>4}개 (Man:{m_count:>3}, Woman:{w_count:>4}) | {status_str}")

    print("="*60)
    print(f"✅ 최종 리포트")
    print(f"   - 총 처리 성공: {processed_count}")
    print(f"   - 스킵 (Woman 1200개 초과): {skipped_full_woman}")
    print(f"   - 스킵 (Man 800개 초과): {skipped_full_man}")
    print(f"   - 스킵 (ID 중복): {skipped_dup_count}")
    print("="*60)

if __name__ == "__main__":
    main()