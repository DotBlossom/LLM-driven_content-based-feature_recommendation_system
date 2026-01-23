import json
import os
import random

# ==========================================
# 1. 설정: 경로 및 목표값
# ==========================================
# 기본 폴더 경로 (이 경로 아래에 json 파일들이 있다고 가정)
BASE_DIR = r"C:\Users\candyform\Desktop\inferenceCode\localprops"

# 처리할 카테고리 목록
TARGET_CATEGORIES = [
    
    #"blouse", "cardigan", "coat", "jacket", "jumper", "shirt", "sweater", "t-shirt", "vest",
     "pants" , "skirt", "dress", "jumpsuit"
]

# 샘플링 목표 설정
TARGET_TOTAL = 500
TARGET_MAN = 200

# ==========================================
# 2. 유틸리티 함수
# ==========================================
def clean_metadata_keys(item):
    """
    metadata. 접두사 제거 함수
    """
    try:
        clothes_data = item.get("feature_data", {}).get("clothes", {})
        new_clothes_data = {}

        for key, value in clothes_data.items():
            if key.startswith("metadata."):
                new_key = key.replace("metadata.", "", 1)
            else:
                new_key = key
            new_clothes_data[new_key] = value
        
        item["feature_data"]["clothes"] = new_clothes_data
        return item
    except Exception as e:
        print(f"⚠️ 키 변환 에러 (ID: {item.get('product_id')}): {e}")
        return item

def load_existing_ids(file_path):
    """
    이미 샘플링된 파일에서 product_id를 추출하여 Set으로 반환합니다.
    Set 자료구조를 사용하여 조회 속도를 O(1)로 만듭니다.
    """
    existing_ids = set()
    if not os.path.exists(file_path):
        return existing_ids # 파일이 없으면 빈 세트 반환

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                # ID 타입 불일치 방지를 위해 문자열로 통일
                pid = str(item.get("product_id"))
                existing_ids.add(pid)
    except Exception as e:
        print(f"⚠️ 참조 파일 로드 중 에러 ({os.path.basename(file_path)}): {e}")
    
    return existing_ids

# ==========================================
# 3. 메인 실행 로직
# ==========================================
def main():
    print(f"🚀 작업 시작: 총 {len(TARGET_CATEGORIES)}개 카테고리 처리 예정\n")

    for category in TARGET_CATEGORIES:
        # -----------------------------------------------------------
        # 경로 설정 (폴더 구조에 맞게 수정 필요시 여기를 변경하세요)
        # 예: bottom 폴더 안에 있다면 os.path.join(BASE_DIR, "bottom", f"{category}.json")
        # -----------------------------------------------------------
        
        # 1. 원본 파일 (전체 데이터)
        input_file = os.path.join(BASE_DIR, "bottom", f"{category}.json") 
        
        # 2. 참조 파일 (이미 뽑힌 데이터 - 제외 대상)
        reference_file = os.path.join(BASE_DIR, "sampler", f"{category}_sampled.json")
        
        # 3. 결과 파일 (새로 뽑을 데이터 저장소)
        output_file = os.path.join(BASE_DIR, "sampler", f"{category}_sampled_half.json")

        if not os.path.exists(input_file):
            print(f"❌ [SKIP] 원본 파일 없음: {category}")
            continue

        print(f"🔹 [{category}] 처리 중...")

        # [단계 1] 참조 파일에서 이미 존재하는 ID 로드 (O(1) 조회를 위한 Set 생성)
        excluded_ids = load_existing_ids(reference_file)
        print(f"   - 참조 파일 확인: 이미 존재하는 ID {len(excluded_ids)}개 제외 예정")

        # [단계 2] 원본 데이터 로드
        with open(input_file, "r", encoding="utf-8") as f:
            all_data = json.load(f)

        # [단계 3] 중복 제거 필터링
        candidates = []
        skipped_count = 0
        
        for item in all_data:
            pid = str(item.get("product_id"))
            
            # ⚡ 핵심: O(1) 속도로 제외 여부 확인
            if pid in excluded_ids:
                skipped_count += 1
            else:
                candidates.append(item)
        
        print(f"   - 원본 {len(all_data)}개 중 {skipped_count}개 중복 제외 -> 후보 {len(candidates)}개 확보")

        if not candidates:
            print("   ⚠️ 남은 후보 데이터가 없어 스킵합니다.")
            continue

        # [단계 4] 성별 분리 및 샘플링 로직 (Man 우선)
        men_items = []
        women_items = []

        for item in candidates:
            gender = item.get("feature_data", {}).get("clothes", {}).get("gender", "")
            if gender == "man":
                men_items.append(item)
            elif gender == "woman":
                women_items.append(item)

        # 수량 계산
        count_to_pick_man = min(len(men_items), TARGET_MAN)
        remaining_slots = TARGET_TOTAL - count_to_pick_man
        count_to_pick_woman = min(len(women_items), remaining_slots)

        # 랜덤 추출
        selected_men = random.sample(men_items, count_to_pick_man)
        selected_women = random.sample(women_items, count_to_pick_woman)
        
        raw_result = selected_men + selected_women

        # [단계 5] 메타데이터 키 정리
        final_result = []
        for item in raw_result:
            cleaned_item = clean_metadata_keys(item)
            final_result.append(cleaned_item)

        # [단계 6] 저장
        # 폴더가 없으면 생성
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=4, ensure_ascii=False)

        print(f"   ✅ 저장 완료: {os.path.basename(output_file)}")
        print(f"   👉 결과: 총 {len(final_result)}개 (Man: {len(selected_men)}, Woman: {len(selected_women)})\n")

if __name__ == "__main__":
    main()