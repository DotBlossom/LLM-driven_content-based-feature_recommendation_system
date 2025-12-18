import json
import os
import random

# ==========================================
# 1. 설정: 입출력 파일 경로
# ==========================================
# 원본 데이터가 들어있는 JSON 파일 경로
INPUT_FILE_PATH = r"C:\Users\candyform\Desktop\inferenceCode\localprops\bottom\jumpsuit.json" 

# 결과를 저장할 JSON 파일 경로
OUTPUT_FILE_PATH = r"C:\Users\candyform\Desktop\inferenceCode\localprops\sampler\jumpsuit_sampled2.json"

# 목표 설정
TARGET_TOTAL = 50
TARGET_MAN = 20


TARGET_CATEGORIES = {
    "blouse", "cardigan", "coat", "jacket", 
    "jumper", "shirt", "sweater", "t-shirt", "vest",
    "pants" , "skirt", "dress", "jumpsuit"
}

# ==========================================
# 2. 키 이름 변환 함수 (metadata. 제거)
# ==========================================
def clean_metadata_keys(item):
    """
    feature_data -> clothes 내부의 키 중에서
    'metadata.'로 시작하는 키의 이름을 정리합니다.
    예: 'metadata.top.chest_size' -> 'top.chest_size'
    """
    try:
        clothes_data = item.get("feature_data", {}).get("clothes", {})
        new_clothes_data = {}

        for key, value in clothes_data.items():
            # 'metadata.'로 시작하면 잘라내기
            if key.startswith("metadata."):
                new_key = key.replace("metadata.", "", 1) # 맨 앞의 metadata.만 제거
            else:
                new_key = key
            
            new_clothes_data[new_key] = value
        
        # 변환된 딕셔너리로 교체
        item["feature_data"]["clothes"] = new_clothes_data
        return item
        
    except Exception as e:
        # 데이터 구조가 예상과 다르면 에러 출력 후 원본 반환
        print(f"⚠️ 키 변환 중 에러 발생 (ID: {item.get('product_id')}): {e}")
        return item

# ==========================================
# 3. 메인 실행 로직
# ==========================================
def main():
    if not os.path.exists(INPUT_FILE_PATH):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {INPUT_FILE_PATH}")
        return

    print("📂 데이터 로딩 중...")
    with open(INPUT_FILE_PATH, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    
    print(f"   -> 총 {len(all_data)}개의 데이터를 불러왔습니다.")

    # 1. 성별 분리
    men_items = []
    women_items = []

    for item in all_data:
        gender = item.get("feature_data", {}).get("clothes", {}).get("gender", "")
        if gender == "man":
            men_items.append(item)
        elif gender == "woman":
            women_items.append(item)

    print(f"   -> 남성 데이터: {len(men_items)}개 / 여성 데이터: {len(women_items)}개")

    # 2. 수량 계산 (남성 30 보장, 부족 시 여성으로 채움)
    # (A) 남성 뽑을 개수
    count_to_pick_man = min(len(men_items), TARGET_MAN)
    
    # (B) 여성 뽑을 개수 (전체 100 - 남성 뽑은 수)
    remaining_slots = TARGET_TOTAL - count_to_pick_man
    count_to_pick_woman = min(len(women_items), remaining_slots)

    # 3. 랜덤 샘플링
    selected_men = random.sample(men_items, count_to_pick_man)
    selected_women = random.sample(women_items, count_to_pick_woman)
    
    raw_result = selected_men + selected_women

    # 4. 키 이름 정리 (metadata. 제거)
    final_result = []
    for item in raw_result:
        cleaned_item = clean_metadata_keys(item)
        final_result.append(cleaned_item)

    # 5. 결과 저장
    print("\n💾 결과 저장 중...")
    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)

    print(f"✅ 완료! 저장 경로: {OUTPUT_FILE_PATH}")
    print(f"   - 총 저장 개수: {len(final_result)}개")
    print(f"   - 구성: Man {len(selected_men)}개 / Woman {len(selected_women)}개")

if __name__ == "__main__":
    main()