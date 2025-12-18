import json
import os

# ==========================================
# 1. 파일 경로 설정
# ==========================================
# 파일이 있는 실제 경로로 수정해주세요
BASE_STD_DIR = r"C:\Users\candyform\Desktop\inferenceCode\dataset\sampler_std" 
BASE_RE_DIR = r"C:\Users\candyform\Desktop\inferenceCode\dataset\bottom_re" 
BASE_DIR = r"C:\Users\candyform\Desktop\inferenceCode\dataset" 
RE_FILE = os.path.join(BASE_RE_DIR, "jumpsuit.json")
STD_FILE = os.path.join(BASE_STD_DIR, "jumpsuit_sampled.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "jumpsuit_merged.json")


TARGET_CATEGORIES = {
    "blouse", "cardigan", "coat", "jacket", 
    "jumper", "shirt", "sweater", "t-shirt", "vest",
    "pants" , "skirt", "dress", "jumpsuit"
}

def main():
    # -------------------------------------------------
    # 1. blouse_re.json 로드 및 매핑 테이블 생성
    # -------------------------------------------------
    print(f"📂 Loading reference data: {RE_FILE}")
    with open(RE_FILE, 'r', encoding='utf-8') as f:
        re_data_list = json.load(f)

    # 검색 속도를 위해 { "product_id" : "reinforced_feature_value" } 형태의 딕셔너리로 변환
    # ID 타입 불일치(str vs int) 방지를 위해 str()로 통일
    re_map = {}
    for item in re_data_list:
        p_id = str(item.get("product_id"))
        re_val = item.get("reinforced_feature_value", {})
        re_map[p_id] = re_val
    
    print(f"   -> Reference mapping created ({len(re_map)} items)")

    # -------------------------------------------------
    # 2. blouse_std.json 로드 및 데이터 병합
    # -------------------------------------------------
    print(f"📂 Loading standard data: {STD_FILE}")
    with open(STD_FILE, 'r', encoding='utf-8') as f:
        std_data_list = json.load(f)

    merged_count = 0
    
    for item in std_data_list:
        # std 데이터의 ID 추출
        p_id = str(item.get("product_id"))

        # 매핑 테이블에 해당 ID가 있는지 확인
        if p_id in re_map:
            # 존재하면 feature_data 내부의 reinforced_feature_value를 업데이트
            if "feature_data" in item:
                item["feature_data"]["reinforced_feature_value"] = re_map[p_id]
                merged_count += 1
            else:
                # 혹시 feature_data 구조가 없는 경우 생성 후 할당
                item["feature_data"] = {
                    "reinforced_feature_value": re_map[p_id]
                }
                merged_count += 1

    # -------------------------------------------------
    # 3. 결과 저장
    # -------------------------------------------------
    print(f"\n💾 Saving merged data to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(std_data_list, f, indent=4, ensure_ascii=False)

    print("="*50)
    print(f"✅ 병합 완료 리포트")
    print(f"   - 전체 대상(std) 개수 : {len(std_data_list)}")
    print(f"   - 매칭 성공 및 업데이트 : {merged_count}")
    print(f"   - 매칭 실패(re 데이터 없음) : {len(std_data_list) - merged_count}")
    print("="*50)

if __name__ == "__main__":
    main()