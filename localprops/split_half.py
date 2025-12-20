import json
import os

# ==========================================
# 1. 설정: 경로 및 카테고리
# ==========================================
# sampler 폴더 경로 (파일들이 위치한 곳)
BASE_DIR = r"C:\Users\candyform\Desktop\inferenceCode\localprops\sampler"

TARGET_CATEGORIES = [
    "blouse", "cardigan", "coat", "jacket", 
    "jumper", "shirt", "sweater", "t-shirt", "vest",
    "pants" , "skirt", "dress", "jumpsuit"
]

# ==========================================
# 2. 메인 로직
# ==========================================
def main():
    print(f"📂 작업 경로: {BASE_DIR}")
    print(f"🚀 총 {len(TARGET_CATEGORIES)}개 카테고리 분할 시작...\n")

    for category in TARGET_CATEGORIES:
        # 입력 파일명 구성
        input_filename = f"{category}_sampled_half.json"
        input_path = os.path.join(BASE_DIR, input_filename)

        # 파일 존재 여부 확인
        if not os.path.exists(input_path):
            print(f"⚠️ [SKIP] 파일을 찾을 수 없음: {input_filename}")
            continue

        # 1. 데이터 로드
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ [ERROR] 파일 읽기 실패 ({input_filename}): {e}")
            continue

        total_count = len(data)
        if total_count == 0:
            print(f"⚠️ [SKIP] 데이터가 비어있음: {input_filename}")
            continue

        # 2. 절반 나누기 (Slicing)
        # 몫(Integer Division)을 기준으로 나눕니다.
        # 예: 50개 -> mid=25 -> 0~24(25개) / 25~49(25개)
        # 예: 51개 -> mid=25 -> 0~24(25개) / 25~50(26개)
        mid_index = total_count // 2 

        part_1_data = data[:mid_index]
        part_2_data = data[mid_index:]

        # 3. 저장할 파일명 구성
        output_name_1 = f"{category}_sampled_half_1.json"
        output_name_2 = f"{category}_sampled_half_2.json"
        
        output_path_1 = os.path.join(BASE_DIR, output_name_1)
        output_path_2 = os.path.join(BASE_DIR, output_name_2)

        # 4. 파일 쓰기
        with open(output_path_1, "w", encoding="utf-8") as f1:
            json.dump(part_1_data, f1, indent=4, ensure_ascii=False)
            
        with open(output_path_2, "w", encoding="utf-8") as f2:
            json.dump(part_2_data, f2, indent=4, ensure_ascii=False)

        print(f"✅ [{category}] 분할 완료")
        print(f"   - 원본: {total_count}개")
        print(f"   - 저장1 ({output_name_1}): {len(part_1_data)}개")
        print(f"   - 저장2 ({output_name_2}): {len(part_2_data)}개")
        print("-" * 40)

    print("\n🏁 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()