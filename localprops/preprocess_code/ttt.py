import json
import os

# 1. 설정: 대상 디렉토리 및 파일 이름
BASE_DIR = r"D:\trainDataset\localprops"
FILE_NAME = "filtered_data_reinforced.json"
FILE_PATH = os.path.join(BASE_DIR, FILE_NAME)

o_FILE_NAME = "filtered_data_integ.json"
o_FILE_PATH = os.path.join(BASE_DIR, o_FILE_NAME)

def process_json_data():
    try:
        # 파일 읽기
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 제거할 키 목록
        exclude_keys = {
            "product_code", "product_group_name", 
            "colour_group_name", "perceived_colour_master_name", "index_code", 
            "index_name", "index_group_name", "garment_group_name", "detail_desc"
        }

        new_data = []

        for item in data:
            # 1. article_id를 product_id 값으로 추출
            # (데이터에 article_id가 없는 경우를 대비해 get 사용, 없으면 None)
            p_id = item.get("article_id")
            p_n = item.get("prod_name", None)
            # 2. feature_data 생성
            # 조건: exclude_keys에 포함되지 않아야 함 AND article_id가 아니어야 함
            f_data = {
                k: v for k, v in item.items()
                if k not in exclude_keys and k != "article_id" and k != "prod_name"
            }

            # 3. 최종 딕셔너리 구조 조립
            structured_item = {
                "product_id": p_id,
                "product_name": p_n,
                "feature_data": f_data
            }

            new_data.append(structured_item)

        # 3. 결과 저장
        with open(o_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=4, ensure_ascii=False)
            
        print(f"성공: {o_FILE_PATH}에 데이터가 재구조화되어 저장되었습니다.")

    except FileNotFoundError:
        print(f"오류: {FILE_PATH} 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    process_json_data()