# article
## product_id(1개), article_id(n)

# filtered_item
## article_id(1개)

# re_data -> filtered_item에 매핑, Merge 필요 


# filtered_data == re_data 위치 동일

# filtered_data(article_id) -> <search : article> 
##  <find in> [article_id in product_id] 
## output : product_id -- <list>article_ids 

# re_data [in tokendata] --<concat in>--> filtered_data [<list>article id]Output_1
## Query : filtered_data(<List>Output) V: re_data 
import json
import os

import json
import os

def merge_reinforced_features():
    # ==========================================
    # 1. 경로 설정
    # ==========================================
    
    # Target Data 경로 (여기에 reinforced_feature를 추가함)
    FILTERED_DATA_DIR = r"D:\trainDataset\localprops"
    FILTERED_FILE_NAME = "filtered_data.json"
    FILTERED_FILE_PATH = os.path.join(FILTERED_DATA_DIR, FILTERED_FILE_NAME)

    # Reference Data 경로 (여기서 데이터를 가져옴)
    RESULT_DATA_DIR = r"C:\Users\candyform\Desktop\inferenceCode\localprops\results"
    RESULT_FILE_NAME = "final_ordered_result.json"
    RESULT_FILE_PATH = os.path.join(RESULT_DATA_DIR, RESULT_FILE_NAME)

    # 결과 저장 경로
    OUTPUT_FILE_PATH = os.path.join(FILTERED_DATA_DIR, "filtered_data_reinforced.json")

    print(f"[1] 데이터 로드 시작")
    print(f" - Filtered Data: {FILTERED_FILE_PATH}")
    print(f" - Result Data  : {RESULT_FILE_PATH}")

    try:
        # ==========================================
        # 2. JSON 파일 로드
        # ==========================================
        with open(FILTERED_FILE_PATH, 'r', encoding='utf-8') as f:
            filtered_data = json.load(f)
            
        with open(RESULT_FILE_PATH, 'r', encoding='utf-8') as f:
            final_results = json.load(f)

        # [로그 추가] 전체 데이터 개수 계산
        total_count = len(filtered_data) if isinstance(filtered_data, list) else 1
        print(f" >> filtered_data 로드 완료 (총 {total_count}개 항목)")

        # ==========================================
        # 3. 고속 검색을 위한 매핑 테이블 생성
        # ==========================================
        exclude_keys = {"text", "similarity_score", "key_correct"}
        result_map = {}
        
        for item in final_results:
            text_key = item.get("text")
            if text_key:
                filtered_props = {
                    k: v for k, v in item.items() if k not in exclude_keys
                }
                result_map[text_key] = filtered_props

        # ==========================================
        # 4. 데이터 병합 (Reinforced Feature 추가)
        # ==========================================
        match_count = 0
        
        print(" >> 데이터 매칭 작업 시작...")

        if isinstance(filtered_data, list):
            for item in filtered_data:
                desc = item.get("detail_desc")
                
                # desc가 존재하고(null 아님), 매핑 테이블에 키가 있다면
                if desc and (desc in result_map):
                    item["reinforced_feature"] = result_map[desc]
                    match_count += 1
        
        elif isinstance(filtered_data, dict):
             desc = filtered_data.get("detail_desc")
             if desc and (desc in result_map):
                 filtered_data["reinforced_feature"] = result_map[desc]
                 match_count = 1

        # ==========================================
        # 5. 결과 파일 저장
        # ==========================================
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)

        print("-" * 30)
        print("작업 완료!")
        print(f" - 원본 데이터 개수: {total_count}")
        print(f" - 매칭 성공 개수  : {match_count}")
        print(f" - 매칭률          : {(match_count / total_count) * 100:.2f}%")
        print(f"저장된 파일 경로: {OUTPUT_FILE_PATH}")

    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다.\n{e}")
    except json.JSONDecodeError:
        print("오류: JSON 파일 형식이 올바르지 않습니다.")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    merge_reinforced_features()