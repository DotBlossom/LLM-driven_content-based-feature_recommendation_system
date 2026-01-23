import json
import os

def extract_unique_descriptions_and_ids():
    # 1. 입력 파일 경로 (Raw String)
    input_path = r"C:\Users\candyform\Desktop\inferenceCode\localprops\filtered_data.json"
    
    # 2. 출력 파일 경로 설정 (입력 파일과 같은 폴더)
    base_dir = os.path.dirname(input_path)
    output_desc_path = os.path.join(base_dir, "articles_detail_desc.json")
    output_id_path = os.path.join(base_dir, "articles_ids.json")

    print(f"입력 파일 읽는 중: {input_path}")

    try:
        # 3. JSON 파일 로드
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        unique_descs = []
        filtered_ids = []
        seen_descs = set() # 중복 체크를 위한 집합(Set)

        # 4. 데이터 순회 및 중복 제거
        if isinstance(data, list):
            for item in data:
                # detail_desc 가져오기 (없으면 빈 문자열)
                desc = item.get("detail_desc", "")
                aid = item.get("article_id", "")

                # 이 설명이 이전에 나온 적이 없는 경우에만 추가 (중복 패스)
                if desc not in seen_descs:
                    seen_descs.add(desc)
                    unique_descs.append(desc)
                    filtered_ids.append(aid)
                    
        elif isinstance(data, dict):
            # 데이터가 1개인 경우 바로 추가
            unique_descs = [data.get("detail_desc", "")]
            filtered_ids = [data.get("article_id", "")]
        
        # 5. 설명(desc) 파일 저장
        with open(output_desc_path, 'w', encoding='utf-8') as f:
            json.dump(unique_descs, f, ensure_ascii=False, indent=4)

        # 6. ID 파일 저장
        with open(output_id_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_ids, f, ensure_ascii=False, indent=4)
            
        print("-" * 30)
        print("작업 완료!")
        print(f"원본 데이터 개수: {len(data) if isinstance(data, list) else 1}")
        print(f"중복 제거 후 개수: {len(unique_descs)}")
        print(f"1. 설명 저장 완료: {output_desc_path}")
        print(f"2. ID 저장 완료: {output_id_path}")

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다.\n경로를 확인해주세요: {input_path}")
    except json.JSONDecodeError:
        print("오류: JSON 파일 형식이 올바르지 않습니다.")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    extract_unique_descriptions_and_ids()