from collections import Counter
import json
import math
import os

# 1. 파일 경로 설정
input_path = r"C:\Users\candyform\Desktop\inferenceCode\localprops\articles.json"      
output_data_path = r"C:\Users\candyform\Desktop\inferenceCode\localprops\filtered_data.json" # 저장할 파일 경로
output_stats_path = r"C:\Users\candyform\Desktop\inferenceCode\localprops\unique_stats.json"  # 통계 저장 경로
# 삭제하지 않고 남겨둘 숫자형 ID 키
KEEP_ID_KEYS = {"article_id", "product_code"}

# 중복 없이 값을 수집할 필드 목록
COLLECT_FIELDS = [
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "index_name",
    "index_group_name",
    "department_name",
    "section_name",
]

if not os.path.exists(input_path):
    print(f"오류: '{input_path}' 파일을 찾을 수 없습니다.")
else:
    try:
        # 2. 파일 로드
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []
        
        # 유니크 값을 담을 딕셔너리 초기화 (Key: 필드명, Value: Set 자료구조)
        unique_stats = {field: set() for field in COLLECT_FIELDS}

        # 3. 데이터 순회 및 처리
        for item in data:
            # [조건 1] 필터링: Baby/Children 이거나 Underwear 인 경우 제외
            # 이 조건에 걸리면 continue로 넘어가므로 집계도 되지 않습니다.
            if (item.get("index_group_name") == "Baby/Children" or 
                item.get("product_group_name") == "Underwear"):
                continue

            # [조건 2] 유니크 값 집계 (필터링 통과한 데이터만)
            for field in COLLECT_FIELDS:
                value = item.get(field)
                if value:  # 값이 존재하는 경우에만 추가
                    unique_stats[field].add(value)

            # [조건 3] 데이터 정제 (숫자 필드 삭제 로직)
            new_item = {}
            for key, value in item.items():
                is_number = isinstance(value, (int, float))
                
                # 숫자가 아니거나, 남겨야 할 ID 키인 경우에만 저장
                if not is_number or key in KEEP_ID_KEYS:
                    new_item[key] = value
            
            processed_data.append(new_item)

        # 4. 결과 저장 (정제된 데이터)
        with open(output_data_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)

        # 5. 집계 결과 저장 (유니크 값들)
        # Set은 JSON 변환이 안되므로 List로 변환 필요
        serializable_stats = {k: list(v) for k, v in unique_stats.items()}
        
        with open(output_stats_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=4, ensure_ascii=False)

        print("-" * 30)
        print("작업 완료!")
        print(f"원본 데이터 개수: {len(data)}")
        print(f"필터링 후 저장된 데이터 개수: {len(processed_data)}")
        print("-" * 30)
        print("집계된 유니크 값 목록 (상위 5개씩만 미리보기):")
        for key, values in serializable_stats.items():
            print(f"[{key}]: {len(values)}개 발견 -> {values[:5]} ...")

    except Exception as e:
        print(f"오류 발생: {e}")

def count_product_groups_fixed_path():
    # 1. 파일 경로 설정
    input_path = r"C:\Users\candyform\Desktop\inferenceCode\localprops\articles.json"
    
    # 결과 파일 저장 경로 (같은 폴더에 'group_counts.json'으로 저장)
    output_path = os.path.join(os.path.dirname(input_path), "group_counts.json")

    # 2. 기준이 되는 카테고리 리스트 (사용자 제공)
    target_categories = [
        "Shoes",
        "Garment Lower body",
        "Stationery",
        "Unknown",
        "Garment Upper body",
        "Accessories",
        "Socks & Tights",
        "Nightwear",
        "Bags",
        "Garment Full body",
        "Swimwear",
        "Items",
        "Garment and Shoe care",
        "Furniture"
    ]

    print(f"데이터 읽는 중... {input_path}")

    try:
        # 3. 데이터 로드
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 데이터가 리스트가 아니라 단일 객체일 경우 리스트로 변환
        if isinstance(data, dict):
            data = [data]

        # 4. 개수 세기 (Counter 사용)
        # 데이터 내의 모든 product_group_name을 추출하여 카운트
        real_counts = Counter([item.get("product_group_name", "Unknown") for item in data])

        # 5. 결과 정리 (순서 유지 및 딕셔너리 생성)
        final_result = {}
        
        print("\n" + "="*40)
        print(" [ product_group_name 별 개수 집계 ]")
        print("="*40)

        # (1) 제공된 리스트 순서대로 출력
        for category in target_categories:
            count = real_counts.get(category, 0) # 데이터에 없으면 0으로 처리
            final_result[category] = count
            # 보기 좋게 정렬하여 출력 (ljust는 왼쪽 정렬)
            print(f"{category.ljust(25)} : {count} 건")

        # (2) 제공된 리스트에는 없는데, 실제 데이터에는 있는 항목이 있는지 체크
        existing_keys = set(real_counts.keys())
        target_keys = set(target_categories)
        extra_keys = existing_keys - target_keys

        if extra_keys:
            print("-" * 40)
            print(" [ 리스트 외 추가 발견된 항목 ]")
            for key in extra_keys:
                count = real_counts[key]
                final_result[key] = count
                print(f"{key.ljust(25)} : {count} 건")

        # 6. 결과를 JSON 파일로 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)

        print("="*40)
        print(f"집계 완료! 결과 파일이 생성되었습니다.")
        print(f"저장 위치: {output_path}")

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다.\n경로: {input_path}")
    except Exception as e:
        print(f"오류 발생: {e}")
        
        
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_product_counts():
    # 1. 파일 경로 설정
    base_dir = r"C:\Users\candyform\Desktop\inferenceCode\localprops"
    input_path = os.path.join(base_dir, "group_counts.json")
    output_img_path = os.path.join(base_dir, "group_counts_chart.png")

    print(f"데이터 로드 중: {input_path}")

    try:
        # 2. JSON 데이터 읽기
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 3. 데이터 프레임 변환 (Pandas 사용)
        # 딕셔너리를 DataFrame으로 변환 후, 개수(Count) 기준으로 내림차순 정렬
        df = pd.DataFrame(list(data.items()), columns=['Category', 'Count'])
        df = df.sort_values(by='Count', ascending=False)

        # 4. 시각화 설정
        plt.figure(figsize=(12, 8)) # 이미지 크기 (가로, 세로)
        sns.set_theme(style="whitegrid") # 배경 스타일

        # 5. 막대 그래프 그리기 (가로형)
        # x=개수, y=카테고리
        ax = sns.barplot(x='Count', y='Category', data=df, palette='viridis', hue='Category', legend=False)

        # 6. 차트 꾸미기
        plt.title('Product Group Distribution', fontsize=16, pad=20)
        plt.xlabel('Number of Items', fontsize=12)
        plt.ylabel('Category', fontsize=12)

        # 7. 막대 옆에 정확한 숫자 표시
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            # 숫자가 0보다 클 때만 표시
            if width > 0:
                ax.text(width + (width * 0.01),  # x 좌표 (막대 끝에서 약간 오른쪽)
                        p.get_y() + p.get_height() / 2, # y 좌표 (막대 중앙)
                        f'{int(width):,}',      # 천 단위 콤마 찍기
                        va='center', fontsize=10, color='black')

        # 8. 여백 조정 및 저장
        plt.tight_layout()
        plt.savefig(output_img_path, dpi=300) # 고해상도 저장
        
        print("="*40)
        print("시각화 완료!")
        print(f"이미지 저장됨: {output_img_path}")
        print("="*40)
        
        # (선택) 창 띄우기 - 서버 환경이 아니면 주석 해제하여 바로 확인 가능
        # plt.show() 

    except FileNotFoundError:
        print(f"오류: 데이터 파일({input_path})을 찾을 수 없습니다.")
        print("이전 단계의 집계 코드를 먼저 실행해주세요.")
    except Exception as e:
        print(f"오류 발생: {e}")
        

def split_json_file():
    # 1. 입력 파일 경로 (이전 단계에서 만든 파일 경로로 설정)
    input_path = r"C:\Users\candyform\Desktop\inferenceCode\localprops\articles_detail_desc.json"
    
    # 2. 쪼갤 단위 설정 (500개)
    CHUNK_SIZE = 500

    print(f"파일 읽는 중: {input_path}")

    try:
        # 파일 로드
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print("오류: JSON 파일의 최상위 구조가 리스트([])가 아닙니다.")
            return

        total_items = len(data)
        total_files = math.ceil(total_items / CHUNK_SIZE) # 총 파일 개수 계산

        # 3. 저장할 폴더 생성 (입력 파일 위치 하위에 'split_files' 폴더 생성)
        base_dir = os.path.dirname(input_path)
        output_dir = os.path.join(base_dir, "split_files")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"폴더 생성됨: {output_dir}")

        print(f"총 {total_items}개의 데이터를 {CHUNK_SIZE}개씩 나누어 저장합니다. (예상 파일 수: {total_files}개)")
        print("-" * 40)

        # 4. 데이터 분할 및 저장 루프
        for i in range(total_files):
            start_idx = i * CHUNK_SIZE
            end_idx = start_idx + CHUNK_SIZE
            
            # 리스트 슬라이싱 (마지막 조각은 알아서 남은 만큼만 잘림)
            chunk_data = data[start_idx:end_idx]
            
            # 파일명 생성 (file_1.json, file_2.json ...)
            file_name = f"file_{i + 1}.json"
            output_path = os.path.join(output_dir, file_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=4)
                
            # 진행 상황 출력 (너무 많으면 100개 단위로 출력)
            if (i + 1) % 10 == 0 or (i + 1) == total_files:
                print(f"저장 완료: {file_name} ({len(chunk_data)}개 항목)")

        print("-" * 40)
        print("모든 작업이 완료되었습니다!")
        print(f"저장 경로: {output_dir}")

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {input_path}")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    #count_product_groups_fixed_path()
    #visualize_product_counts()
    split_json_file()