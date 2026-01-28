import json

import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def time_line_segments():

    # 1. 설정 및 데이터 로드
    BASE_DIR = r"D:\trainDataset\localprops"
    FILE_PATH = os.path.join(BASE_DIR, "transactions_train_filtered.json")
    SAVE_IMAGE_PATH = os.path.join(BASE_DIR, "data_distribution_analysis.png") # 저장할 파일명

    print(f"Reading data from {FILE_PATH}...")

    # 메모리 효율을 위해 t_dat만 로드
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 날짜 데이터만 추출
    dates = [item['t_dat'] for item in raw_data]
    del raw_data 

    # DataFrame 변환 및 정렬
    print("Processing dates...")
    df = pd.DataFrame(dates, columns=['t_dat'])
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    df = df.sort_values('t_dat').reset_index(drop=True)

    total_count = len(df)
    print(f"Total Transactions: {total_count}")

    # ---------------------------------------------------------
    # 날짜 기준 설정 (최근 1년 / 6개월)
    # ---------------------------------------------------------
    TARGET_END_DATE = pd.Timestamp("2020-09-22")
    START_DATE_1Y = TARGET_END_DATE - pd.DateOffset(years=1)
    START_DATE_6M = TARGET_END_DATE - pd.DateOffset(months=6)

    # ---------------------------------------------------------
    # 시각화 함수
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    def plot_window(ax, start_date, end_date, title_text, color_fill):
        # 전체 데이터 히스토그램 (배경 - 회색)
        ax.hist(df['t_dat'], bins=100, color='lightgray', alpha=0.5, label='Excluded Data')
        
        # 선택된 기간 데이터 필터링
        mask = (df['t_dat'] >= start_date) & (df['t_dat'] <= end_date)
        selected_data = df.loc[mask, 't_dat']
        count = len(selected_data)
        ratio = (count / total_count) * 100
        
        # 선택된 구간 히스토그램 (강조색)
        ax.hist(selected_data, bins=30, color=color_fill, edgecolor='black', alpha=0.8, label='Selected Data')
        
        # 기준선 (Start Date)
        ax.axvline(start_date, color='red', linestyle='--', linewidth=2, label=f'Start: {start_date.date()}')
        
        # 스타일링
        ax.set_title(f'{title_text}\n(Data Count: {count:,} / {ratio:.1f}%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Transaction Count')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 텍스트 주석
        ax.text(start_date, ax.get_ylim()[1]*0.7, f'  Start: {start_date.date()}', color='red', fontweight='bold')

    # 그래프 1: 최근 1년 데이터
    plot_window(axes[0], START_DATE_1Y, TARGET_END_DATE, "Scenario A: Last 1 Year Data", "royalblue")

    # 그래프 2: 최근 6개월 데이터
    plot_window(axes[1], START_DATE_6M, TARGET_END_DATE, "Scenario B: Last 6 Months Data", "forestgreen")

    # X축 포맷팅
    axes[1].set_xlabel('Date')
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()

    # ---------------------------------------------------------
    # ★ 이미지 저장 코드 추가 ★
    # ---------------------------------------------------------
    print(f"Saving graph to {SAVE_IMAGE_PATH}...")
    # dpi=300: 고해상도 저장
    # bbox_inches='tight': 여백 잘림 방지
    plt.savefig(SAVE_IMAGE_PATH, dpi=300, bbox_inches='tight') 
    print("Save Complete!")

    plt.show()



import json
import os

def extract_unique_descriptions_but_allow_duplicate_nulls():
    # 1. 입력 파일 경로 (Raw String)
    input_path = r"D:\trainDataset\localprops\filtered_data.json"
    
    # 2. 출력 파일 경로 설정
    base_dir = os.path.dirname(input_path)
    output_desc_path = os.path.join(base_dir, "articles_detail_desc_2.json")
    output_id_path = os.path.join(base_dir, "articles_ids_2.json")

    print(f"입력 파일 읽는 중: {input_path}")

    try:
        # 3. JSON 파일 로드
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        unique_descs = []
        filtered_ids = []
        seen_descs = set() # 일반 텍스트 중복 체크를 위한 집합

        # 4. 데이터 순회
        if isinstance(data, list):
            for item in data:
                # [중요 변경] get("", "") 대신 get()을 써서 null을 None으로 받습니다.
                desc = item.get("detail_desc") 
                aid = item.get("article_id")

                # --- [로직 변경 구간] ---
                # Case 1: 값이 Null(None)인 경우 -> 중복 체크 없이 무조건 추가
                if desc is None:
                    unique_descs.append(desc)
                    filtered_ids.append(aid)
                
                # Case 2: 값이 있는 경우 -> seen_descs를 통해 중복 제거 수행
                elif desc not in seen_descs:
                    seen_descs.add(desc)
                    unique_descs.append(desc)
                    filtered_ids.append(aid)
                # -----------------------
                    
        elif isinstance(data, dict):
            # 데이터가 1개인 경우 바로 추가
            unique_descs = [data.get("detail_desc")]
            filtered_ids = [data.get("article_id")]
        
        # 5. 설명(desc) 파일 저장
        with open(output_desc_path, 'w', encoding='utf-8') as f:
            json.dump(unique_descs, f, ensure_ascii=False, indent=4)

        # 6. ID 파일 저장
        with open(output_id_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_ids, f, ensure_ascii=False, indent=4)
            
        print("-" * 30)
        print("작업 완료!")
        print(f"원본 데이터 개수: {len(data) if isinstance(data, list) else 1}")
        print(f"처리된 데이터 개수: {len(unique_descs)}")
        print(">> 일반 텍스트는 중복 제거됨 / Null 값은 중복 허용됨")
        print(f"1. 설명 저장 완료: {output_desc_path}")
        print(f"2. ID 저장 완료: {output_id_path}")

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다.\n경로를 확인해주세요: {input_path}")
    except json.JSONDecodeError:
        print("오류: JSON 파일 형식이 올바르지 않습니다.")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    extract_unique_descriptions_but_allow_duplicate_nulls()
