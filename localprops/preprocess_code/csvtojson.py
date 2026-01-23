import pandas as pd
import os

# 1. 경로 설정
target_dir = r'C:\Users\candyform\Desktop\inferenceCode\localprops'
input_file = os.path.join(target_dir, 'articles.csv')
output_file = os.path.join(target_dir, 'articles.json')

input_file_t = os.path.join(target_dir, 'transactions_train.csv')
output_file_t = os.path.join(target_dir, 'transactions_train.json')
def csv_to_json_article(input_f, output_f, target):

    if not os.path.exists(target):
        print(f"경로를 찾을 수 없습니다: {target}")
    else:
        try:
            # 2. CSV 파일 읽기
            # 파일이 해당 경로에 있어야 합니다.
            df = pd.read_csv(input_f)

            # 3. JSON으로 저장
            # orient='records': 행 단위로 객체 생성
            # force_ascii=False: 한글/특수문자 유지
            # indent=4: 가독성을 위한 들여쓰기
            df.to_json(output_f, orient='records', force_ascii=False, indent=4)

            print(f"변환 완료! 파일 저장 위치: {output_f}")
        
        except FileNotFoundError:
            print(f"파일을 찾을 수 없습니다. {input_f} 파일이 해당 폴더에 있는지 확인해주세요.")
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            
            
def csv_to_json_article_transaction(input_f, output_f):
    try:
        # dtype=str: 모든 ID 컬럼을 문자열로 지정하여 '0'이 잘리는 것을 방지합니다.
        df = pd.read_csv(input_f, dtype={'article_id': str, 'customer_id': str})

        # 3. 데이터 확인 (선택 사항)
        # 데이터가 제대로 로드되었는지 상위 5개 행을 확인합니다.
        print("데이터 로드 성공! 상위 5개 행:")
        print(df.head())

        # 4. JSON으로 저장
        # orient='records': [{컬럼:값}, {컬럼:값}...] 형태의 리스트 구조
        # indent=4: 들여쓰기를 적용하여 보기 좋게 저장
        df.to_json(output_f, orient='records', force_ascii=False, indent=4)

        print(f"\n변환 완료! '{output_f}' 파일이 생성되었습니다.")

    except FileNotFoundError:
            print(f"에러: '{input_f}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    except Exception as e:
            print(f"에러 발생: {e}")
            
            
            
csv_to_json_article_transaction(input_file_t, output_file_t)
