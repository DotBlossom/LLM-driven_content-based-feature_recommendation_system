import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os

def create_stratified_folds():
    # 1. 파일 경로 설정
    input_path = r"C:\Users\candyform\Desktop\inferenceCode\localprops\filtered_data.json"
    output_path = r"C:\Users\candyform\Desktop\inferenceCode\localprops\articles_with_folds.csv"
    
    # K-Fold 설정 (보통 5 또는 10 사용)
    N_SPLITS = 5 
    SEED = 42

    print(f"데이터 로드 중... {input_path}")
    
    # 2. 데이터 로드 및 DataFrame 변환
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        

        target_col = 'product_group_name'
        
        print(f"전체 데이터 개수: {len(df)}")

        # 데이터 정제 (K개 미만인 클래스 제거)

        class_counts = df[target_col].value_counts()
        valid_classes = class_counts[class_counts >= N_SPLITS].index
        
        dropped_classes = class_counts[class_counts < N_SPLITS]
        if not dropped_classes.empty:
            print("\n[경고] 데이터 개수가 너무 적어 학습에서 제외되는 항목:")
            print(dropped_classes)
        
        # 필터링 적용
        df_filtered = df[df[target_col].isin(valid_classes)].copy()
        df_filtered = df_filtered.reset_index(drop=True) # 인덱스 초기화
        
        print(f"\n필터링 후 데이터 개수: {len(df_filtered)}")
        
        # Stratified K-Fold 적용
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        
        # 'fold' 컬럼을 만들고 초기화 (-1)
        df_filtered['fold'] = -1
        
        # X는 인덱스, y는 타겟값(product_group_name)
        for fold_num, (train_idx, val_idx) in enumerate(skf.split(df_filtered, df_filtered[target_col])):
            # 해당 인덱스에 fold 번호 부여
            df_filtered.loc[val_idx, 'fold'] = fold_num
            
            # --- 검증 (잘 나누어졌는지 확인) ---
            if fold_num == 0: # 첫 번째 폴드만 예시로 비율 확인
                print(f"\n[Fold 0 검증] Train: {len(train_idx)}, Val: {len(val_idx)}")
                val_ratios = df_filtered.iloc[val_idx][target_col].value_counts(normalize=True)
                print("검증 데이터셋(Validation) 내 클래스 비율 상위 3개:")
                print(val_ratios.head(3))


        df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*30)
        print("작업 완료!")
        print(f"저장된 파일: {output_path}")
        print("이제 학습 시 'fold' 컬럼을 기준으로 Train/Val을 나누시면 됩니다.")
        
    except FileNotFoundError:
        print("입력 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == "__main__":
    create_stratified_folds()