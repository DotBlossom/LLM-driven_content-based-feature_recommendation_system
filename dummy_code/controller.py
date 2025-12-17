'''
def preprocess_batch_input(products: List[ProductInput]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    [Residual Field Embedding용 전처리]
    딕셔너리를 순회하는 것이 아니라, 고정된 'ALL_FIELD_KEYS'를 순회하여
    Tensor의 각 인덱스가 항상 특정 필드(속성)를 가리키도록 정렬합니다.
    """
    batch_std_ids = []
    batch_re_ids = []
    
    for product in products:
        # 1. 데이터 추출
        feature_data: Dict[str, Any] = getattr(product, 'feature_data', {})
        clothes_data = feature_data.get("clothes", {})
        re_data = feature_data.get("reinforced_feature_value", {})
        
        row_std_ids = []
        row_re_ids = []

        # 2. [핵심] 고정된 Key 리스트를 순회 (순서 및 위치 보장)
        for key in ALL_FIELD_KEYS:
            
            # --- A. STD ID 추출 ---
            std_val = clothes_data.get(key)
            
            # 리스트로 들어오는 경우 첫 번째 값 사용 (단일 라벨 가정)
            if isinstance(std_val, list) and len(std_val) > 0:
                std_val = std_val[0]
            elif isinstance(std_val, list) and len(std_val) == 0:
                std_val = None
                
            # vocab.py의 함수 호출 (Key 정보도 함께 전달하여 확장성 확보)
            # 값이 없으면(None) 내부에서 0(PAD) 반환
            s_id = vocab.get_std_id(key, std_val)
            row_std_ids.append(s_id)
            
            
            # --- B. RE ID 추출 (Hashing) ---
            re_val_list = re_data.get(key)
            re_val = None
            
            # RE 데이터는 보통 List 형태이므로 첫 번째 값 추출
            if re_val_list and isinstance(re_val_list, list) and len(re_val_list) > 0:
                re_val = re_val_list[0]
            elif isinstance(re_val_list, str):
                re_val = re_val_list
            
            # Hashing 함수 호출 (저장 X, 즉시 변환)
            # 값이 없으면(None) 내부에서 0(PAD) 반환
            r_id = vocab.get_re_hash_id(re_val)
            row_re_ids.append(r_id)

        # 3. 행 단위 추가
        # 이제 row_std_ids의 길이는 항상 len(ALL_FIELD_KEYS)로 고정됨
        batch_std_ids.append(row_std_ids)
        batch_re_ids.append(row_re_ids)
    
    # 4. 텐서 변환 (pad_sequence 불필요 -> torch.tensor로 직변환)
    # Shape: (Batch_Size, Num_Fields)
    t_std_batch = torch.tensor(batch_std_ids, dtype=torch.long, device=DEVICE)
    t_re_batch = torch.tensor(batch_re_ids, dtype=torch.long, device=DEVICE)

    return t_std_batch, t_re_batch
    '''