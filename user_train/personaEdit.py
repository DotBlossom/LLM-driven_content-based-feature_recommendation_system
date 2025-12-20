from datetime import datetime, timedelta

# 앞서 정의한 클래스들 (ActionType, Season 등)을 사용한다고 가정

# ==========================================
# Case 1. [20대 남성] "한파 대비 바지 하나만 필요해" (Single Item / Purpose Driven)
# - 특징: UPT 1, 검색 후 바로 구매 (High Intent)
# - 선택 상품: 32441 (Winter Slim Fit Pants) - 겨울용, 두꺼움
# ==========================================
case_1_male_single = {
    "user_profile": {
        "user_id": 1001,
        "gender": "man",
        "age_group": "20s",
        "style_preference": "Minimal"
    },
    "session_data": {
        "session_id": "sess_m_001_winter",
        "season": "Winter",
        "events": [
            # 1. 탐색 (클릭)
            {
                "product_id": 32441, 
                "action_type": 1, # CLICK
                "timestamp": datetime(2024, 12, 10, 19, 30, 0)
            },
            # 2. 장바구니 담기
            {
                "product_id": 32441, 
                "action_type": 3, # CART
                "timestamp": datetime(2024, 12, 10, 19, 32, 0)
            },
            # 3. 구매 (Positive Target)
            {
                "product_id": 32441, 
                "action_type": 5, # PURCHASE
                "timestamp": datetime(2024, 12, 10, 19, 35, 0)
            }
        ]
    }
}

# ==========================================
# Case 2. [20대 남성] "겨울 남친룩 풀착장" (Multi Item / Cross-Category)
# - 특징: UPT 3, 상의->하의->이너 순서로 흐름
# - 선택 상품: 
#   1. 273582 (Brown Wool Cardigan, Winter) - 메인 아우터 느낌
#   2. 25532 (Blue Long-sleeve T-shirt, Winter) - 이너 포인트
#   3. 205643 (Wool Blazer) -> 클릭만 하고 구매 안 함 (비교군)
#   4. 17081 (Black Pants, Spring/Fall) - 겨울에도 입을 기본 바지
# ==========================================
case_2_male_multi = {
    "user_profile": {
        "user_id": 1002,
        "gender": "man",
        "age_group": "20s",
        "style_preference": "Dandy/Classic"
    },
    "session_data": {
        "session_id": "sess_m_002_winter",
        "season": "Winter",
        "events": [
            # 흐름 1: 메인 가디건 탐색
            {"product_id": 273582, "action_type": 1, "timestamp": datetime(2024, 12, 15, 14, 0, 0)}, # Brown Cardigan
            
            # 흐름 2: 같이 입을 이너 탐색
            {"product_id": 25532, "action_type": 1, "timestamp": datetime(2024, 12, 15, 14, 2, 0)},  # Blue T-shirt
            {"product_id": 25532, "action_type": 3, "timestamp": datetime(2024, 12, 15, 14, 3, 0)},  # CART T-shirt
            
            # 흐름 3: 아우터 비교 (자켓을 봤으나 사지 않음 - Negative/Hard Negative 후보)
            {"product_id": 205643, "action_type": 1, "timestamp": datetime(2024, 12, 15, 14, 10, 0)}, # Wool Blazer
            
            # 흐름 4: 다시 가디건으로 돌아와서 장바구니
            {"product_id": 273582, "action_type": 3, "timestamp": datetime(2024, 12, 15, 14, 12, 0)}, # CART Cardigan
            
            # 흐름 5: 바지 매칭
            {"product_id": 17081, "action_type": 1, "timestamp": datetime(2024, 12, 15, 14, 20, 0)},  # Black Pants
            
            # 흐름 6: 일괄 구매 (가디건, 티셔츠, 바지)
            {"product_id": 273582, "action_type": 5, "timestamp": datetime(2024, 12, 15, 14, 25, 0)},
            {"product_id": 25532, "action_type": 5, "timestamp": datetime(2024, 12, 15, 14, 25, 0)},
            {"product_id": 17081, "action_type": 5, "timestamp": datetime(2024, 12, 15, 14, 25, 0)}
        ]
    }
}

# ==========================================
# Case 3. [20대 여성] "기본 검정 코트 찾기" (Single Item / Mock Data)
# - 상황: 여성 상품 데이터가 없으므로 가상 ID 사용 (90000번대)
# ==========================================
case_3_female_single = {
    "user_profile": {
        "user_id": 2001,
        "gender": "woman",
        "age_group": "20s",
        "style_preference": "Chic"
    },
    "session_data": {
        "session_id": "sess_w_001_winter",
        "season": "Winter",
        "events": [
            # 가상 상품: 90001 (Women Black Wool Coat)
            {"product_id": 90001, "action_type": 1, "timestamp": datetime(2024, 12, 20, 10, 0, 0)},
            {"product_id": 90001, "action_type": 5, "timestamp": datetime(2024, 12, 20, 10, 5, 0)}
        ]
    }
}

# ==========================================
# Case 4. [20대 여성] "연말 파티룩 코디" (Multi Item / Tone-on-Tone)
# - 상황: 상의(Knit) + 하의(Skirt) 조합 구매
# ==========================================
case_4_female_multi = {
    "user_profile": {
        "user_id": 2002,
        "gender": "woman",
        "age_group": "20s",
        "style_preference": "Feminine"
    },
    "session_data": {
        "session_id": "sess_w_002_winter",
        "season": "Winter",
        "events": [
            # 가상 상품: 90005 (Ivory Angora Knit)
            {"product_id": 90005, "action_type": 1, "timestamp": datetime(2024, 12, 22, 18, 0, 0)},
            
            # 가상 상품: 90006 (Check Mini Skirt) - 스타일 매칭
            {"product_id": 90006, "action_type": 1, "timestamp": datetime(2024, 12, 22, 18, 5, 0)},
            
            # 동시 구매
            {"product_id": 90005, "action_type": 5, "timestamp": datetime(2024, 12, 22, 18, 10, 0)},
            {"product_id": 90006, "action_type": 5, "timestamp": datetime(2024, 12, 22, 18, 10, 0)}
        ]
    }
}

# 전체 데이터셋 리스트화
synthetic_dataset = [
    case_1_male_single, 
    case_2_male_multi, 
    case_3_female_single, 
    case_4_female_multi
]