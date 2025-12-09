# vocab.py
import threading
# 0번은 Padding/Unknown 용도로 예약
PAD_TOKEN = "<PAD>"
PAD_ID = 0
RE_VOCAB_LOCK = threading.Lock()

from transformers import AutoTokenizer

# 1. 가볍고 성능 좋은 Tokenizer 로드 (한 번만 로드)
# 한국어/영어 혼합이라면 'bert-base-multilingual-cased' 또는 경량화된 모델 추천
TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
TITLE_MAX_LEN = 32  # 제목은 핵심만 남겼으므로 길지 않게 설정

# Disentangled Representation 위한거임../ 그리고 cross attention으로 qyery 강화


# 도메인별 유효 값 정의 (Finite Domain)

STD_VOCAB_CONFIG = {
    "category": [
        "01outer_01coat", "01outer_02jacket", "01outer_03jumper", "01outer_04cardigan",
        "02top_01blouse", "02top_02t-shirt", "02top_03sweater", "02top_04shirt", "02top_05vest",
        "03-1onepiece(dress)", "03-2onepiece(jumpsuite)",
        "04bottom_01pants", "04bottom_02skirt"
    ],
    "season": ["spring&fall", "summer", "winter"],
    "fiber_composition": [
        "Cotton", "Hemp", "cellulose fiber Others", "Silk", "Wool", "protein fiber Others",
        "Viscos rayon", "regenerated fiber Others", "Polyester", "Nylon", "Polyurethane", "synthetic fiber Others"
    ],
    "elasticity": ["none at all", "none", "contain", "contain little", "contain a lot"],
    "transparency": ["none at all", "none", "contain", "contain little", "contain a lot"],
    "isfleece": ["fleece_contain", "fleece_none"],
    "color": [
        "Black", "White", "Gray", "Red", "Orange", "Pink", "Yellow", "Brown", "Green", "Blue", "Purple", "Beige", "Mixed"
    ],
    "gender": ["male", "female", "both"],
    "category_specification": ["outer", "top", "onepiece", "bottom"],
    
    # --- Detailed Attributes ---
    "top.length_type": ["crop", "nomal", "long", "midi", "short"],
    "top.sleeve_length_type": ["sleeveless", "short sleeves", "long sleeves"],
    "top.neck_color_design": [
        "shirts collar", "bow collar", "sailor collar", "shawl collar", "polo collar", "Peter Pan collar",
        "tailored collar", "Chinese collar", "band collar", "hood", "round neck", "U-neck", "V-neck",
        "halter neck", "off shoulder", "one shoulder", "square neck", "turtle neck", "boat neck",
        "cowl neck", "sweetheart neck", "no neckline", "Others"
    ],
    "top.sleeve_design": [
        "basic sleeve", "ribbed sleeve", "shirt sleeve", "puff sleeve", "cape sleeve", "petal sleeve", "Others"
    ],
    "pant.silhouette": ["skinny", "normal", "wide", "loose", "bell-bottom", "Others"],
    "skirt.design": ["A-line and bell line", "mermaid line", "Others"],
    
    # --- Metadata (Optional) ---

    "metadata_keywords": [] 
}

# 추가 dummy 필요 1개씩 or zero, key는 있어야지.
GENERATED_METADATA = {
    "category" : [],
    "fiber_composition":[],
    "color": []

}

# --- A. STD Vocabulary 구축 (정적) ---

ALL_STD_TOKENS = set()
for key, values in STD_VOCAB_CONFIG.items():
    for v in values:
        ALL_STD_TOKENS.add(v)

SORTED_STD_TOKENS = sorted(list(ALL_STD_TOKENS))
# ID는 1부터 시작 (0은 PAD). STD와 RE 모두 이 로직을 따릅니다.
STD_TOKEN_TO_ID = {token: i + 1 for i, token in enumerate(SORTED_STD_TOKENS)}
STD_VOCAB_SIZE = len(STD_TOKEN_TO_ID) + 1  # (STD 토큰 개수) + PAD(1)


# --- B. RE Vocabulary 구축 (동적 관리 대상) ---

# RE 사전을 관리하는 전역 딕셔너리
# 초기 GENERATED_METADATA 로드 및 ID 할당 로직 (STD와 동일한 ID 부여 로직 적용)
INITIAL_RE_TOKENS = set()
for key, new_values in GENERATED_METADATA.items():
    for value in new_values:
        # STD와 중복되지 않는 고유한 값만 RE 토큰으로 등록
        if value not in ALL_STD_TOKENS:
            INITIAL_RE_TOKENS.add(value)

SORTED_INITIAL_RE_TOKENS = sorted(list(INITIAL_RE_TOKENS))

# 초기 RE 사전 구축: ID는 1부터 시작 (0은 PAD)
RE_TOKEN_TO_ID = {token: i + 1 for i, token in enumerate(SORTED_INITIAL_RE_TOKENS)}

# 동적 ID 할당을 위한 카운터: 이미 할당된 다음 ID로 설정
_RE_ID_COUNTER = len(RE_TOKEN_TO_ID) + 1





def get_std_id(value: str) -> int:
    """
    STD 토큰에 대한 정수 ID (1부터 시작)를 반환합니다.
    STD에 없는 값은 0 (PAD)을 반환합니다.
    """
    if value is None:
        return PAD_ID
    return STD_TOKEN_TO_ID.get(str(value), PAD_ID)


def get_re_id(value: str) -> int:
    """
    RE 토큰에 대한 정수 ID (1부터 시작)를 반환합니다.
    
    1. 값이 STD 토큰에 있으면 0 (PAD)을 반환합니다.
    2. RE 사전에 값이 있으면 해당 ID를 반환합니다.
    3. RE 사전에 값이 없으면, 새로운 ID를 부여하고 사전에 등록한 후 반환합니다.
    """
    global _RE_ID_COUNTER
    if value is None:
        return PAD_ID
    
    str_value = str(value)

    # 1. STD 토큰과 중복 검사 (STD 토큰은 RE 인코더로 들어갈 수 없음)
    if str_value in STD_TOKEN_TO_ID:
        return PAD_ID
    
    # 2. RE 사전에 이미 있는지 확인
    if str_value in RE_TOKEN_TO_ID:
        return RE_TOKEN_TO_ID[str_value]
    
    # 3. 새로운 RE 토큰인 경우: 락을 걸고 등록
    with RE_VOCAB_LOCK:
        # 락을 잡은 후 다시 한번 확인 (Race Condition 방지)
        if str_value in RE_TOKEN_TO_ID:
            return RE_TOKEN_TO_ID[str_value]

        # 새로운 ID 할당 및 카운터 증가
        new_id = _RE_ID_COUNTER
        RE_TOKEN_TO_ID[str_value] = new_id
        _RE_ID_COUNTER += 1
        
        # print(f"DEBUG: New RE token registered: '{str_value}' -> {new_id}")
        return new_id


def get_vocab_sizes() -> tuple[int, int]:
    """
    STD의 전체 어휘 크기 (PAD 포함)와 현재 RE의 전체 어휘 크기를 반환합니다.
    """
    # RE 크기는 PAD(1) + 현재 등록된 토큰 개수
    current_re_size = len(RE_TOKEN_TO_ID) + 1
    return STD_VOCAB_SIZE, current_re_size

