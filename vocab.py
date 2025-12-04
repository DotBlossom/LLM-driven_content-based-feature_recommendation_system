# vocab.py

# 0번은 Padding/Unknown 용도로 예약
PAD_TOKEN = "<PAD>"
PAD_ID = 0

# 도메인별 유효 값 정의 (Finite Domain)
# reinforced_feature_value의 값들도 이 도메인 안에 포함된다고 가정합니다.
VOCAB_CONFIG = {
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
    # specification.metadata가 자연어라면 별도 Tokenizer가 필요하지만,
    # 만약 정형화된 태그(예: "Modern", "Classic")라면 여기에 추가하면 됩니다.
    "metadata_keywords": [] 
}

# --- 1. 통합 사전 생성 (Shared Vocabulary Build) ---
ALL_TOKENS = set()

for key, values in VOCAB_CONFIG.items():
    for v in values:
        ALL_TOKENS.add(v)

# 정렬하여 순서 보장 (매번 실행해도 ID가 바뀌지 않게 함)
SORTED_TOKENS = sorted(list(ALL_TOKENS))

# ID 매핑 생성 (0: PAD, 1~N: Tokens)
TOKEN_TO_ID = {token: i+1 for i, token in enumerate(SORTED_TOKENS)}
ID_TO_TOKEN = {i+1: token for i, token in enumerate(SORTED_TOKENS)}

# 0번 추가
ID_TO_TOKEN[PAD_ID] = PAD_TOKEN

# 전체 단어장 크기 (Embedding Layer의 input dimension)
VOCAB_SIZE = len(TOKEN_TO_ID) + 1

# --- 2. Helper Functions ---

def get_token_id(value: str) -> int:
    """
    문자열 값을 입력받아 정수 ID를 반환합니다.
    사전에 정의되지 않은 값(Unknown)은 0(PAD)을 반환하여 무시합니다.
    """
    if value is None:
        return PAD_ID
    return TOKEN_TO_ID.get(str(value), PAD_ID)

def get_token_from_id(token_id: int) -> str:
    """
    정수 ID를 입력받아 원래 문자열을 반환합니다.
    """
    return ID_TO_TOKEN.get(token_id, PAD_TOKEN)

# 디버깅용 정보 출력
if __name__ == "__main__":
    print(f"✅ Vocab Loaded!")
    print(f"Total Unique Tokens: {len(TOKEN_TO_ID)}")
    print(f"Vocab Size (incl. PAD): {VOCAB_SIZE}")
    print(f"Sample Mapping: 'Black' -> {get_token_id('Black')}")
    print(f"Sample Mapping: '01outer_01coat' -> {get_token_id('01outer_01coat')}")