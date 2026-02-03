

from typing import Any, Dict, List
# H&M 데이터셋 기준 유의미한 STD 필드 / 값 실제로 다 추가
STD_VOCAB_CONFIG = {
 "product_type_name": [
        "Beanie",
        "Washing bag",
        "Waterbottle",
        "Mobile case",
        "Dog wear",
        "Heels",
        "T-shirt",
        "Wallet",
        "Cap",
        "Bootie",
        "Watch",
        "Eyeglasses",
        "Alice band",
        "Shirt",
        "Necklace",
        "Sewing kit",
        "Sarong",
        "Polo shirt",
        "Unknown",
        "Ring",
        "Skirt",
        "Top",
        "Pyjama set",
        "Vest top",
        "Sweater",
        "Hair clip",
        "Costumes",
        "Shoulder bag",
        "Outdoor Waistcoat",
        "Shorts",
        "Garment Set",
        "Other shoe",
        "Cross-body bag",
        "Hair ties",
        "Sunglasses",
        "Zipper head",
        "Earrings",
        "Hair string",
        "Wedge",
        "Hair/alice band",
        "Pumps",
        "Flip flop",
        "Bracelet",
        "Bikini top",
        "Tie",
        "Felt hat",
        "Braces",
        "Belt",
        "Bucket hat",
        "Hairband",
        "Dungarees",
        "Gloves",
        "Pyjama bottom",
        "Headband",
        "Blouse",
        "Pyjama jumpsuit/playsuit",
        "Umbrella",
        "Wood balls",
        "Flat shoe",
        "Backpack",
        "Hat/beanie",
        "Hat/brim",
        "Night gown",
        "Bag",
        "Jacket",
        "Boots",
        "Leggings/Tights",
        "Tote bag",
        "Marker pen",
        "Other accessories",
        "Cardigan",
        "Keychain",
        "Coat",
        "Bodysuit",
        "Dress",
        "Cap/peaked",
        "Swimsuit",
        "Straw hat",
        "Ballerinas",
        "Earring",
        "Dog Wear",
        "Sandals",
        "Underwear Tights",
        "Tailored Waistcoat",
        "Flat shoes",
        "Side table",
        "Hoodie",
        "Outdoor trousers",
        "Weekend/Gym bag",
        "Trousers",
        "Clothing mist",
        "Swimwear bottom",
        "Sneakers",
        "Heeled sandals",
        "Giftbox",
        "Slippers",
        "Scarf",
        "Stain remover spray",
        "Swimwear set",
        "Socks",
        "Wireless earphone case",
        "Blazer",
        "Jumpsuit/Playsuit"
    ],
    "graphical_appearance_name": [
        "Embroidery",
        "Unknown",
        "Front print",
        "Transparent",
        "Slub",
        "Jacquard",
        "Mixed solid/pattern",
        "Hologram",
        "Treatment",
        "Mesh",
        "Glittering/Metallic",
        "Denim",
        "Solid",
        "Chambray",
        "Application/3D",
        "Sequin",
        "All over pattern",
        "Metallic",
        "Placement print",
        "Other pattern",
        "Lace",
        "Stripe",
        "Contrast",
        "Other structure",
        "Colour blocking",
        "Check",
        "Melange",
        "Argyle",
        "Dot",
        "Neps"
    ],
    "colour_group_name": [
        "Unknown",
        "Transparent",
        "Other",
        "Light Green",
        "Light Red",
        "Light Grey",
        "Pink",
        "Light Turquoise",
        "Dark Grey",
        "Greyish Beige",
        "Dark Blue",
        "Dark Purple",
        "Other Pink",
        "Light Beige",
        "Other Green",
        "Dark Pink",
        "Dark Yellow",
        "Light Blue",
        "Orange",
        "White",
        "Other Blue",
        "Other Purple",
        "Blue",
        "Turquoise",
        "Dark Turquoise",
        "Beige",
        "Red",
        "Greenish Khaki",
        "Other Turquoise",
        "Black",
        "Purple",
        "Grey",
        "Other Red",
        "Other Yellow",
        "Bronze/Copper",
        "Other Orange",
        "Yellow",
        "Dark Beige",
        "Light Purple",
        "Yellowish Brown",
        "Light Orange",
        "Off White",
        "Light Pink",
        "Dark Green",
        "Light Yellow",
        "Dark Orange",
        "Silver",
        "Dark Red",
        "Gold",
        "Green"
    ],
    "department_name": [
        "AK Tops Knitwear",
        "Knit & Woven",
        "AK Tops Jersey & Woven",
        "Woven Premium",
        "Promotion/ Other /Offer",
        "Trousers DS",
        "Shoes / Boots inactive from s5",
        "Knitwear Basic",
        "Suit Extended inactive from s1",
        "Heels",
        "Scarves",
        "Divided Swimwear",
        "Woven Tops",
        "Jersey Occasion",
        "Outwear & Blazers",
        "Shirt Extended inactive from s1",
        "Woven top",
        "Bottoms Girls",
        "Woven",
        "Casual Lingerie",
        "Shirt",
        "Shirt S&T",
        "Jersey/Knitwear Premium",
        "Gloves/Hats",
        "Nursing",
        "Shoes Other",
        "Tops Fancy Jersey",
        "Accessories Boys",
        "Other items",
        "Belts",
        "Small Bags",
        "Tights basic",
        "Blouse & Dress",
        "Take Care External",
        "Knitwear inactive from s1",
        "EQ Divided Basics",
        "Mama Lingerie",
        "Dresses",
        "Socks Wall",
        "Jacket Smart",
        "Equatorial",
        "Skirt",
        "Underwear Jersey",
        "Tops Girls",
        "Functional Lingerie",
        "Bags & Items",
        "Skirts DS",
        "Tops woven DS",
        "Shorts & Skirts",
        "Conscious Exclusive",
        "Accessories",
        "Local relevance",
        "Trousers & Skirt",
        "Price Items",
        "Blanks",
        "Promotion/Other/Offer",
        "Projects Woven Tops",
        "Bags",
        "Shopbasket Socks",
        "EQ H&M Man",
        "Heavy Basic Jersey",
        "Shorts",
        "Basic 1",
        "Denim trousers",
        "Equatorial Assortment",
        "Bottoms",
        "Jacket Casual",
        "Denim shorts",
        "Sunglasses",
        "Projects Jersey & Knitwear",
        "Socks Bin",
        "Projects Woven Bottoms",
        "Jewellery Extended",
        "Bottoms Boys",
        "Tops & Bottoms Other",
        "Test Ladies",
        "Denim Other Garments",
        "Jersey Fancy",
        "Projects Dresses",
        "Jersey fancy",
        "Trouser S&T",
        "Read & React",
        "Trouser",
        "Dresses DS",
        "Flats",
        "Shoes",
        "Denim wardrobe H&M man inactive from S.6",
        "OL Extended Sizes",
        "AK Bottoms",
        "Premium Quality",
        "Take care",
        "Accessories Other",
        "On Demand",
        "Jersey",
        "Outdoor/Blazers DS",
        "Blazer S&T",
        "Jersey Basic",
        "Ladies Sport Acc",
        "Outwear",
        "Blouse",
        "EQ & Special Collections",
        "Suit",
        "Denim Trousers",
        "Woven bottoms",
        "Jackets",
        "Jacket",
        "Outdoor inactive from s1",
        "Boots",
        "Jewellery",
        "Jersey Fancy DS",
        "Clean Lingerie",
        "Skirts",
        "Jersey License",
        "Ladies Sport Bras",
        "Men Sport Tops",
        "Outdoor/Blazers",
        "Expressive Lingerie",
        "Other Accessories",
        "Men Sport Acc",
        "Light Basic Jersey",
        "Jersey inactive from S.6",
        "Hair Accessories",
        "Tops Woven",
        "Ladies Sport Bottoms",
        "Dress",
        "Knitwear",
        "Small Accessories Extended",
        "Swimwear",
        "Tops Knitwear DS",
        "Tops Boys",
        "Woven Occasion",
        "Suit jacket",
        "Projects",
        "Campaigns",
        "EQ Ladies Denim",
        "AK Dresses & Outdoor",
        "Nightwear",
        "Trousers",
        "Shorts DS",
        "UW",
        "Sneakers",
        "Limited Edition",
        "Divided Shoes",
        "Basics",
        "Men Sport Woven",
        "Divided+ inactive from s.1",
        "Special Collection",
        "Studio Collection",
        "Ladies Sport Woven",
        "Jersey inactive from s1",
        "EQ Divided Blue",
        "Jacket Street",
        "AK Other",
        "Men Sport Bottoms",
        "Small Accessories",
        "Woven bottoms inactive from S.7",
        "Everyday Waredrobe Denim",
        "Tops Knitwear",
        "Divided+",
        "Asia Assortment",
        "Socks",
        "Loungewear",
        "Blazer",
        "Woven inactive from s1",
        "Underwear Jersey Fancy inactive from s1"
    ],
  "section_name": [
        "EQ Divided",
        "Womens Everyday Collection",
        "Men Accessories",
        "Men Edition",
        "Contemporary Street",
        "Denim Men",
        "Divided Asia keys",
        "Womens Shoes",
        "Men Other 2",
        "Contemporary Casual",
        "Womens Tailoring",
        "Mama",
        "Womens Small accessories",
        "Divided Projects",
        "Womens Big accessories",
        "Men Project",
        "Men Other",
        "Special Collections",
        "Divided Basics",
        "Divided Collection",
        "Divided Accessories",
        "Divided Selected",
        "H&M+",
        "Collaborations",
        "Womens Nightwear, Socks & Tigh",
        "Men Underwear",
        "Womens Jackets",
        "Contemporary Smart",
        "Ladies H&M Sport",
        "Mens Outerwear",
        "Womens Everyday Basics",
        "Womens Swimwear, beachwear",
        "Ladies Other",
        "Divided Complements Other",
        "Womens Trend",
        "Womens Casual",
        "Men Shoes",
        "Ladies Denim",
        "Womens Premium",
        "Womens Lingerie",
        "Men H&M Sport",
        "Kids Sports",
        "Men Suits & Tailoring"
    ],
      "perceived_colour_value_name": [
        "Unknown",
        "Undefined",
        "Dusty Light",
        "Bright",
        "Dark",
        "Light",
        "Medium",
        "Medium Dusty"
    ],

}

# RE Feature Keys (9 fields)
RE_FEATURE_KEYS = [
    "[CAT]", "[MAT]", "[DET]", "[FIT]", "[FNC]", 
    "[SPC]", "[COL]", "[CTX]", "[LOC]"
]

# --- Vocab Helper Functions ---
PAD_ID = 0
UNK_ID = 1

# 모든 STD Value를 하나의 Vocab으로 통합
ALL_STD_TOKENS = sorted(list(set(
    token for values in STD_VOCAB_CONFIG.values() for token in values
)))
STD_TOKEN_TO_ID = {token: i + 2 for i, token in enumerate(ALL_STD_TOKENS)}

def get_std_vocab_size():
    return len(STD_TOKEN_TO_ID) + 2

def get_std_id(value: str) -> int:
    if not value: return PAD_ID
    return STD_TOKEN_TO_ID.get(str(value), UNK_ID)

def get_std_field_keys() -> List[str]:
    return list(STD_VOCAB_CONFIG.keys())








'''# vocab.py
import threading
import zlib
# 0번은 Padding/Unknown 용도로 예약
PAD_TOKEN = "<PAD>"
PAD_ID = 0
# RE_VOCAB_LOCK = threading.Lock()
RE_MAX_CAPACITY = 500
from transformers import AutoTokenizer

# 1. 가볍고 성능 좋은 Tokenizer 로드 (한 번만 로드)
# 한국어/영어 혼합이라면 'bert-base-multilingual-cased' 또는 경량화된 모델 추천
RE_TOKENIZER = AutoTokenizer.from_pretrained("distilbert-base-uncased")
RE_VOCAB_SIZE = RE_TOKENIZER.vocab_size  # 약 30,522개
RE_MAX_TOKEN_LEN = 12

# Disentangled Representation 위한거임.


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
    
    #others, accessory.. key 공유를 옷들과는 다르게 해서, 추후에 따로 쿼터로 뽑게 하는게
    
    # --- Re Attributes ---
    # 1. Fit & Feel
    "fit.vibe": [],          
    "silhouette.shape": [],
    "length.feeling": [],
    "fabric.feature": [],

    # 2. Geometry (Flattened) , but finited domains
    "geo_width_flow": [],      
    "geo_waist_contour": [],
    "geo_vertical_balance": [],
    "geo_shoulder_geometry": [],
    "geo_sleeve_profile": [],
    
    "geo_rise_profile": [],
    "geo_hip_to_hem_flow": [],
    "geo_waist_type": [],
    "geo_volumetric_fit": []
    
    # accessory 는 ?
}



# Exports
ORDERED_FEATURE_KEYS = list(STD_VOCAB_CONFIG.keys())

# --- A. STD Vocabulary 구축 (정적) ---

ALL_STD_TOKENS = set()
for key, values in STD_VOCAB_CONFIG.items():
    for v in values:
        ALL_STD_TOKENS.add(v)

SORTED_STD_TOKENS = sorted(list(ALL_STD_TOKENS))
# ID는 1부터 시작 (0은 PAD). STD와 RE 모두 이 로직을 따릅니다.
STD_TOKEN_TO_ID = {token: i + 1 for i, token in enumerate(SORTED_STD_TOKENS)}
STD_VOCAB_SIZE = len(STD_TOKEN_TO_ID) + 1  # (STD 토큰 개수) + PAD(1)



def get_std_id(key: str, value: str) -> int:
    """
    STD 토큰에 대한 정수 ID (1부터 시작)를 반환합니다.
    
    [수정 사항]
    Dataset 코드와의 호환성을 위해 `key` 인자를 받도록 서명을 변경했습니다.
    현재 로직에서는 Global Unique Value를 가정하므로 `key`는 사용하지 않지만,
    호출 측(Dataset)에서 (key, value)를 넘겨주므로 이를 받아줘야 TypeError가 발생하지 않습니다.
    """
    if not value:
        return PAD_ID
    
    # 입력값이 config에 정의된 문자열과 정확히 일치해야 함
    # key는 현재 로직에서 참조하지 않음 (값 자체가 유니크하므로)
    return STD_TOKEN_TO_ID.get(str(value), PAD_ID)
def get_re_hash_id(value: str) -> int:
    """
    [Stateless Hashing]
    RE 토큰(가변적인 상세 속성)을 고정된 범위의 ID로 즉시 변환합니다.
    저장소가 필요 없으며, 서버 재부팅 시에도 동일한 ID를 보장합니다.
    """
    if not value or value == "":
        return PAD_ID
    
    # 1. 문자열을 바이트로 변환 후 CRC32 체크섬 계산 (OS/Platform 독립적)
    # 2. 버킷 사이즈로 나눈 나머지 계산
    # 3. 0은 PAD이므로 +1 하여 1 ~ (CAPACITY-1) 범위로 매핑
    
    hash_val = zlib.crc32(str(value).encode('utf-8'))
    return (hash_val % (RE_MAX_CAPACITY - 1)) + 1

def get_vocab_sizes() -> tuple[int, int]:
    """
    (STD Vocab Size, RE Max Capacity) 반환
    """
    return STD_VOCAB_SIZE, RE_MAX_CAPACITY


'''






# --- B. RE Vocabulary 구축 (동적 관리 대상) ---

# RE 사전을 관리하는 전역 딕셔너리
# 초기 GENERATED_METADATA 로드 및 ID 할당 로직 (STD와 동일한 ID 부여 로직 적용)
'''
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


'''


'''
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
'''