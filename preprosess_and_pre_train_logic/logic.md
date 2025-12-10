### STD Data -> Re Data without IMG or TItles Prompt 초안


```
        "metadata.clothes.type": "01outer_04cardigan",

        "metadata.clothes.season": "spring&fall",

        "metadata.clothes.fiber_composition": "Polyester",

        "metadata.clothes.elasticity": "contain",

        "metadata.clothes.transparency": "contain",

        "metadata.clothes.isfleece": "fleece_none",

        "metadata.clothes.color": "Black",

        "metadata.clothes.top_length_type": "normal",

        "metadata.clothes.sleeve_length_type": "long sleeves",

        "metadata.clothes.topneck_color_design": "V-neck",

        "metadata.clothes.topsleeve_design": "basic sleeve",

        "metadata.clothes.pants_silhouette": "null",

        "metadata.clothes.skirt_design": "null",

        "metadata.top.front_length": "60.0",

        "metadata.top.chest_size": "80.0",

        "metadata.top.waist_size": "60.0",

        "metadata.top.hem_size": "48.0",

        "metadata.top.shoulder_width": "34.0",

        "metadata.top.sleeve_length": "57.0",

        "metadata.top.sleeve_cuff_size": "14.0",



1. 위의 메타데이터의 사이즈에 집중해서, 옷의 shape 및 상세 특성등을 중심으로 추론한 후,

2. 아래 데이터의 value를 key에 맞게 채워줘

ex) 가디건 사이즈 비율을 따졌을때, 아래로 긴 비율이면, categoty에 long_cadigun 등 추가적인 정보를 너가 넣어줘



"reinforced_feature_value" : {

"category" : [""],

"fiber_composition":[""],

"color": [""],

"category_specification": [""],

"specification.metadata":[""]

},  

```

### result

```json
"reinforced_feature_value" : {
    "category" : [
        "Cardigan",
        "Slim Fit Cardigan",
        "V-neck Cardigan",
        "Basic Cardigan"
    ],
    "fiber_composition": [
        "Polyester",
        "Synthetic Fiber",
        "Stretch Fabric"
    ],
    "color": [
        "Black",
        "Dark Tone",
        "Monotone"
    ],
    "category_specification": [
        "Slim Silhouette",
        "Tight Fit",
        "Standard Length",
        "Long Sleeve",
        "V-Neck Line"
    ],
    "specification.metadata": [
        "high_elasticity",
        "narrow_shoulder_fit",
        "waist_accentuated",
        "sheer_texture"
    ]
}

```