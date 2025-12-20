

1. HubData 
    -  전체적인 실루엣, shape 디테일, 그러나 디자인 디테일 X
    -  거시적인 ITEM 판단(전체 모양새가 중요한 옷들에 효과적인 )
    - fabric등 간접적인 shaping , feeling 단서 존재

2. H&M ProductData + Trans_log
    - Design Detail Props + Shape Descriptions
    - id_color (1,2,3) 
    - t-shirt와 같은곳에서, 문양? 그래픽적 특징들을 설명해주는


### 2 -> LLM Conv(vocab config) -> 1
 Shape Descriptions(2) -> shape detail Domain(1)
 Design Factor -> 0(2) , 1(1) -> specification Tag 느낌?






# user data -> valid data 느낌으로, 처음부터 만들떄 한정된 pool control ㄱㄱ (500개 )
```json
[
  {
    "product_id": <int>,
    "reinforced_feature_value": {
      "category": "<Refined Value: Contextual_Synthesis_Result>",
      "season": "<Micro-Season Value>",

      "fit.vibe": "<Text: Generate best fit referencing [Oversized, Relaxed, Slouchy, Standard, Structured, Slim-fit, Bodycon...]>",
      "silhouette.shape": "<Text: Generate best fit referencing [Boxy, H-line, A-line, Trapeze, Inverted-Triangle, Cocoon, Blouson, Hourglass...]>",
      "length.feeling": "<Tier from Step B>",
      "fabric.feature": "<Texture_Weight Combination>",
      
      "structural.geometry": {
        "width_flow": "Expanding" | "Straight" | "Tapered",
        "waist_contour": "Hourglass" | "Tubular" | "Barrel",
        "vertical_balance": "<Tier from Step B>",
        "shoulder_geometry": "Drop" | "Standard" | "Narrow",
        "sleeve_profile": "Wide/Bell" | "Standard" | "Tapered" | "Sleeveless"
      }
    }
  }
]
```