현재 직면하신 **Loss 8.8 (Random Guess  보다 나쁨)** 상황과 **Item Tower 붕괴** 문제를 해결하기 위한 **User Tower 리팩토링 및 학습 전략**을 제안합니다.

현재 코드의 `RobustFusion`(단순 Concat + MLP) 방식은 GNN(전역적 선호)과 Sequence(일시적/최근 선호) 간의 충돌을 제대로 중재하지 못할 가능성이 큽니다. 이를 해결하기 위해 **"Gated Information Fusion"** 구조로의 변경을 핵심으로 한 전략을 제시합니다.

---

### 🔍 전략 1: Fusion Layer의 고도화 (Gated Fusion Mechanism)

현재 `HybridUserTower`는 GNN 벡터, Sequence 벡터, Meta 벡터를 단순히 `concat` 하여 MLP에 태우고 있습니다. 이는 각 피처의 신뢰도나 중요도를 모델이 동적으로 조절하기 어렵게 만듭니다.

#### 🛠️ 제안: Gated Fusion Network (혹은 MoE - Mixture of Experts Lite)

유저가 "최근 본 상품(Sequence)"에 영향을 받아 구매하는 경우와, "원래 취향(GNN)"대로 구매하는 경우를 모델이 스스로 판단하여 가중치를 두게 합니다.

1. **아이디어:** GNN 벡터()와 Sequence 벡터()에 대해 **Gate(0~1)** 값을 생성합니다.
2. **수식:**



3. **효과:** 노이즈가 많은 Sequence(갑자기 엉뚱한 걸 클릭함)는 무시하고 GNN(장기 선호)을 따르거나, 반대로 GNN 정보가 부족한 신규 유저(Cold Start)는 Sequence에 의존하도록 유도합니다.

> **📚 근거 (Reference)**
> * **Google YouTube RecSys (2019):** "Deep Neural Networks for YouTube Recommendations" 및 후속 논문들에서, 여러 소스의 피처를 합칠 때 단순 Concat 대신 **Gating Network**나 **Tower 구조의 분리**를 통해 피처 간의 간섭(Gradient Conflict)을 줄이는 방식을 사용합니다.
> * **MMoE (Multi-gate Mixture-of-Experts, KDD 2018):** 서로 다른 성격의 정보(Task)를 결합할 때 Gating Network가 필수적임을 증명했습니다.
> 
> 

#### 🏗️ Conceptual Structure

```python
class GatedFusion(nn.Module):
    def forward(self, v_gnn, v_seq, v_meta):
        # 1. Gating Signal 생성 (전체 맥락을 보고 결정)
        context = torch.cat([v_gnn, v_seq, v_meta], dim=1)
        gate = torch.sigmoid(self.gate_linear(context)) # 0.0 ~ 1.0 사이 값

        # 2. 가중치 적용 (Adaptive Weighting)
        # v_seq가 중요하면 gate가 1에 가까워짐
        weighted_feat = gate * v_seq + (1 - gate) * v_gnn
        
        # 3. Meta 정보는 Bias처럼 더하거나 Concat 후 Projection
        final_vec = weighted_feat + self.meta_proj(v_meta)
        
        return self.final_mlp(final_vec)

```

---

### 🔍 전략 2: Temperature Annealing & LogQ Correction 수정

현재 Loss 8.8의 주범 중 하나는 **Temperature (0.07)**와 **LogQ Correction**의 부조화일 가능성이 큽니다.

#### 🛠️ 제안: Dynamic Temperature & Conservative LogQ

1. **Temperature Start High:** 학습 초기에는 User Tower가 Item Space의 어디를 가리킬지 모릅니다. 0.07은 너무 뾰족(Sharp)해서, 조금만 틀려도 Loss가 폭발합니다.
* **전략:** 로 시작하여 학습이 진행됨에 따라 로 낮추는 **Annealing** 스케줄링을 적용하세요.


2. **LogQ Lambda 축소:** `LAMBDA_LOGQ`가 높으면 인기 아이템을 정답으로 맞췄을 때 오히려 점수를 깎아버립니다(Penalize). 아직 모델이 수렴하지 않은 상태에서는 이 페널티가 "정답을 맞추지 말라"는 신호로 오작동하여 Loss를 높입니다.
* **전략:** 초기 1~2 Epoch 동안은 `LAMBDA_LOGQ = 0`으로 두어 모델이 일단 정답을 맞추게 하고, 이후에 0.01~0.1로 서서히 올리세요.



> **📚 근거 (Reference)**
> * **SimCLR (ICML 2020), MoCo:** Contrastive Learning에서 Temperature 는 Gradient의 Scale을 결정하는 핵심 하이퍼파라미터입니다. 낮은 는 Hard Negative에 집중하게 하지만, 초기 학습 안정성을 해칩니다.
> * **Google "Sampling-Bias-Corrected Neural Modeling" (RecSys 2019):** LogQ Correction은 수렴된 모델의 "인기도 편향"을 제거하기 위한 것이지, 학습 초기부터 적용하면 학습 난이도를 불필요하게 높입니다.
> 
> 

---

### 🔍 전략 3: Sequence Encoder의 정체성 확립 (Transformer vs Mean)

현재 `TransformerEncoder`를 사용하고 계시지만, User Tower Dataset의 구성을 보면 `cut_idx`를 통해 **마지막 아이템 하나**를 맞추는 Task입니다.

#### 🛠️ 제안: Attention Pooling 적용

Transformer의 출력 중 단순히 `v_seq = seq_out[..., last_indices]`를 쓰는 것은 "가장 마지막 아이템"의 정보에만 과도하게 집중할 위험이 있습니다. 유저의 의도는 시퀀스 전체에 녹아있을 수 있습니다.

1. **Attention Pooling:** Transformer의 모든 출력()에 대해 가중합을 구하여 하나의 벡터로 만듭니다.


2. **효과:** 단순히 마지막에 본 것뿐만 아니라, 시퀀스 내에서 반복적으로 등장한 패턴이나 중요한 아이템에 더 가중치를 둡니다.

> **📚 근거 (Reference)**
> * **BERT4Rec (CIKM 2019), SASRec (ICDM 2018):** 시퀀스 모델링에서 단순 Last Token 사용보다, 전체 시퀀스에 대한 Attention Pooling이나 [CLS] 토큰 방식이 노이즈에 더 강건함(Robust)을 보입니다.
> 
> 

---

### 🔍 전략 4: Curriculum Learning (Freeze -> Fine-tune)

이것은 코드 구조보다는 **학습 루틴(Train Loop)**의 리팩토링입니다. 질문자님의 상황에서 가장 중요합니다.

#### 🛠️ 제안: 2-Stage Training Pipeline

현재의 "Unfreeze All" 방식은 절대 금물입니다.

1. **Stage 1: User Alignment (Warm-up)**
* **Freeze:** Item Tower (전체), GNN Embedding (전체).
* **Train:** User Tower의 `SeqEncoder`, `FusionLayer`만 학습.
* **목표:** Item Vector들이 형성한 공간(Manifold)에 User Vector를 매핑시키는 것.
* **종료 조건:** Loss가 6.0 이하로 떨어지거나 R@100이 0.05 이상 나오기 시작할 때.


2. **Stage 2: Fine-tuning (Joint Training)**
* **Unfreeze:** Item Tower의 상위 Layer (Projection Head) -> 점차 하위 Layer.
* **LR:** Item Tower 쪽 LR은 User Tower의 1/100 수준()으로 설정.
* **목표:** User와 Item이 서로의 거리를 좁히며 미세 조정.



> **📚 근거 (Reference)**
> * **Transfer Learning Standard:** Pre-trained 모델(여기선 Item Tower)을 사용할 때, Head(여기선 User Tower)를 먼저 학습시키지 않고 Backbone을 풀면 "Catastrophic Forgetting"이 발생한다는 것은 딥러닝의 기초 원칙입니다.
> 
> 

---

### 🚀 요약 및 우선순위

Loss 8.8을 탈출하기 위해 당장 코드에 적용해야 할 리팩토링 우선순위입니다.

1. **[Training] Freeze 적용:** 학습 루프에서 `model.item_content_emb`와 `model.gnn_user_emb`의 `requires_grad = False`로 강제 설정하세요. (가장 시급)
2. **[Structure] Gated Fusion 도입:** `RobustFusion`을 `GatedFusion`으로 교체하여 GNN과 Sequence 정보의 충돌을 방지하세요.
3. **[Loss] Temperature & LogQ 완화:** Temperature , Lambda LogQ 으로 시작하세요.

이 전략을 적용하면, Item Tower의 붕괴 없이 User Tower가 안정적으로 학습 궤도(Loss 6점대 진입)에 오를 것입니다.