
### 구조도
![description](./image/IMG_4814.jpeg)


### 차원 I/O

64 (Embedding) -> 64 (Pooled) -> 256 (Expanded) -> 512 (Item Tower Out) 
-> 128 (Final DB Vector)

### local 환경: gtx 2070 super 기준, python 3.10.11 ver req



graph TD
    %% 스타일 정의
    classDef input fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef tower fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef deep fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef optim fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef serving fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    subgraph "1. Data Preprocessing & Input"
        Raw[Raw JSON Data]:::input --> Split{Feature Splitting}
        Split --"Basic (Category, Color)"--> Std[Standard Features]:::input
        Split --"Detailed (Material, Neckline)"--> Re[Reinforced Features]:::input
    end

    subgraph "2. Stage 1: Coarse-to-Fine Item Tower"
        Std & Re --> SharedEmb[Shared Embedding Layer<br/>(64 dim)]
        
        SharedEmb --"Self-Attn"--> Std_Enc[Standard Encoder]
        SharedEmb --"Self-Attn"--> Re_Enc[Reinforced Encoder]
        
        Std_Enc & Re_Enc --> CrossAttn[<b>Cross-Attention</b><br/>Refinement]:::tower
        Note1[Query: Standard<br/>Key/Val: Reinforced] -.-> CrossAttn
        
        CrossAttn --> AddNorm[Residual + LayerNorm]:::tower
        AddNorm --> Pooling[Mean Pooling]
    end

    subgraph "3. Deep Residual Pyramid Head"
        Pooling --"(64d)"--> Expand[Expansion Layer<br/>(Linear 64 -> 256)]:::deep
        Expand --> ResBlock[<b>Residual Blocks</b><br/>(Non-linearity Learning)]:::deep
        ResBlock --> Compress[Compression Layer<br/>(Linear 256 -> 512)]:::deep
        Compress --> L2Norm1[L2 Normalization]
    end

    subgraph "4. Stage 2: Metric Learning Optimization"
        L2Norm1 --"(512d)"--> Adapter[Optimization Adapter<br/>(Triplet Loss Training)]:::optim
        
        Sampler[<b>Hierarchical Sampler</b><br/>(Hard Negative Mining)] -.-> Adapter
        Adapter --> FinalVec[<b>Final Item Vector</b><br/>(128 dim)]:::optim
    end

    subgraph "5. Serving & Reranking"
        FinalVec --> VectorDB[(Vector DB)]:::serving
        User[User Context] --> UserTower[User Tower]:::serving
        UserTower --> ANN[ANN Search<br/>(Top-100)]
        VectorDB <--> ANN
        ANN --> Rerank[<b>Self-Attention Reranker</b><br/>(Diversity & Accuracy)]:::serving
        Rerank --> RecList[Final Recommendation]
    end