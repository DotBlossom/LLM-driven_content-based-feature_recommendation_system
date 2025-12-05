
"""
class RichAttributeDataset(Dataset):
    
    #Stage 2 학습을 위한 데이터셋. 입력은 Stage 1의 128 pre-vector
    
def __init__(self, product_list):
        # 1. 임시 저장소 (가벼운 Python List 사용)
        temp_vectors = []
        temp_ids = []  # <--- ID 저장을 위한 리스트
        self.fine_labels = []
        self.coarse_labels = []
        self.label_to_id = {}
        
        # 2. Loop: 데이터 정제 및 라벨링 (여기서는 Tensor 변환 금지!)
        for item in product_list:
            # --- Vector 처리 ---
            vec = item.get('vector')
            p_id = item.get('id') # ID 추출
            
            if vec is None:
                # Mock vector (리스트 형태)
                vec = np.random.randn(512).astype(np.float32).tolist()
            
            # 여기서 torch.tensor(vec)를 하지 않고, 그냥 리스트(혹은 numpy) 상태로 둡니다.
            temp_vectors.append(vec)
            temp_ids.append(p_id)
            
            # --- Label 처리 ---, DTO 주의
            full_cat = item['clothes']['category'][0]
            coarse_cat = full_cat[:2] # 예: "top" from "top/tee"
            
            if full_cat not in self.label_to_id:
                self.label_to_id[full_cat] = len(self.label_to_id)
            
            self.fine_labels.append(self.label_to_id[full_cat])
            self.coarse_labels.append(coarse_cat)

        # 3. [핵심 최적화] Bulk Conversion (한방에 텐서화)
        # 리스트의 리스트 -> 하나의 거대한 FloatTensor 
        # 이 방식이 메모리를 훨씬 적게 쓰고, 연산 속도도 빠릅니다.
        self.data = torch.tensor(temp_vectors, dtype=torch.float32)
        self.ids = torch.tensor(temp_ids, dtype=torch.long) 
        
        # 라벨도 마찬가지로 LongTensor로 한 번에 변환
        self.fine_labels = torch.tensor(self.fine_labels, dtype=torch.long)

        print(f"[Dataset] Created tensors. Shape: {self.data.shape}")
        
        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            # 인덱싱 속도도 훨씬 빠름 (텐서 슬라이싱)
            return self.data[idx], self.fine_labels[idx], self.ids[idx]

class HierarchicalBatchSampler(Sampler):
    # ... (Sampler 클래스 정의 유지) ...
    def __init__(self, dataset, batch_size, samples_per_class=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.structure = defaultdict(lambda: defaultdict(list))
        self.all_indices = []
        
        for idx, (fine_id, coarse_name) in enumerate(zip(dataset.fine_labels, dataset.coarse_labels)):
            self.structure[coarse_name][fine_id].append(idx)
            self.all_indices.append(idx)
            
        self.coarse_keys = list(self.structure.keys())

    def __iter__(self):
        num_batches = len(self.dataset) // self.batch_size
        
        for _ in range(num_batches):
            batch_indices = []
            target_coarse = random.choice(self.coarse_keys)
            fine_dict = self.structure[target_coarse]
            available_fine_labels = list(fine_dict.keys())
            num_classes_needed = self.batch_size // self.samples_per_class
            
            # A. 충분한 소분류가 있는 경우 (Hard Negative Mode)
            if len(available_fine_labels) >= num_classes_needed:
                selected_fines = random.sample(available_fine_labels, k=num_classes_needed)
                
                for f_label in selected_fines:
                    indices = fine_dict[f_label]
                    selected_indices = random.choices(indices, k=self.samples_per_class)
                    batch_indices.extend(selected_indices)
            
            # B. 소분류가 부족한 경우 (Fallback: Noise Mixing Mode)
            else:
                for f_label in available_fine_labels:
                    indices = fine_dict[f_label]
                    selected_indices = random.choices(indices, k=self.samples_per_class)
                    batch_indices.extend(selected_indices)
                
                remaining_slots = self.batch_size - len(batch_indices)
                if remaining_slots > 0:
                    noise_indices = random.choices(self.all_indices, k=remaining_slots)
                    batch_indices.extend(noise_indices)
            
            yield batch_indices

    def __len__(self):
        return len(self.dataset) // self.batch_size

def train_model(product_list, epochs=5, batch_size=32, save_path="models/final_optimized_adapter.pth"):
    # ... (train_model 함수 로직 유지) ...
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    dataset = RichAttributeDataset(product_list)
    
    if len(dataset) < batch_size:
        dataloader = DataLoader(dataset, batch_size=len(dataset))
    else:
        sampler = HierarchicalBatchSampler(dataset, batch_size)
        dataloader = DataLoader(dataset, batch_sampler=sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptimizedItemTower().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    ###### Triplet Logic
    
    #distance = distances.CosineSimilarity()
    #loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance)
    #mining_func = miners.TripletMarginMiner(margin=0.2, distance=distance, type_of_triplets="semihard")

    #####
    
    
    # infoNCE / Temperature (0.07 ~ 0.1 추천): 낮을수록 Hard Negative에 집중, 높을수록 부드럽게 학습
    loss_func = losses.NTXentLoss(temperature=0.093) 
    
    model.train()
    history = []
    
    for epoch in range(epochs):
        total_loss = 0
        triplets_count = 0
        
        for batch_vecs, batch_labels, _ in dataloader:
            batch_vecs = batch_vecs.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            embeddings = model(batch_vecs)
            #indices_tuple = mining_func(embeddings, batch_labels)
            loss = loss_func(embeddings, batch_labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        log = f"Epoch {epoch+1}/{epochs}: Avg Loss={avg_loss:.4f}"
        print(log)
        history.append(log)

    torch.save(model.state_dict(), save_path)
    return history
    
    
    
    
    # vector_triplet에 저장, real cos
def load_and_infer(input_vector, model_path="models/final_optimized_adapter.pth"):
    print("="*60)
    print(f"[Inference Step 1] Initializing Inference Pipeline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference Step 2] Target Device: {device}")
    
    # 모델 초기화
    model = OptimizedItemTower().to(device)
    
    # 모델 가중치 로드
    if not os.path.exists(model_path):
        print(f"[Error] Model file not found at: {model_path}")
        raise FileNotFoundError("Model file not found. Train first.")
    
    print(f"[Inference Step 3] Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # 평가 모드 전환 (Dropout, Batchnorm 고정)
    model.eval()
    print(f"[Inference Step 4] Model set to EVAL mode.")
    
    with torch.no_grad():
        # 입력 텐서 변환
        tensor_in = torch.tensor([input_vector], dtype=torch.float32).to(device)
        print(f"[Inference Step 5] Input Tensor created on {device}. Shape: {tensor_in.shape}")
        
        # 모델 Forward 실행 (여기서 위에서 정의한 [Model Internal] 로그가 찍힘)
        print("-" * 30 + " Model Forward Start " + "-" * 30)
        output = model(tensor_in)
        print("-" * 30 + " Model Forward End " + "-" * 30)
        
    # 결과 반환
    result_vector = output.cpu().numpy().tolist()[0]
    print(f"[Inference Step 6] Final Output Vector generated. Length: {len(result_vector)}")
    print("="*60)
    
    return result_vector
    """
