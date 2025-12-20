
import datetime
from fastapi import BackgroundTasks, FastAPI, Depends, HTTPException, APIRouter
from pydantic import BaseModel, Field

from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database import InteractionEvent, ProductInferenceInput, ProductInferenceVectors, UserProfile, UserSession, get_db, Base
from typing import List, Dict, Any, Optional


controller_router = APIRouter()



# ---  API Implementation ---

class ProductCreateRequest(BaseModel):
    product_id: int
    feature_data: Dict[str, Any]  # JSONB 컬럼에 매핑될 딕셔너리



@controller_router.post("/products/ingest")
def ingest_products(payload: List[ProductCreateRequest], db: Session = Depends(get_db)):
    """
    [API 1] N개의 상품 데이터를 받아서 DB에 저장합니다.
    초기 저장 시 is_vectorized는 default로 False가 됩니다.
    """
    try:
        saved_count = 0
        for item in payload:
            # 중복 체크 (이미 있으면 업데이트 or 패스 정책 결정)
            existing = db.query(ProductInferenceInput).filter_by(product_id=item.product_id).first()
            
            if existing:
                existing.feature_data = item.feature_data
                existing.is_vectorized = False # 데이터가 바뀌었으니 다시 벡터화해야 함
            else:
                new_product = ProductInferenceInput(
                    product_id=item.product_id,
                    feature_data=item.feature_data,
                    is_vectorized=False
                )
                db.add(new_product)
            saved_count += 1
        
        db.commit()
        return {"status": "success", "message": f"Saved {saved_count} items."}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    


@controller_router.get("/similarity/pgvector/{item_id}")
def check_similarity_pgvector(item_id: int, db: Session = Depends(get_db)):
    """
    pgvector를 사용하여 DB 내에서 고속 유사도 검색을 수행합니다.
    """
    
    # 1. 타겟 아이템(Query)의 벡터 조회
    # (이미 DB에 임베딩이 저장되어 있다고 가정)
    stmt = select(ProductInferenceVectors).where(ProductInferenceVectors.id == item_id)
    target_item = db.execute(stmt).scalar_one_or_none()

    if not target_item:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found in vector table")

    if target_item.vector_embedding is None:
        raise HTTPException(status_code=400, detail="Target item has no vector embedding yet")

    target_vec = target_item.vector_embedding

    # 2. pgvector 유사도 검색 (Cosine Distance)
    # 문법: 컬럼명.cosine_distance(비교벡터)
    # 자기 자신(item_id)은 제외하고 검색
    
    similarity_expr = ProductInferenceVectors.vector_embedding.cosine_distance(target_vec)
    
    stmt_search = (
        select(ProductInferenceVectors, similarity_expr.label("distance"))
        .where(ProductInferenceVectors.id != item_id) # 자기 자신 제외
        .where(ProductInferenceVectors.vector_embedding.is_not(None)) # 벡터 없는 것 제외
        .order_by(similarity_expr.asc()) # 거리 짧은 순 (=유사도 높은 순)
        .limit(20) # Top 5
    )
    
    results = db.execute(stmt_search).all()

    if not results:
        return {"message": "No similar items found."}

    # 3. 결과 포맷팅
    response_list = []
    for row in results:
        item = row[0]       # ProductInferenceVectors 객체
        distance = row[1]   # cosine_distance 값 (0~2)
        
        # 보기 편하게 Similarity(0~1)로 변환: 1 - distance
        similarity_score = 1 - distance
        
        response_list.append({
            "rank_score": round(similarity_score, 4), # 유사도 점수
            "raw_distance": round(distance, 4),       # 거리 점수 (pgvector 원본)
            "id": item.id,
            "category": item.category,
            # "vector": str(item.vector_embedding)[:50] + "..." # 벡터 값은 필요하면 출력
        })

    return {
        "query_item": {
            "id": target_item.id,
            "category": target_item.category
        },
        "top_5_similar": response_list
    }
    
    
    
'''
{
  "products": [
    { "id":  10411  },
    { "id": 102127  },
    { "id":  17236 }
  ],
  "users": [
    {
      "gender": "F",
      "age_group": "20s",
      "sessions": [
        {
          "session_id": "sess_short_01",
          "season": "WINTER",
          "events": [
            {
              "product_id": 10411,
              "action_type": "VIEW",
              "timestamp": "2024-12-21T10:00:00"
            },
            {
              "product_id": 102127,
              "action_type": "PURCHASE",
              "timestamp": "2024-12-21T10:05:00"
            }
          ]
        }
      ]
    }
  ] 
 
}
'''   
    
    
    
    
class ProductInput(BaseModel):
    id: int 

class EventInput(BaseModel):
    product_id: int
    action_type: str = "PURCHASE"
    timestamp: datetime.datetime

class SessionInput(BaseModel):
    session_id: str
    season: str
    context: Optional[str] = None
    events: List[EventInput]

class UserInput(BaseModel):
    gender: str
    age_group: str
    sessions: List[SessionInput]    
    
class ManualSeedRequest(BaseModel):
    # 검증할 상품 ID 목록 (선택 사항)
    products: List[ProductInput] 
    users: List[UserInput]
    
@controller_router.post("/api/v1/debug/insert-manual-data")
def insert_manual_test_data(req: ManualSeedRequest, db: Session = Depends(get_db)):
    """
    [데이터 수동 주입 API - V2]
    - 상품 벡터를 입력받지 않습니다.
    - 입력받은 product_id가 실제 DB(ProductInferenceVectors)에 존재하는지만 검증합니다.
    - 유저/세션/이벤트 데이터만 INSERT 합니다.
    """
    try:
        # 1. 상품 존재 여부 검증 (Optional)
        # 입력된 상품 ID들이 실제 DB에 벡터가 있는지 확인
        input_product_ids = {p.id for p in req.products}
        
        if input_product_ids:
            existing_products = db.query(ProductInferenceVectors.id).filter(
                ProductInferenceVectors.id.in_(input_product_ids),
                ProductInferenceVectors.vector_embedding.isnot(None)
            ).all()
            
            existing_ids = {row[0] for row in existing_products}
            missing_ids = input_product_ids - existing_ids
            
            if missing_ids:
                # 경고만 남기거나 에러를 발생시킬 수 있음. 여기선 에러 발생.
                raise HTTPException(
                    status_code=400, 
                    detail=f"❌ 다음 상품 ID들은 DB에 벡터가 없습니다: {missing_ids}"
                )

        # 2. 유저 -> 세션 -> 이벤트 저장 (기존 로직 유지)
        created_counts = {"users": 0, "sessions": 0, "events": 0}

        for u_input in req.users:
            # User
            user = UserProfile(
                gender=u_input.gender, 
                age_group=u_input.age_group, 
                is_vectorized=False
            )
            db.add(user)
            db.flush()
            created_counts["users"] += 1

            for s_input in u_input.sessions:
                # Session
                sess_time = s_input.events[0].timestamp if s_input.events else datetime.datetime.now()
                session = UserSession(
                    session_id=s_input.session_id,
                    user_id=user.user_id,
                    season=s_input.season,
                    timestamp=sess_time,
                    context_cart_description=s_input.context
                )
                db.add(session)
                created_counts["sessions"] += 1

                # Events
                for e_input in s_input.events:
                    # 이벤트에 들어있는 상품 ID가 DB에 있는지 체크하고 싶다면 여기서 추가 조회 가능
                    event = InteractionEvent(
                        session_id=s_input.session_id,
                        product_id=e_input.product_id,
                        action_type=e_input.action_type,
                        timestamp=e_input.timestamp
                    )
                    db.add(event)
                    created_counts["events"] += 1
        
        db.commit()

        return {
            "status": "success",
            "message": "User history data inserted successfully.",
            "validated_products": len(input_product_ids),
            "stats": created_counts
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Insert Failed: {str(e)}")