from enum import Enum
from datetime import datetime
from typing import List, Optional
from pgvector import Vector
from pydantic import BaseModel
from sqlalchemy import Boolean, Column, Index
from sqlalchemy.orm import relationship, mapped_column

class Season(str, Enum):
    SPRING_AUTUMN = "Spring/Autumn"
    SUMMER = "Summer"
    WINTER = "Winter"

class ActionType(int, Enum):
    CLICK = 1       # 가중치 1
    CART = 3        # 가중치 3
    PURCHASE = 5    # 가중치 5 (핵심 Positive Sample)

class InteractionEvent(BaseModel):
    product_id: int
    action_type: ActionType
    timestamp: datetime
    # Item Tower에서 추출한 벡터를 캐싱해서 쓸 수도 있음 (선택사항)

class UserSession(BaseModel):
    session_id: str
    user_id: int
    season: Season          # 계절 정보 (User Tower의 Context Feature)
    events: List[InteractionEvent] # 
    timestamp: datetime # 세션 필요에따라 가져오려면..
    
    @property
    def is_purchase_session(self) -> bool:
        return any(e.action_type == ActionType.PURCHASE for e in self.events)
    
    __table_args__ = (
        Index('user_event_timestamp', "user_id", "timestamp"),
    )

class UserProfile(BaseModel):
    user_id: int
    gender: str
    age_group: str          # 20s, 30s...
    style_preference: str   # (Optional) 데이터 생성용 메타 정보 (예: Minimal, Street)
    
    
    # vector 저장용 컬럼
    user_service_vector =  mapped_column(Vector(128), nullable=True)
    is_vectorized = Column(Boolean, default=False, index=True)
