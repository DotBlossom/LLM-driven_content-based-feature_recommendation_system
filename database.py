from datetime import datetime
import os
from dotenv import load_dotenv
from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Index, Text, create_engine, func,select, Column, Integer , String
from typing import List, Dict, Any, Optional
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from pydantic import BaseModel, Field
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship,sessionmaker

'''
/stage1/batch-embed: Stage 1 모델을 사용하여 Raw Data -> DB(vector_pre) 파이프라인 수행.

/stage2/batch-inference: Stage 2 모델을 사용하여 vector_pre -> vector_triplet 변환 수행. (DB 저장 없이 결과만 리턴하여 서빙 로직에서 즉시 사용 가능하도록 설계)


'''

# 1. .env 파일 로드 (환경변수로 등록됨)
load_dotenv()

# 2. 환경변수에서 가져오기
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

# DB 세션 생성 함수 (Dependency)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:

        db.close()





class Season(str, Enum):
    SPRING_AUTUMN = "Spring/Autumn"
    SUMMER = "Summer"
    WINTER = "Winter"

class ActionType(int, Enum):
    CLICK = 1       # 가중치 1
    CART = 3        # 가중치 3
    PURCHASE = 5    # 가중치 5 (핵심 Positive Sample)



class ProductInferenceInput(Base):
    """
    [학습/배치용 입력 데이터]
    Raw Feature를 보관하며, 벡터화 대상인지 관리
    """
    __tablename__ = "product_inference_input"

    product_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # JSONB를 사용하면 DB 레벨에서 JSON 내부 필드 검색/인덱싱 가능 (속도 유리)
    # Python에서는 dict 형태로 사용됨
    feature_data: Mapped[dict[str, Any]] = mapped_column(JSONB)
    
    is_vectorized: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    __table_args__ = (
        # [최적화] 벡터화 안 된 상품만 골라내는 부분 인덱스
        Index('idx_inf_not_vectorized', "product_id", postgresql_where=(is_vectorized == False)),
    )


class ProductInferenceVectors(Base):
    """
    [학습/배치용 벡터 저장소]
    임베딩 모델이 뱉은 Raw Vector와 서빙용으로 가공된 Vector 모두 저장
    """
    __tablename__ = "product_inference_vectors"

    # 일반적으로 product_id와 1:1 매핑되므로 id를 product_id로 간주하거나 FK 설정 권장
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    vector_embedding: Mapped[list[float] | None] = mapped_column(Vector(128), nullable=True)
    vector_serving: Mapped[list[float] | None] = mapped_column(Vector(128), nullable=True)
    
    category: Mapped[str | None] = mapped_column(String, nullable=True, index=True)

## real data / valid data

class ProductServiceInput(Base):
    """
    [서빙용 메타 데이터]
    실제 서비스에 노출될 검증된(Valid) 데이터
    """
    __tablename__ = "product_service_input"

    product_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    feature_data: Mapped[dict[str, Any]] = mapped_column(JSONB)
    
    is_vectorized: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    __table_args__ = (
        # [최적화] 서빙 파이프라인에서 벡터화 대기중인 상품 추출용
        Index('idx_svc_not_vectorized', "product_id", postgresql_where=(is_vectorized == False)),
    )


class ProductServiceVectors(Base):
    """
    [서빙용 벡터 저장소]
    검색 엔진(Vector DB) 역할. 실제 유저 쿼리와 내적(Dot Product) 연산 수행
    """
    __tablename__ = "product_service_vectors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    vector_embedding: Mapped[list[float] | None] = mapped_column(Vector(128), nullable=True)
    
    category: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    
    
    

class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    gender: Mapped[str] = mapped_column(String(10))        # 'M', 'F' 등
    age_group: Mapped[str] = mapped_column(String(20))     # '20s', '30s'
    style_preference: Mapped[str | None] = mapped_column(String(50), nullable=True)
    
    # Vector 저장용 (128차원)
    user_service_vector: Mapped[list[float] | None] = mapped_column(Vector(128), nullable=True)
    
    # 벡터화 여부 플래그
    is_vectorized: Mapped[bool] = mapped_column(Boolean, default=False)

    # 관계 설정: User -> Sessions (1:N)
    sessions: Mapped[List["UserSession"]] = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")

    # 인덱스 설정
    __table_args__ = (
        # [전략] 벡터화가 아직 안 된(False) 유저만 빠르게 뽑아내기 위한 부분 인덱스
        Index('idx_not_vectorized_users', "user_id", postgresql_where=(is_vectorized == False)),
    )


# 3. UserSession (세션 정보)
class UserSession(Base):
    __tablename__ = "user_sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True) # UUID 등을 사용한다면 String
    
    # ForeignKey: user_profiles 테이블의 user_id 참조
    user_id: Mapped[int] = mapped_column(ForeignKey("user_profiles.user_id"), nullable=False)
    
    season: Mapped[str] = mapped_column(String(20)) # Season Enum 값 저장
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # 요청하신 긴 텍스트 (Nullable)
    context_cart_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 관계 설정: Session -> User (N:1)
    user: Mapped["UserProfile"] = relationship("UserProfile", back_populates="sessions")
    
    # 관계 설정: Session -> InteractionEvents (1:N)
    # 기존 Pydantic의 List[InteractionEvent]를 대체함
    events: Mapped[List["InteractionEvent"]] = relationship("InteractionEvent", back_populates="session", cascade="all, delete-orphan")

    # 파이썬 로직용 프로퍼티 (DB 컬럼 아님)
    # 주의: session.events가 로딩된 상태에서만 동작함
    @property
    def is_purchase_session(self) -> bool:
        return any(e.action_type == ActionType.PURCHASE.value for e in self.events)

    # 인덱스 설정
    __table_args__ = (
        # 특정 유저의 세션을 시간순으로 조회
        Index('idx_user_session_timestamp', "user_id", "timestamp"),
    )


# 4. InteractionEvent (개별 행동 로그 - 별도 테이블로 분리 필수)
class InteractionEvent(Base):
    __tablename__ = "interaction_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # 어떤 세션에 속한 이벤트인지 연결
    session_id: Mapped[str] = mapped_column(ForeignKey("user_sessions.session_id"), nullable=False)
    
    product_id: Mapped[int] = mapped_column(Integer, nullable=False)
    action_type: Mapped[str] = mapped_column(String(20)) # ActionType Enum 값 저장
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # 관계 설정: Event -> Session (N:1)
    session: Mapped["UserSession"] = relationship("UserSession", back_populates="events")

    # 필요하다면 여기에 상품 벡터 캐싱 컬럼 추가 가능
    # item_vector_cache = mapped_column(Vector(128), nullable=True)