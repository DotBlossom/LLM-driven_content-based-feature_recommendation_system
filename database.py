import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

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