from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import torch
import uvicorn

app = FastAPI()

# CORS configuration
origins = [
    "*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/ai-api")


@app.get("/")
def home():
    cuda_status = torch.cuda.is_available()
    return {
    "message": "FastAPI가 정상 작동 중입니다!",
    "cuda_available": cuda_status  # boolean 값 그대로 전달
    }


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query_param": q}



if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5050, reload=True)