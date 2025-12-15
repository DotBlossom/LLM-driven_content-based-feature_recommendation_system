import uvicorn

# 터미널 명령어 줄이기 / python run_dev.py

if __name__ == "__main__":
    uvicorn.run(
        "main:app",            
        host="0.0.0.0",        
        port=8000,             
        reload=True,           
        reload_excludes=["airflow", "airflow_data",] 
    )