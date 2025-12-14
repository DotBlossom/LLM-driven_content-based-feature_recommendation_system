from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator
from datetime import datetime, timedelta

# --- 설정 ---
API_CONN_ID = "fastapi_server"  # 이전에 설정한 Connection ID 사용
DAG_ID = "product_training_weekly"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

# --- DAG 정의 ---
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Weekly SimCSE Model Training Trigger',
    # [스케줄 설정] Cron 표현식: 분(0) 시(3) 일(*) 월(*) 요일(1=월요일)
    schedule_interval='0 3 * * 1', 
    start_date=datetime(2023, 1, 1),
    catchup=False, # 과거 실행분 무시
    tags=['mlops', 'training', 'simcse'],
) as dag:

    # 1. 학습 요청 Task
    # FastAPI의 /train/start 는 BackgroundTasks를 사용하므로
    # 요청을 보내면 즉시 "Training started" 응답을 받고 Task는 성공(Success) 처리됩니다.
    trigger_train_task = SimpleHttpOperator(
        task_id='trigger_simcse_train',
        http_conn_id=API_CONN_ID,
        endpoint='/train/start',
        method='POST',
        headers={"Content-Type": "application/json"},
        # 필요 시 body에 배치 사이즈 등을 담아 보낼 수 있습니다.
        # data=json.dumps({"batch_size": 64}), 
        response_check=lambda response: response.status_code == 200,
        log_response=True
    )

    trigger_train_task