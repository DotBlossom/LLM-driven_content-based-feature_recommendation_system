from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.http.hooks.http import HttpHook
from datetime import datetime, timedelta
import json
import logging

# --- 설정 ---
API_CONN_ID = "fastapi_server"  # Airflow Connection ID (Admin -> Connections에서 설정)
DAG_ID = "product_embedding_pipeline"
BATCH_SIZE = 100  # 한 번 요청 시 처리할 배치 사이즈 (FastAPI 설정과 맞춤)

try:
    from temp_data import TEST_PRODUCT_DATA
except ImportError:
    # 파일이 없는 경우를 대비한 안전 장치 (실제 데이터는 비어있음)
    TEST_PRODUCT_DATA = []


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# --- 로거 설정 ---
logger = logging.getLogger("airflow.task")

def _preprocess_data(**context):
    """
    [Future Work] 향후 전처리 로직이 들어갈 자리입니다.
    예: S3나 HDFS에서 Raw 파일을 읽어 정제 후, XCom이나 로컬 파일로 넘깁니다.
    현재는 Ingest 단계로 넘길 더미 데이터를 생성한다고 가정합니다.
    """
    logger.info("Starting Preprocessing...")
    
    # 예시: 전처리된 데이터를 생성 (실제로는 파일 로드 등 수행)
    processed_data = TEST_PRODUCT_DATA
    if not processed_data:
        logger.warning("TEST_PRODUCT_DATA is empty. Check temp_data.py.")
    
    # 다음 Task(Ingest)에서 사용할 수 있도록 XCom에 Push
    context['ti'].xcom_push(key='clean_data', value=processed_data)
    logger.info(f"Preprocessed {len(processed_data)} items.")

def _ingest_products(**context):
    """
    [API 1 호출] 전처리된 데이터를 FastAPI의 /products/ingest 로 전송합니다.
    """
    # 1. 이전 단계(Preprocessing)에서 데이터 가져오기
    clean_data = context['ti'].xcom_pull(key='clean_data', task_ids='preprocess_feature_task')
    
    if not clean_data:
        logger.info("No data to ingest.")
        return

    # 2. HTTP Hook을 사용해 API 호출
    http = HttpHook(method='POST', http_conn_id=API_CONN_ID)
    endpoint = "/api/controller/products/ingest"
    
    headers = {"Content-Type": "application/json"}
    response = http.run(endpoint, json=clean_data, headers=headers)
    
    if response.status_code == 200:
        logger.info(f"Ingest Success: {response.text}")
    else:
        raise Exception(f"Ingest Failed: {response.status_code} - {response.text}")

def _trigger_vectorization(**context):
    """
    [API 3 호출] /vectors/process-pending 을 호출하여 벡터화를 수행합니다.
    API가 한 번에 batch_size만큼만 처리하므로, 'processed_count'가 0이 될 때까지 반복 호출합니다.
    """
    http = HttpHook(method='POST', http_conn_id=API_CONN_ID)
    endpoint = "/ai-api/serving/vectors/process-pending"
    
    total_processed = 0
    loop_limit = 100 # 무한 루프 방지용 안전 장치
    
    for i in range(loop_limit):
        logger.info(f"Requesting vectorization batch {i+1}...")
        
        response = http.run(endpoint)
        result = response.json()
        
        # API 응답 파싱 (status, processed_count 확인)
        count = result.get("processed_count", 0)
        
        if count == 0:
            logger.info("No more pending products. Vectorization complete.")
            break
            
        total_processed += count
        logger.info(f"Processed {count} items in this batch.")
    
    logger.info(f"Total vectorized items in this run: {total_processed}")

# --- DAG 정의 ---
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Product Ingest -> Preprocess -> Vectorize Pipeline',
    schedule_interval='@hourly', # 1시간마다 실행 (필요에 따라 조정)
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['mlops', 'recommendation'],
) as dag:

    # 1. 전처리 Task (Future Implementation)
    preprocess_task = PythonOperator(
        task_id='preprocess_feature_task',
        python_callable=_preprocess_data,
        provide_context=True
    )

    # 2. 데이터 적재 Task (Ingest)
    ingest_task = PythonOperator(
        task_id='ingest_to_db_task',
        python_callable=_ingest_products,
        provide_context=True
    )

    # 3. 벡터화 수행 Task (Inference Loop)
    vectorization_task = PythonOperator(
        task_id='process_pending_vectors_task',
        python_callable=_trigger_vectorization,
        provide_context=True
    )

    # --- Task 의존성 설정 ---
    # 전처리 -> DB적재 -> 벡터화 순서로 실행
    preprocess_task >> ingest_task >> vectorization_task