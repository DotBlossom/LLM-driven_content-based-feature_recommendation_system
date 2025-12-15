FROM python:3.10-slim

# Airflow 버전 및 Python 버전 제약 조건 설정 
ARG AIRFLOW_VERSION=2.10.3
ARG PYTHON_VERSION=3.10
ARG CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# 환경 변수 설정
ENV AIRFLOW_HOME=/usr/local/airflow

# 시스템 업데이트 및 필요한 패키지 설치
RUN apt-get update && \
    apt-get install -y gcc libc-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Airflow 및 Provider 설치
RUN pip install "apache-airflow[http,slack]==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}" 

# Airflow 홈 디렉터리 생성 및 작업 디렉터리 설정
RUN mkdir -p $AIRFLOW_HOME
WORKDIR $AIRFLOW_HOME


EXPOSE 8080

CMD ["airflow", "webserver"]