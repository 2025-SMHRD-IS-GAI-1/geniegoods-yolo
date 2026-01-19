# 1단계: 빌드 환경
FROM python:3.12-slim AS builder
WORKDIR /app

# 빌드 필수 도구
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

# [핵심] CPU 전용 토치 설치 (용량 다이어트)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 패키지 설치
COPY requirements.txt .
# typing_extensions를 포함하여 모든 의존성을 표준 경로에 설치
RUN pip install --no-cache-dir typing_extensions && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------
# 2단계: 실행 환경
FROM python:3.12-slim
WORKDIR /app

# YOLO/OpenCV 런타임 필수 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# [해결책] 빌드 단계의 패키지를 실행 단계의 '동일한 경로'로 통째로 복사
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 소스 코드 복사 (빌드 캐시 최적화를 위해 패키지 설치 뒤에 배치)
COPY . .

EXPOSE 8000

CMD ["python", "main.py"]