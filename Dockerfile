# 1단계: 빌드용
FROM python:3.12-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

# [핵심] CPU 전용 PyTorch 먼저 설치
# 이 설정을 먼저 해야 나중에 설치되는 ultralytics가 무거운 GPU용 torch를 다시 받지 않습니다.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 2단계: 실행용
FROM python:3.12-slim
WORKDIR /app

# 런타임 필수 패키지 (libgomp1은 YOLO CPU 연산에 필수입니다)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# 소스 복사
COPY . .

EXPOSE 8000

# 애플리케이션 실행
CMD ["python", "main.py"]