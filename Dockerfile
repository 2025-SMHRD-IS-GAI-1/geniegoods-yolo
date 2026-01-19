FROM python:3.12-slim

WORKDIR /app

# 시스템 패키지 설치 (OpenCV, YOLO에 필요)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    mesa-utils \
    apt-transport-https \
    ca-certificates \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# YOLO 모델 다운로드 (첫 실행 시 자동 다운로드되지만 미리 다운로드 가능)
# RUN python -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt')"

# 애플리케이션 코드 복사
COPY main.py .

# 포트 노출
EXPOSE 8000

# 애플리케이션 실행
CMD ["python", "main.py"]