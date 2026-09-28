FROM python:3.10-slim

# 安裝 OpenCV 與 MediaPipe 必備的 Linux 系統繪圖與多媒體函式庫
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglx-mesa0 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
