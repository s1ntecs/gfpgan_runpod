FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .
COPY download_weights.py .

# Устанавливаем PyTorch с CUDA
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Устанавливаем остальные зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Скачиваем все веса моделей
RUN python3 download_weights.py

# Копируем основной код
COPY handler.py .

CMD ["python3", "-u", "handler.py"]