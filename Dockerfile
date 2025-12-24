# RunPod рекомендует базовые PyTorch образы; пример тега: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WEIGHTS_DIR=/app/weights \
    TORCH_HOME=/app/.torch

WORKDIR /app

# system deps (opencv может требовать libGL на некоторых сборках)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# copy project
COPY . /app

# install python deps
RUN python -m pip install --upgrade pip \
 && pip install -r req.txt

# (опционально) если вы хотите использовать код из репозитория как пакет
# GFPGAN репозиторий обычно содержит setup.py
RUN if [ -f "setup.py" ]; then pip install -e .; fi

# preload weights at build time (ускоряет первый запрос)
RUN python download_weights.py

CMD ["python", "-u", "rp_handler.py"]
