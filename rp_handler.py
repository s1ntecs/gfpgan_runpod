import os
import time
import cv2
import numpy as np
import io
import requests
import runpod
from gfpgan.utils import GFPGANer
import base64
import torch

# Пути к предзагруженным весам
WEIGHTS_DIR = "weights"
GFPGAN_MODEL_PATH = os.path.join(WEIGHTS_DIR, "gfpgan", "GFPGANv1.4.pth")

# Устанавливаем переменные окружения для facexlib
os.environ['FACEXLIB_WEIGHTS'] = os.path.join(WEIGHTS_DIR, "facexlib")

# Проверяем доступность GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Инициализируем GFPGANer
face_enhancer = GFPGANer(
    model_path=GFPGAN_MODEL_PATH,
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    device=device,
    # Указываем путь к весам фонового апскейлера (опционально)
    bg_upsampler=None  # или можно подключить RealESRGAN
)

print("✅ Model loaded successfully!")


def handler(job):
    start_time = time.time()
    job_input = job["input"]

    image_url = job_input["image_url"]
    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image from {image_url}")

    image_data = response.content
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    _, _, output = face_enhancer.enhance(
        image,
        has_aligned=False,
        only_center_face=False,
        paste_back=True
    )

    success, encoded_image = cv2.imencode('.png', output)
    if not success:
        raise Exception("Image encoding failed")

    output_buffer = io.BytesIO(encoded_image.tobytes())
    base64_image = base64.b64encode(output_buffer.getvalue()).decode()

    processing_time = round(time.time() - start_time, 2)
    print(f"⏱ Processing time: {processing_time}s")

    return {
        "images_base64": [base64_image],
        "time": processing_time,
    }


runpod.serverless.start({"handler": handler})