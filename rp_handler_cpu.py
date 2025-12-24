import os
import time
import cv2
import numpy as np
import io
import requests
import runpod
from gfpgan.utils import GFPGANer
import base64


# Функция для загрузки модели GFPGAN, если её нет на диске
def download_model(url, model_path):
    if not os.path.exists(model_path):
        print(f"Downloading {model_path}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                f.write(response.content)
            print(f"Model {model_path} downloaded successfully!")
        else:
            raise Exception(f"Failed to download {model_path} from {url}")


# Задаём URL и путь для модели GFPGAN
gfpgan_model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
gfpgan_model_path = 'GFPGANv1.4.pth'

# Загружаем модель, если она не скачана ранее
download_model(gfpgan_model_url, gfpgan_model_path)

# Инициализируем GFPGANer с указанными параметрами
face_enhancer = GFPGANer(
    model_path=gfpgan_model_path,
    upscale=2,
    arch='clean', channel_multiplier=2)


def handler(job):
    job_input = job["input"]

    # Получаем URL изображения из входных данных
    image_url = job_input["image_url"]
    response = requests.get(image_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image from {image_url}")

    image_data = response.content
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Улучшаем изображение с помощью GFPGAN
    _, _, output = face_enhancer.enhance(image,
                                         has_aligned=False,
                                         only_center_face=False,
                                         paste_back=True)

    # Кодируем выходное изображение в PNG и затем в base64
    success, encoded_image = cv2.imencode('.png', output)
    if not success:
        raise Exception("Image encoding failed")

    output_buffer = io.BytesIO(encoded_image.tobytes())
    base64_image = base64.b64encode(output_buffer.getvalue()).decode()

    # Возвращаем изображение в виде data URL
    return {
        "images_base64": [base64_image],
        "time": round(time.time() - job["created"], 2) if "created" in job else None,
    }


runpod.serverless.start({"handler": handler})
