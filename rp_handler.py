import os
import time
import io
import base64
import inspect

import cv2
import numpy as np
import requests
import runpod
import torch

from realesrgan.utils import RealESRGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact


# Пути к предзагруженным весам
WEIGHTS_DIR = "weights"
REALESRGAN_MODEL_PATH = os.path.join(WEIGHTS_DIR,
                                     "realesrgan",
                                     "realesr-general-x4v3.pth")

# Проверяем устройство
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- Инициализация Real-ESRGAN x2plus ---
# Конфиг x2plus RRDBNet берётся из официального inference_realesrgan.py. :contentReference[oaicite:1]{index=1}
model = SRVGGNetCompact(num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_conv=32,
                        upscale=4,
                        act_type='prelu')
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(
    scale=4,
    model_path=REALESRGAN_MODEL_PATH,
    model=model,
    tile=0,
    tile_pad=10,
    pre_pad=0,
    half=half)


print("✅ RealESRGANer loaded successfully!")


def _download_image(image_url: str, timeout: int = 120) -> np.ndarray:
    r = requests.get(image_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("cv2.imdecode вернул None (битая картинка или неподдерживаемый формат).")
    return img


def handler(job):
    start_time = time.time()
    job_input = job.get("input", {})

    image_url = job_input["image_url"]

    # Опциональные параметры (полезно для больших изображений, чтобы не словить OOM)
    # tile=0 — без тайлинга; 256/512/1024 — в зависимости от VRAM
    # tile = int(job_input.get("tile", 0))
    # tile_pad = int(job_input.get("tile_pad", 10))
    # pre_pad = int(job_input.get("pre_pad", 0))

    # Жёстко держим outscale=2 (как ты и просил)
    outscale = 4

    # Применяем параметры тайлинга на лету (без пересоздания модели)
    # upsampler.tile_size = tile
    # upsampler.tile_pad = tile_pad
    # upsampler.pre_pad = pre_pad

    img = _download_image(image_url)

    # RealESRGANer.enhance(img, outscale=...) возвращает (output, _)
    # Сигнатура enhance: enhance(self, img, outscale=None, alpha_upsampler='realesrgan') :contentReference[oaicite:2]{index=2}
    output, _ = upsampler.enhance(img, outscale=outscale)

    success, encoded = cv2.imencode(".png", output)
    if not success:
        raise RuntimeError("cv2.imencode('.png', ...) не смог закодировать изображение")

    base64_image = base64.b64encode(encoded.tobytes()).decode("utf-8")

    processing_time = round(time.time() - start_time, 2)
    print(f"⏱ Processing time: {processing_time}s")

    return {
        "images_base64": [base64_image],
        "time": processing_time,
        "meta": {
            "outscale": outscale,
            "device": device,
            "half": half,
        },
    }


runpod.serverless.start({"handler": handler})
