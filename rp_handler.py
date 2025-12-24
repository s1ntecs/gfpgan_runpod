import base64
import os
import threading
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import requests
import runpod
import torch

from gfpgan import GFPGANer

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "/app/weights")
GFPGAN_MODEL_PATH = os.getenv("GFPGAN_MODEL_PATH", os.path.join(WEIGHTS_DIR, "GFPGANv1.4.pth"))
REALESRGAN_MODEL_PATH = os.getenv("REALESRGAN_MODEL_PATH", os.path.join(WEIGHTS_DIR, "RealESRGAN_x2plus.pth"))

DEFAULTS = {
    "upscale": 2,                 # GFPGAN обычно используют x2 (как в reference inference)
    "arch": "clean",              # clean | original
    "channel_multiplier": 2,      # обычно 2
    "use_bg_upsampler": True,     # Real-ESRGAN для фона
    "bg_tile": 400,               # тайлинг для VRAM
    "only_center_face": False,
    "aligned": False,
    "paste_back": True,
    "output_format": "png",       # png | jpg
    "jpeg_quality": 95,
    "return_faces": False,        # вернуть массив восстановленных лиц
}

_requests_session = requests.Session()

# Кэшируем рестореры, чтобы не грузить модель на каждый запрос
_restorer_lock = threading.Lock()
_restorer_cache: Dict[Tuple[Any, ...], GFPGANer] = {}

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def _strip_data_uri(data: str) -> str:
    # data:image/png;base64,....
    if data.startswith("data:") and "base64," in data:
        return data.split("base64,", 1)[1]
    return data

def _download_url(url: str, timeout: int = 30) -> bytes:
    r = _requests_session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content

def _decode_image_bytes(img_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Не удалось декодировать изображение (cv2.imdecode вернул None).")
    return img

def _get_image_from_input(job_input: Dict[str, Any]) -> np.ndarray:
    """
    Поддерживаем варианты:
      - {"image": "<base64 или data-uri>"}
      - {"image_url": "https://..."}
      - {"image": "https://..."}  (на всякий)
      - {"image": {"base64": "..."}}
    """
    if "image_url" in job_input and job_input["image_url"]:
        img_bytes = _download_url(str(job_input["image_url"]))
        return _decode_image_bytes(img_bytes)

    image_val = job_input.get("image")
    if image_val is None:
        raise ValueError("Нужен input.image (base64/data-uri) или input.image_url.")

    if isinstance(image_val, dict) and image_val.get("base64"):
        b64 = _strip_data_uri(str(image_val["base64"]))
        img_bytes = base64.b64decode(b64)
        return _decode_image_bytes(img_bytes)

    if isinstance(image_val, str):
        s = image_val.strip()
        # Если строка — URL
        if _is_url(s):
            img_bytes = _download_url(s)
            return _decode_image_bytes(img_bytes)

        # Иначе считаем base64 / data-uri
        b64 = _strip_data_uri(s)
        try:
            img_bytes = base64.b64decode(b64, validate=False)
        except Exception as e:
            raise ValueError(f"input.image не похож ни на URL, ни на base64: {e}") from e
        return _decode_image_bytes(img_bytes)

    raise ValueError("input.image должен быть строкой (base64/url) или объектом {base64: ...}.")

def _encode_image(img_bgr: np.ndarray, fmt: str, jpeg_quality: int = 95) -> str:
    fmt = fmt.lower()
    if fmt not in ("png", "jpg", "jpeg"):
        raise ValueError("output_format должен быть png или jpg.")

    if fmt == "png":
        ok, buf = cv2.imencode(".png", img_bgr)
        if not ok:
            raise RuntimeError("cv2.imencode(.png) не смог закодировать изображение.")
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return "data:image/png;base64," + b64

    # jpg/jpeg
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    ok, buf = cv2.imencode(".jpg", img_bgr, params)
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) не смог закодировать изображение.")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return "data:image/jpeg;base64," + b64


def _ensure_weights_exist():
    # Идемпотентно — если веса уже есть, ничего не делает.
    from download_weights import ensure_weights
    ensure_weights(weights_dir=WEIGHTS_DIR)


def _build_bg_upsampler(bg_tile: int):
    """
    Реализация по референсу inference_gfpgan.py:
      - RRDBNet(scale=2)
      - RealESRGANer(scale=2, model_path=RealESRGAN_x2plus.pth, tile=bg_tile, half=True)
    """
    if not torch.cuda.is_available():
        return None

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2
    )

    return RealESRGANer(
        scale=2,
        model_path=REALESRGAN_MODEL_PATH,
        model=model,
        tile=int(bg_tile),
        tile_pad=10,
        pre_pad=0,
        half=True
    )


def _get_restorer(
    upscale: int,
    arch: str,
    channel_multiplier: int,
    use_bg_upsampler: bool,
    bg_tile: int,
) -> GFPGANer:
    # Ключ кэша
    key = (
        int(upscale),
        str(arch),
        int(channel_multiplier),
        bool(use_bg_upsampler),
        int(bg_tile),
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    with _restorer_lock:
        if key in _restorer_cache:
            return _restorer_cache[key]

        bg_upsampler = _build_bg_upsampler(bg_tile) if use_bg_upsampler else None

        restorer = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=int(upscale),
            arch=str(arch),
            channel_multiplier=int(channel_multiplier),
            bg_upsampler=bg_upsampler,
        )

        _restorer_cache[key] = restorer
        return restorer

# ---------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod job format: {"id": "...", "input": {...}}
    """
    try:
        job_input = job.get("input", {}) or {}

        # weights
        _ensure_weights_exist()

        # params
        upscale = int(job_input.get("upscale", DEFAULTS["upscale"]))
        arch = str(job_input.get("arch", DEFAULTS["arch"]))
        channel_multiplier = int(job_input.get("channel_multiplier", DEFAULTS["channel_multiplier"]))
        use_bg_upsampler = bool(job_input.get("use_bg_upsampler", DEFAULTS["use_bg_upsampler"]))
        bg_tile = int(job_input.get("bg_tile", DEFAULTS["bg_tile"]))

        only_center_face = bool(job_input.get("only_center_face", DEFAULTS["only_center_face"]))
        aligned = bool(job_input.get("aligned", DEFAULTS["aligned"]))
        paste_back = bool(job_input.get("paste_back", DEFAULTS["paste_back"]))

        output_format = str(job_input.get("output_format", DEFAULTS["output_format"]))
        jpeg_quality = int(job_input.get("jpeg_quality", DEFAULTS["jpeg_quality"]))
        return_faces = bool(job_input.get("return_faces", DEFAULTS["return_faces"]))

        # sanity
        if upscale < 1:
            raise ValueError("upscale должен быть >= 1.")
        if arch not in ("clean", "original"):
            raise ValueError("arch должен быть 'clean' или 'original'.")

        # image
        input_img = _get_image_from_input(job_input)

        # inference
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        restorer = _get_restorer(
            upscale=upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            use_bg_upsampler=use_bg_upsampler,
            bg_tile=bg_tile,
        )

        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img,
            has_aligned=aligned,
            only_center_face=only_center_face,
            paste_back=paste_back,
        )

        result: Dict[str, Any] = {
            "meta": {
                "upscale": upscale,
                "arch": arch,
                "channel_multiplier": channel_multiplier,
                "use_bg_upsampler": use_bg_upsampler,
                "bg_tile": bg_tile,
                "only_center_face": only_center_face,
                "aligned": aligned,
                "paste_back": paste_back,
            }
        }

        # main output
        if paste_back:
            if restored_img is None:
                raise RuntimeError("paste_back=True, но restored_img=None.")
            result["image"] = _encode_image(restored_img, output_format, jpeg_quality)
        else:
            # если paste_back=False, логичнее вернуть восстановленные лица
            if len(restored_faces) == 0:
                raise RuntimeError("Лица не найдены или restored_faces пустой.")
            # вернем первое лицо как 'image'
            result["image"] = _encode_image(restored_faces[0], output_format, jpeg_quality)

        # optional faces
        if return_faces:
            result["faces"] = {
                "cropped": [_encode_image(cf, output_format, jpeg_quality) for cf in cropped_faces],
                "restored": [_encode_image(rf, output_format, jpeg_quality) for rf in restored_faces],
            }

        return result

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
