import os
import pathlib
import requests

# GFPGAN v1.4 (официальный релиз)
GFPGAN_V1_4_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"

# RealESRGAN x2plus (используется как bg_upsampler в reference inference_gfpgan.py)
REALESRGAN_X2PLUS_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"


def _download(url: str, dst_path: str, timeout: int = 60) -> None:
    dst = pathlib.Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # идемпотентность
    if dst.exists() and dst.stat().st_size > 0:
        return

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp_path = str(dst) + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp_path, str(dst))


def ensure_weights(weights_dir: str = "/app/weights") -> None:
    os.makedirs(weights_dir, exist_ok=True)

    gfpgan_path = os.path.join(weights_dir, "GFPGANv1.4.pth")
    realesrgan_path = os.path.join(weights_dir, "RealESRGAN_x2plus.pth")

    _download(GFPGAN_V1_4_URL, gfpgan_path)
    _download(REALESRGAN_X2PLUS_URL, realesrgan_path)


if __name__ == "__main__":
    ensure_weights(os.getenv("WEIGHTS_DIR", "/app/weights"))
    print("Weights are ready.")
