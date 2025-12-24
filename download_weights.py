#!/usr/bin/env python3
"""
download_weights.py
Скрипт для предварительной загрузки всех весов моделей GFPGAN и зависимостей.
"""

import os
import requests
from tqdm import tqdm

# Директории для хранения моделей
WEIGHTS_DIR = "weights"
GFPGAN_DIR = os.path.join(WEIGHTS_DIR, "gfpgan")
FACEXLIB_DIR = os.path.join(WEIGHTS_DIR, "facexlib")
REALESRGAN_DIR = os.path.join(WEIGHTS_DIR, "realesrgan")

# Все модели для скачивания
MODELS = {
    # GFPGAN основная модель
    "gfpgan": [
        {
            "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
            "path": os.path.join(GFPGAN_DIR, "GFPGANv1.4.pth"),
        },
    ],
    # FaceXLib - детекция и парсинг лиц
    "facexlib": [
        {
            "url": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
            "path": os.path.join(FACEXLIB_DIR, "detection_Resnet50_Final.pth"),
        },
        {
            "url": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
            "path": os.path.join(FACEXLIB_DIR, "parsing_parsenet.pth"),
        },
    ],
    # RealESRGAN - для апскейла фона (опционально, но рекомендуется)
    "realesrgan": [
        {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "path": os.path.join(REALESRGAN_DIR, "RealESRGAN_x2plus.pth"),
        },
    ],
}


def download_file(url: str, dest_path: str) -> None:
    """Скачивает файл с прогресс-баром."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(dest_path):
        print(f"✓ Уже существует: {dest_path}")
        return
    
    print(f"⬇ Скачиваю: {url}")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    
    print(f"✓ Сохранено: {dest_path}")


def setup_environment_paths():
    """Устанавливает переменные окружения для путей к моделям."""
    # facexlib ищет модели по этим путям
    os.environ['FACEXLIB_WEIGHTS'] = FACEXLIB_DIR
    
    # Создаём симлинки в стандартные директории (для совместимости)
    standard_paths = [
        (GFPGAN_DIR, "gfpgan/weights"),
        (FACEXLIB_DIR, "facexlib/weights"),
    ]
    
    for src, dst in standard_paths:
        dst_dir = os.path.dirname(dst)
        if dst_dir and not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        if not os.path.exists(dst) and os.path.exists(src):
            try:
                os.symlink(os.path.abspath(src), dst)
                print(f"🔗 Симлинк: {dst} -> {src}")
            except OSError:
                pass


def main():
    print("=" * 60)
    print("Загрузка весов моделей GFPGAN")
    print("=" * 60)
    
    # Создаём директории
    for dir_path in [WEIGHTS_DIR, GFPGAN_DIR, FACEXLIB_DIR, REALESRGAN_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Скачиваем все модели
    for category, models in MODELS.items():
        print(f"\n📦 {category.upper()}")
        print("-" * 40)
        for model in models:
            try:
                download_file(model["url"], model["path"])
            except Exception as e:
                print(f"❌ Ошибка при скачивании {model['url']}: {e}")
                raise
    
    # Настраиваем пути
    setup_environment_paths()
    
    print("\n" + "=" * 60)
    print("✅ Все модели успешно загружены!")
    print("=" * 60)
    
    # Выводим итоговые размеры
    total_size = 0
    for category, models in MODELS.items():
        for model in models:
            if os.path.exists(model["path"]):
                size = os.path.getsize(model["path"])
                total_size += size
                print(f"  {os.path.basename(model['path'])}: {size / 1024 / 1024:.1f} MB")
    
    print(f"\n📊 Общий размер: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()