#!/usr/bin/env python3
"""
download_weights.py
Скрипт для предварительной загрузки всех весов моделей GFPGAN и зависимостей.
"""

import os
import time
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
            "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",  # noqa
            "path": os.path.join(GFPGAN_DIR, "GFPGANv1.4.pth"),
        },
    ],
    # FaceXLib - детекция и парсинг лиц
    "facexlib": [
        {
            "url": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",  # noqa
            "path": os.path.join(FACEXLIB_DIR, "detection_Resnet50_Final.pth"),
        },
        {
            "url": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",  # noqa
            "path": os.path.join(FACEXLIB_DIR, "parsing_parsenet.pth"),
        },
    ],
    # RealESRGAN - для апскейла фона (опционально)
    "realesrgan": [
        {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",  # noqa
            "path": os.path.join(REALESRGAN_DIR, "realesr-general-x4v3.pth"),
        },
    ],
}

# Настройки retry
MAX_RETRIES = 5
RETRY_DELAY = 10  # секунд
TIMEOUT = 300  # 5 минут на скачивание


def download_file(url: str, dest_path: str) -> None:
    """Скачивает файл с прогресс-баром и retry логикой."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if os.path.exists(dest_path):
        print(f"✓ Уже существует: {dest_path}")
        return

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"⬇ Скачиваю (попытка {attempt}/{MAX_RETRIES}): {url}")

            response = requests.get(
                url,
                stream=True,
                timeout=TIMEOUT,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            # Сначала скачиваем во временный файл
            temp_path = dest_path + ".tmp"

            with open(temp_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

            # Переименовываем после успешного скачивания
            os.rename(temp_path, dest_path)
            print(f"✓ Сохранено: {dest_path}")
            return

        except (requests.RequestException, IOError) as e:
            print(f"⚠ Попытка {attempt} не удалась: {e}")

            # Удаляем частично скачанный файл
            temp_path = dest_path + ".tmp"
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if attempt < MAX_RETRIES:
                print(f"⏳ Ждём {RETRY_DELAY} секунд "
                      f"перед повторной попыткой...")
                time.sleep(RETRY_DELAY)
            else:
                raise Exception(f"Не удалось скачать {url}"
                                f"после {MAX_RETRIES} попыток")


def setup_environment_paths():
    """Устанавливает переменные окружения для путей к моделям."""
    os.environ['FACEXLIB_WEIGHTS'] = FACEXLIB_DIR

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
            download_file(model["url"], model["path"])
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
                print(f"  {os.path.basename(model['path'])}:"
                      f"{size / 1024 / 1024:.1f} MB")

    print(f"\n📊 Общий размер: {total_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
