import os
import gdown

BASE_DIR = "xray-classification"
MODEL_ID = "1UZx4Ifx-TjpYz4gplzk5Ao-vkMH3mSWM"
MODEL_PATH = os.path.join(BASE_DIR, "model.pt")

# Создаём папку (не упадёт, если уже есть)
if not os.path.exists(BASE_DIR):
  os.makedirs(BASE_DIR, exist_ok=True)
  print(f"[+] Папка готова: {BASE_DIR}")

# Скачиваем модель, если её ещё нет
if os.path.exists(MODEL_PATH):
    print(f"[!] Модель скачана: {MODEL_PATH}")
else:
    print("[-] Скачиваю модель с Google Drive...")
    result = gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)
    if result is None:
        raise RuntimeError("Не удалось скачать модель. Проверьте доступ к файлу.")
    print(f"[!] Модель сохранена: {MODEL_PATH}")
    # Быстрая проверка, что файл на месте
    size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
    print(f"[i] Размер файла: {size_mb:.1f} МБ")
