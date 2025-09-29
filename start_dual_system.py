#!/usr/bin/env python3
"""
Запуск dual системы: Text Generation WebUI + FastAPI Server
Обеспечивает корректную последовательность запуска
"""

import subprocess
import sys
import time
import requests
import os
from pathlib import Path

def check_service(url, name, timeout=5):
    """Проверка доступности сервиса"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except:
        return False

def start_webui():
    """Запуск основной системы WebUI"""
    print("🔄 Запуск Text Generation WebUI...")
    
    # Определяем путь к Python
    python_path = Path("D:/gitai/text-generation-webui-3.13/portable_env/python.exe")
    if not python_path.exists():
        python_path = "python"
    
    # Запуск WebUI
    webui_cmd = [
        str(python_path), "server.py",
        "--model", "Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf",
        "--loader", "llama.cpp",
        "--gpu-layers", "43",
        "--n-ctx", "8192",
        "--api",
        "--listen-port", "7860",
        "--no-stream"
    ]
    
    webui_process = subprocess.Popen(
        webui_cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    print("⏳ Ожидание запуска WebUI...")
    
    # Ожидание запуска WebUI (проверяем доступность)
    max_wait = 120  # 2 минуты
    wait_interval = 5
    waited = 0
    
    while waited < max_wait:
        if check_service("http://127.0.0.1:7860", "WebUI"):
            print("✅ WebUI запущен успешно!")
            break
        
        time.sleep(wait_interval)
        waited += wait_interval
        print(f"⏳ Ожидание... ({waited}/{max_wait}s)")
    else:
        print("❌ WebUI не запустился в отведенное время")
        webui_process.terminate()
        return None
    
    return webui_process

def start_fastapi():
    """Запуск FastAPI сервера"""
    print("🔄 Запуск FastAPI сервера...")
    
    # Определяем путь к Python
    python_path = Path("D:/gitai/text-generation-webui-3.13/portable_env/python.exe")
    if not python_path.exists():
        python_path = "python"
    
    fastapi_cmd = [
        str(python_path), "fastapi_server.py",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]
    
    fastapi_process = subprocess.Popen(
        fastapi_cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    print("⏳ Ожидание запуска FastAPI...")
    
    # Ожидание запуска FastAPI
    max_wait = 30
    wait_interval = 2
    waited = 0
    
    while waited < max_wait:
        if check_service("http://127.0.0.1:8000/health", "FastAPI"):
            print("✅ FastAPI запущен успешно!")
            break
        
        time.sleep(wait_interval)
        waited += wait_interval
        print(f"⏳ Ожидание... ({waited}/{max_wait}s)")
    else:
        print("❌ FastAPI не запустился в отведенное время")
        fastapi_process.terminate()
        return None
    
    return fastapi_process

def main():
    """Основная функция запуска"""
    print("🚀 Запуск комбинированной системы")
    print("=" * 50)
    
    try:
        # Запуск WebUI
        webui_process = start_webui()
        if not webui_process:
            print("❌ Не удалось запустить WebUI")
            return 1
        
        # Дополнительная пауза для полной инициализации
        print("⏳ Дополнительное ожидание для полной инициализации...")
        time.sleep(10)
        
        # Запуск FastAPI
        fastapi_process = start_fastapi()
        if not fastapi_process:
            print("❌ Не удалось запустить FastAPI")
            webui_process.terminate()
            return 1
        
        print("\n🎉 Система запущена успешно!")
        print("=" * 50)
        print("📍 Доступные сервисы:")
        print("  🌐 Gradio WebUI:     http://127.0.0.1:7860")
        print("  🚀 FastAPI Server:   http://127.0.0.1:8000")
        print("  📖 API Docs:         http://127.0.0.1:8000/docs")
        print("  🔄 Redoc:            http://127.0.0.1:8000/redoc")
        print("  🤖 OpenAI API:       http://127.0.0.1:5000")
        print("=" * 50)
        print("⌨️  Нажмите Ctrl+C для остановки")
        
        # Ожидание остановки
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Получен сигнал остановки...")
            
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        return 1
    
    finally:
        # Остановка процессов
        print("🔄 Остановка сервисов...")
        if 'fastapi_process' in locals() and fastapi_process:
            fastapi_process.terminate()
            print("✅ FastAPI остановлен")
        
        if 'webui_process' in locals() and webui_process:
            webui_process.terminate()
            print("✅ WebUI остановлен")
        
        print("✅ Система остановлена")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())