@echo off
echo =================================
echo   Text Generation WebUI + FastAPI
echo   Комбинированный запуск
echo =================================
echo.

REM Переход в правильную директорию
cd /d "D:\gitai\text-generation-webui"

REM Проверка наличия модели
if not exist "user_data\models\Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf" (
    echo ОШИБКА: Модель не найдена!
    echo Убедитесь, что файл Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf находится в user_data\models\
    pause
    exit /b 1
)

REM Проверка наличия portable Python среды
if not exist "D:\gitai\text-generation-webui-3.13\portable_env\python.exe" (
    echo ОШИБКА: Portable Python среда не найдена!
    echo Убедитесь, что установка завершена корректно
    pause
    exit /b 1
)

echo Запуск системы с оптимальными настройками...
echo Модель: Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf
echo GPU слои: 43/49
echo Размер контекста: 8192
echo.
echo После запуска будут доступны:
echo - Gradio WebUI: http://127.0.0.1:7860
echo - FastAPI Server: http://127.0.0.1:8000
echo - FastAPI Docs: http://127.0.0.1:8000/docs
echo - OpenAI API: http://127.0.0.1:5000
echo - Публичный доступ: будет показан в консоли
echo.

REM Запуск основной системы в фоне
echo Запуск основной системы Text Generation WebUI...
start "TextGen WebUI" "D:\gitai\text-generation-webui-3.13\portable_env\python.exe" server.py --model "Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf" --loader llama.cpp --share --api

REM Ждем немного для загрузки основной системы
echo Ожидание загрузки основной системы...
timeout /t 30 /nobreak > nul

REM Запуск FastAPI сервера
echo Запуск FastAPI сервера...
"D:\gitai\text-generation-webui-3.13\portable_env\python.exe" fastapi_server.py --host 127.0.0.1 --port 8000

echo.
echo Системы остановлены.
pause