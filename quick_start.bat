@echo off
echo =================================
echo   Text Generation WebUI
echo   Быстрый запуск с сохраненными настройками
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
echo - Локальный доступ: http://127.0.0.1:7860
echo - Публичный доступ: будет показан в консоли
echo - OpenAI API: http://127.0.0.1:5000
echo.

REM Основная команда запуска
"D:\gitai\text-generation-webui-3.13\portable_env\python.exe" server.py --model "Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf" --loader llama.cpp --auto-launch --share

echo.
echo Система остановлена.
pause