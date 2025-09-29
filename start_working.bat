@echo off
REM Рабочий запуск Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf
REM Протестировано и работает 29.09.2025

echo Запускаем Text Generation Web UI с рабочей конфигурацией...
echo Модель: Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf
echo GPU: RTX 3060 (43 layers on GPU)
echo.

cd /d "D:\gitai\text-generation-webui-3.13"
".\portable_env\python.exe" server.py --model "Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf" --loader llama.cpp --listen --share

pause