# Рабочая конфигурация Text Generation Web UI

## ✅ ФИНАЛЬНАЯ РАБОЧАЯ ВЕРСИЯ - 29.09.2025

### Модель
- **Файл**: `Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf`
- **Размер**: 9.56 GiB (5.56 BPW)
- **Тип**: Q5_K - Small
- **Параметры**: 14.77B

### Железо
- **GPU**: NVIDIA GeForce RTX 3060 (12GB VRAM)
- **VRAM использование**: ~7.7GB из 12GB доступных
- **GPU layers**: 43/49 слоев на GPU (оптимально!)
- **CPU layers**: 6/49 слоев на CPU

### ✅ ФИНАЛЬНЫЕ РАБОЧИЕ ПАРАМЕТРЫ
```yaml
loader: llama.cpp
model_name: Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf
gpu_layers: 43
ctx_size: 8192
batch_size: 256
cache_type: fp16
rope_freq_base: 1000000
compress_pos_emb: 1
flash_attn: true
listen: true
auto_launch: true
share: true
```

### 🚀 РАБОЧАЯ КОМАНДА ЗАПУСКА
```powershell
cd "D:\gitai\text-generation-webui"
& "D:\gitai\text-generation-webui-3.13\portable_env\python.exe" server.py --model "Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf" --loader llama.cpp --auto-launch --share
```

### 🎯 БЫСТРЫЙ ЗАПУСК
Запустите файл: `quick_start.bat`

### ⚡ Производительность
- **Время загрузки**: ~8.54 секунды
- **Инициализация**: ~30 секунд общее время
- **VRAM оптимизация**: максимальное использование GPU

### 🌐 Доступ после запуска
- **✅ Локальный**: http://127.0.0.1:7860 (РАБОТАЕТ!)
- **✅ Публичный**: https://[random].gradio.live 
- **✅ OpenAI API**: http://127.0.0.1:5000

### 🔧 Ключевые особенности РАБОЧЕЙ конфигурации
1. **Обязательно**: использовать portable Python `D:\gitai\text-generation-webui-3.13\portable_env\python.exe`
2. **Обязательно**: запускать из директории `D:\gitai\text-generation-webui`
3. **Обязательно**: использовать флаг `--auto-launch` для правильного localhost
4. **Оптимально**: 43 GPU слоя для RTX 3060
5. **Автоматически**: определяются rope_freq_base и compress_pos_emb из метаданных

### ✅ РЕШЕННЫЕ ПРОБЛЕМЫ
- ❌ Локальный URL 0.0.0.0:7860 → ✅ 127.0.0.1:7860 (добавлен --auto-launch)
- ❌ Ошибки среды Python → ✅ используем portable_env
- ❌ Неоптимальные GPU слои → ✅ автоопределение 43/49
- ❌ Проблемы с llama-cpp → ✅ правильная установка в portable_env

### 🚫 НЕ ИЗМЕНЯТЬ - РАБОТАЕТ КАК ЕСТЬ!
- ✅ Не менять код в modules/
- ✅ Не менять установленные пакеты
- ✅ Не менять настройки GPU слоев
- ✅ Всегда использовать portable среду