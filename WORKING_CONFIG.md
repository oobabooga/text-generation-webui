# Рабочая конфигурация Text Generation Web UI

## ✅ Успешно протестировано 29.09.2025

### Модель
- **Файл**: `Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf`
- **Размер**: 9.56 GiB (5.56 BPW)
- **Тип**: Q5_K - Small
- **Параметры**: 14.77B

### Железо
- **GPU**: NVIDIA GeForce RTX 3060 (11240 MiB free)
- **VRAM использование**: ~7.7GB
- **GPU layers**: 43/48 слоев на GPU
- **CPU layers**: 5/48 слоев на CPU

### Рабочие параметры
```yaml
loader: llama.cpp
gpu_layers: 43
ctx_size: 8192
batch_size: 256
cache_type: fp16
rope_freq_base: 1000000
flash_attn: enabled
```

### Команда запуска
```bash
cd "D:\gitai\text-generation-webui-3.13"
".\portable_env\python.exe" server.py --model "Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf" --loader llama.cpp --listen --share
```

### Или используйте готовый батник
```bash
start_working.bat
```

### Время загрузки
- **Загрузка модели**: ~7.56 секунд
- **Инициализация**: ~30 секунд общее время

### Доступ
- **Локальный**: http://0.0.0.0:7865
- **Публичный**: https://cf596d58d57763b5a7.gradio.live (ссылка меняется)

### Ключевые особенности успешной конфигурации
1. **Портативная среда**: обязательно использовать `D:\gitai\text-generation-webui-3.13\portable_env\python.exe`
2. **Исходный код**: откатили все изменения к рабочему состоянию
3. **rope_freq_base**: работает с значением 1000000 из метаданных модели
4. **GPU offloading**: оптимальное использование VRAM

### Проблемы, которые были решены
- ❌ Ошибка [WinError 87] - решена использованием портативной среды
- ❌ rope_freq_base конфликты - решены откатом изменений
- ❌ Неправильные GPU layers - автоматически оптимизированы
- ❌ Проблемы с llama-server - решены правильной средой

### НЕ МЕНЯТЬ
- Не изменять код в modules/models_settings.py
- Не изменять код в modules/llama_cpp_server.py  
- Не изменять код в server.py
- Использовать только портативную среду из text-generation-webui-3.13