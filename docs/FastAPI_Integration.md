# FastAPI Integration для Text Generation WebUI

## Обзор

FastAPI сервер добавлен в Text Generation WebUI для предоставления современного REST API интерфейса. Это расширение дополняет существующий OpenAI-совместимый API более гибкими и детальными возможностями.

## Файлы

### Основные компоненты

1. **`fastapi_server.py`** - Основной FastAPI сервер с полным набором endpoints
2. **`test_fastapi.py`** - Клиент для тестирования API endpoints
3. **`start_dual_system.py`** - Скрипт для одновременного запуска WebUI + FastAPI
4. **`start_with_fastapi.bat`** - Windows batch скрипт для dual launch

### Дополнительные файлы

- **`quick_start.bat`** - Обновлен для быстрого запуска с сохраненными настройками
- **`user_data/settings.yaml`** - Сохраненные настройки системы
- **`user_data/CMD_FLAGS.txt`** - Сохраненные флаги командной строки

## API Endpoints

### Основные endpoints

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/health` | GET | Проверка состояния сервера и модели |
| `/generate` | POST | Генерация текста |
| `/chat` | POST | Чат completion (совместимо с OpenAI) |
| `/stop` | POST | Остановка текущей генерации |
| `/model/info` | GET | Информация о загруженной модели |
| `/model/load` | POST | Загрузка новой модели |
| `/model/unload` | POST | Выгрузка текущей модели |
| `/settings` | GET | Получение текущих настроек |
| `/settings` | POST | Обновление настроек |

### Документация API

- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`
- **OpenAPI JSON**: `http://127.0.0.1:8000/openapi.json`

## Конфигурация

### Порты

- **FastAPI Server**: 8000
- **Gradio WebUI**: 7860 (существующий)
- **OpenAI API**: 5000 (существующий)

### Настройки модели

Текущая оптимальная конфигурация:

- **Модель**: `Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf`
- **GPU слои**: 43/49 (оптимально для RTX 3060)
- **Контекст**: 8192 токенов
- **Flash Attention**: включено

## Запуск

### Вариант 1: Dual Launch (рекомендуется)

```bash
# Запуск обеих систем одновременно
python start_dual_system.py
```

### Вариант 2: Поэтапный запуск

```bash
# 1. Запуск основной WebUI
.\quick_start.bat

# 2. В другом терминале запуск FastAPI
python fastapi_server.py
```

### Вариант 3: Windows Batch

```bash
# Автоматический dual launch
.\start_with_fastapi.bat
```

## Примеры использования

### Python клиент

```python
import requests

# Проверка здоровья
response = requests.get("http://127.0.0.1:8000/health")
print(response.json())

# Генерация текста
data = {
    "prompt": "Расскажи о космосе",
    "max_tokens": 200,
    "temperature": 0.7
}
response = requests.post("http://127.0.0.1:8000/generate", json=data)
print(response.json())

# Чат
messages = [{"role": "user", "content": "Привет!"}]
data = {"messages": messages}
response = requests.post("http://127.0.0.1:8000/chat", json=data)
print(response.json())
```

### cURL примеры

```bash
# Проверка состояния
curl http://127.0.0.1:8000/health

# Генерация текста
curl -X POST "http://127.0.0.1:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Привет, мир!",
    "max_tokens": 100,
    "temperature": 0.8
  }'

# Чат completion
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Как дела?"}
    ],
    "max_tokens": 150
  }'
```

## Тестирование

### Автоматическое тестирование

```bash
# Запуск тестового клиента
python test_fastapi.py
```

### Ручное тестирование

1. Откройте `http://127.0.0.1:8000/docs` в браузере
2. Используйте интерактивную документацию Swagger
3. Тестируйте различные endpoints

## Интеграция с существующими системами

### Совместимость

- ✅ Полная совместимость с существующим WebUI
- ✅ Совместимость с OpenAI API endpoints
- ✅ Использует те же модели и настройки
- ✅ Совместное использование состояния системы

### Преимущества FastAPI

1. **Современный асинхронный API**
2. **Автогенерация документации**
3. **Валидация данных с Pydantic**
4. **Поддержка типов Python**
5. **Высокая производительность**
6. **WebSocket поддержка (готова к расширению)**

## Мониторинг и логирование

### Логи

- FastAPI использует стандартное Python логирование
- Логи отображаются в консоли при запуске
- Интеграция с существующей системой логирования WebUI

### Здоровье системы

```bash
# Проверка статуса
curl http://127.0.0.1:8000/health

# Ответ
{
    "status": "healthy",
    "model_loaded": true,
    "model_name": "Qwen2.5-14B-Instruct-Uncensored.i1-Q5_K_S.gguf",
    "timestamp": 1234567890.123
}
```

## Безопасность

### Рекомендации

1. **Использование в локальной сети**: API запускается на `127.0.0.1` по умолчанию
2. **Продакшн настройки**: Для публичного доступа добавьте аутентификацию
3. **CORS**: Настроен для локальной разработки
4. **Rate limiting**: Может быть добавлен при необходимости

### Защита

```python
# Пример добавления аутентификации
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.get("/protected")
async def protected_route(token: str = Depends(security)):
    # Валидация токена
    if not validate_token(token):
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"message": "Access granted"}
```

## Расширение функциональности

### Добавление новых endpoints

```python
@app.post("/custom/endpoint")
async def custom_endpoint(data: CustomModel):
    """Пользовательский endpoint"""
    # Ваша логика здесь
    return {"result": "success"}
```

### WebSocket поддержка

```python
@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    """Стриминг генерации через WebSocket"""
    await websocket.accept()
    # Логика стриминга
```

## Устранение неполадок

### Частые проблемы

1. **Порт занят**: Измените порт в `fastapi_server.py`
2. **Модель не загружена**: Убедитесь что WebUI запущен первым
3. **Зависимости**: Проверьте установку `fastapi`, `uvicorn`

### Диагностика

```bash
# Проверка портов
netstat -an | findstr :8000
netstat -an | findstr :7860

# Проверка процессов
tasklist | findstr python.exe
```

## Развитие

FastAPI интеграция готова к расширению:

- ✅ Streaming поддержка
- ✅ Асинхронная обработка
- ✅ Документация API
- 🔄 WebSocket connections (в планах)
- 🔄 Batch processing (в планах)
- 🔄 Model management UI (в планах)

---

*Автор: GitHub Copilot*  
*Дата: 2024*  
*Версия: 1.0*