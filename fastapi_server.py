"""
FastAPI сервер для Text Generation WebUI
Предоставляет RESTful API endpoints для работы с языковыми моделями
"""

import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Добавляем путь к модулям Text Generation WebUI
sys.path.append(str(Path(__file__).parent))

# Импорты из модулей Text Generation WebUI
try:
    from modules import shared, models, text_generation, chat
    from modules.models import load_model, unload_model
    from modules.text_generation import generate_reply, stop_everything_event
    from modules.chat import generate_chat_reply
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}")
    print("Убедитесь, что сервер запущен из корневой директории Text Generation WebUI")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fastapi_server")

# Модели данных Pydantic
class GenerateRequest(BaseModel):
    """Запрос на генерацию текста"""
    prompt: str = Field(..., description="Входной промпт для генерации")
    max_tokens: int = Field(200, description="Максимальное количество токенов")
    temperature: float = Field(0.7, description="Температура генерации (0.0-2.0)")
    top_p: float = Field(0.9, description="Top-p сэмплинг")
    top_k: int = Field(20, description="Top-k сэмплинг")
    repetition_penalty: float = Field(1.18, description="Штраф за повторения")
    stream: bool = Field(False, description="Потоковая генерация")

class ChatMessage(BaseModel):
    """Сообщение в чате"""
    role: str = Field(..., description="Роль: user, assistant, system")
    content: str = Field(..., description="Содержимое сообщения")

class ChatRequest(BaseModel):
    """Запрос на чат"""
    messages: List[ChatMessage] = Field(..., description="История сообщений")
    max_tokens: int = Field(200, description="Максимальное количество токенов")
    temperature: float = Field(0.7, description="Температура генерации")
    stream: bool = Field(False, description="Потоковая генерация")

class ModelInfo(BaseModel):
    """Информация о модели"""
    name: Optional[str] = Field(None, description="Название модели")
    loader: Optional[str] = Field(None, description="Тип загрузчика")
    loaded: bool = Field(False, description="Загружена ли модель")

class GenerateResponse(BaseModel):
    """Ответ на генерацию"""
    text: str = Field(..., description="Сгенерированный текст")
    tokens: int = Field(..., description="Количество токенов")
    generation_time: float = Field(..., description="Время генерации в секундах")

class StatusResponse(BaseModel):
    """Статус сервера"""
    status: str = Field(..., description="Статус сервера")
    model: ModelInfo = Field(..., description="Информация о модели")
    api_version: str = Field("1.0", description="Версия API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    logger.info("Запуск FastAPI сервера для Text Generation WebUI")
    yield
    logger.info("Остановка FastAPI сервера")

# Создание приложения FastAPI
app = FastAPI(
    title="Text Generation WebUI FastAPI",
    description="RESTful API для работы с языковыми моделями через Text Generation WebUI",
    version="1.0.0",
    lifespan=lifespan
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Вспомогательные функции
def get_model_info() -> ModelInfo:
    """Получить информацию о текущей модели"""
    return ModelInfo(
        name=getattr(shared, 'model_name', None),
        loader=getattr(shared.args, 'loader', None),
        loaded=hasattr(shared, 'model') and shared.model is not None
    )

def apply_generation_params(request_data: dict):
    """Применить параметры генерации к shared настройкам"""
    if 'temperature' in request_data:
        shared.settings['temperature'] = request_data['temperature']
    if 'top_p' in request_data:
        shared.settings['top_p'] = request_data['top_p']
    if 'top_k' in request_data:
        shared.settings['top_k'] = request_data['top_k']
    if 'repetition_penalty' in request_data:
        shared.settings['repetition_penalty'] = request_data['repetition_penalty']
    if 'max_tokens' in request_data:
        shared.settings['max_tokens'] = request_data['max_tokens']

# API Endpoints

@app.get("/", response_model=StatusResponse)
async def root():
    """Корневой endpoint - информация о сервере"""
    return StatusResponse(
        status="running",
        model=get_model_info()
    )

@app.get("/health")
async def health_check():
    """Проверка здоровья сервера"""
    model_info = get_model_info()
    return {
        "status": "healthy" if model_info.loaded else "no_model",
        "timestamp": asyncio.get_event_loop().time()
    }

@app.get("/model/info", response_model=ModelInfo)
async def get_model_information():
    """Получить информацию о загруженной модели"""
    return get_model_info()

@app.post("/model/load")
async def load_model_endpoint(model_name: str, background_tasks: BackgroundTasks):
    """Загрузить модель"""
    if not model_name:
        raise HTTPException(status_code=400, detail="Не указано имя модели")
    
    def load_model_task():
        try:
            shared.model_name = model_name
            load_model(model_name)
            logger.info(f"Модель {model_name} успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели {model_name}: {e}")
    
    background_tasks.add_task(load_model_task)
    return {"message": f"Начата загрузка модели {model_name}"}

@app.post("/model/unload")
async def unload_model_endpoint():
    """Выгрузить текущую модель"""
    try:
        unload_model()
        logger.info("Модель выгружена")
        return {"message": "Модель успешно выгружена"}
    except Exception as e:
        logger.error(f"Ошибка выгрузки модели: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка выгрузки модели: {e}")

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Генерация текста по промпту"""
    model_info = get_model_info()
    if not model_info.loaded:
        raise HTTPException(status_code=400, detail="Модель не загружена")
    
    # Применяем параметры генерации
    apply_generation_params(request.dict())
    
    try:
        import time
        start_time = time.time()
        
        # Генерируем ответ
        result = text_generation.generate_reply(
            prompt=request.prompt,
            state={'max_tokens': request.max_tokens},
            stopping_strings=[],
            is_chat=False,
            escape_html=False
        )
        
        generation_time = time.time() - start_time
        
        # Извлекаем текст из результата
        if isinstance(result, tuple) and len(result) > 0:
            generated_text = result[0]
        else:
            generated_text = str(result)
        
        # Подсчет токенов (приблизительно)
        token_count = len(generated_text.split())
        
        return GenerateResponse(
            text=generated_text,
            tokens=token_count,
            generation_time=generation_time
        )
        
    except Exception as e:
        logger.error(f"Ошибка генерации: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {e}")

@app.post("/generate/stream")
async def generate_text_stream(request: GenerateRequest):
    """Потоковая генерация текста"""
    model_info = get_model_info()
    if not model_info.loaded:
        raise HTTPException(status_code=400, detail="Модель не загружена")
    
    apply_generation_params(request.dict())
    
    async def generate():
        try:
            # Начинаем генерацию
            generator = text_generation.generate_reply(
                prompt=request.prompt,
                state={'max_tokens': request.max_tokens},
                stopping_strings=[],
                is_chat=False,
                escape_html=False,
                stream=True
            )
            
            for chunk in generator:
                if isinstance(chunk, tuple) and len(chunk) > 0:
                    text_chunk = chunk[0]
                else:
                    text_chunk = str(chunk)
                
                yield f"data: {json.dumps({'text': text_chunk})}\n\n"
                
        except Exception as e:
            logger.error(f"Ошибка потоковой генерации: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/chat")
async def chat_completion(request: ChatRequest):
    """Чат completion в стиле OpenAI"""
    model_info = get_model_info()
    if not model_info.loaded:
        raise HTTPException(status_code=400, detail="Модель не загружена")
    
    apply_generation_params(request.dict())
    
    try:
        # Конвертируем сообщения в формат Text Generation WebUI
        history = {'internal': [], 'visible': []}
        
        for msg in request.messages:
            if msg.role == 'user':
                history['internal'].append([msg.content, ''])
                history['visible'].append([msg.content, ''])
            elif msg.role == 'assistant':
                if history['internal'] and len(history['internal'][-1]) == 2:
                    history['internal'][-1][1] = msg.content
                    history['visible'][-1][1] = msg.content
        
        # Получаем последнее сообщение пользователя
        user_input = request.messages[-1].content if request.messages else ""
        
        # Генерируем ответ
        reply = chat.generate_chat_reply(
            text=user_input,
            history=history,
            state={'max_tokens': request.max_tokens},
            regenerate=False,
            _continue=False,
            loading_message=True
        )
        
        # Извлекаем ответ
        if isinstance(reply, tuple) and len(reply) > 1:
            response_text = reply[1]['visible'][-1][1] if reply[1]['visible'] else ""
        else:
            response_text = str(reply)
        
        return {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(user_input.split()),
                "completion_tokens": len(response_text.split()),
                "total_tokens": len(user_input.split()) + len(response_text.split())
            }
        }
        
    except Exception as e:
        logger.error(f"Ошибка чата: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка чата: {e}")

@app.post("/stop")
async def stop_generation():
    """Остановить текущую генерацию"""
    try:
        stop_everything_event.set()
        return {"message": "Генерация остановлена"}
    except Exception as e:
        logger.error(f"Ошибка остановки генерации: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка остановки: {e}")

@app.get("/settings")
async def get_settings():
    """Получить текущие настройки"""
    return shared.settings

@app.post("/settings")
async def update_settings(settings: Dict[str, Any]):
    """Обновить настройки"""
    try:
        for key, value in settings.items():
            if key in shared.settings:
                shared.settings[key] = value
        return {"message": "Настройки обновлены", "updated": settings}
    except Exception as e:
        logger.error(f"Ошибка обновления настроек: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обновления настроек: {e}")

if __name__ == "__main__":
    import sys
    
    # Очищаем sys.argv чтобы избежать конфликтов с аргументами основной системы
    original_argv = sys.argv.copy()
    # Фильтруем только наши аргументы
    filtered_argv = [original_argv[0]]  # Оставляем имя скрипта
    
    # Ищем наши специфичные аргументы
    for i, arg in enumerate(original_argv[1:], 1):
        if arg in ["--host", "--port", "--reload"] or (i > 1 and original_argv[i-1] in ["--host", "--port"]):
            filtered_argv.append(arg)
    
    sys.argv = filtered_argv
    
    import argparse
    
    parser = argparse.ArgumentParser(description="FastAPI сервер для Text Generation WebUI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Хост для сервера")
    parser.add_argument("--port", type=int, default=8000, help="Порт для сервера")
    parser.add_argument("--reload", action="store_true", help="Автоперезагрузка при изменениях")
    
    args = parser.parse_args()
    
    print("🚀 Запуск FastAPI сервера для Text Generation WebUI")
    print("=" * 60)
    print(f"🌐 Сервер будет доступен по адресу: http://{args.host}:{args.port}")
    print(f"📖 API документация: http://{args.host}:{args.port}/docs")
    print(f"🔄 Интерактивный API: http://{args.host}:{args.port}/redoc")
    print("=" * 60)
    
    logger.info(f"Запуск FastAPI сервера на {args.host}:{args.port}")
    
    try:
        uvicorn.run(
            "fastapi_server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    finally:
        # Восстанавливаем оригинальные аргументы
        sys.argv = original_argv