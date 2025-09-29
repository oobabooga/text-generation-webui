#!/usr/bin/env python3
"""
FastAPI Proxy Server для Text Generation WebUI
Проксирует запросы к основной системе с дополнительными endpoint'ами
"""

import asyncio
import json
import time
from typing import Optional, Dict, Any, List
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import argparse

# Модели данных
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stop: Optional[List[str]] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    proxy_mode: bool = True
    backend_status: str

# Создание FastAPI приложения
app = FastAPI(
    title="Text Generation WebUI - FastAPI Proxy",
    description="Прокси сервер для Text Generation WebUI с дополнительными endpoint'ами",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация
BACKEND_URL = "http://127.0.0.1:5000"
client = httpx.AsyncClient(timeout=60.0)

@app.on_event("startup")
async def startup_event():
    print("🚀 FastAPI Proxy Server запускается...")
    print(f"📡 Бэкенд: {BACKEND_URL}")

@app.on_event("shutdown")
async def shutdown_event():
    await client.aclose()
    print("👋 FastAPI Proxy Server остановлен")

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "Text Generation WebUI - FastAPI Proxy",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate": "/generate", 
            "chat": "/chat",
            "openai_proxy": "/v1/chat/completions",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка состояния системы"""
    try:
        # Проверяем основной сервер
        response = await client.get(f"{BACKEND_URL}/v1/models", timeout=5.0)
        backend_status = "ready" if response.status_code == 200 else "error"
    except Exception:
        backend_status = "unavailable"
    
    return HealthResponse(
        status="ready" if backend_status in ["ready"] else "error",
        timestamp=time.time(),
        proxy_mode=True,
        backend_status=backend_status
    )

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """Генерация текста через простой prompt"""
    try:
        # Формируем запрос к OpenAI API
        openai_request = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": request.prompt}
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop": request.stop
        }
        
        # Отправляем запрос к основному серверу
        response = await client.post(
            f"{BACKEND_URL}/v1/chat/completions",
            json=openai_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and result["choices"]:
                generated_text = result["choices"][0]["message"]["content"]
                return {
                    "response": generated_text,
                    "status": "success",
                    "prompt": request.prompt,
                    "timestamp": time.time()
                }
            else:
                raise HTTPException(status_code=500, detail="Неожиданный формат ответа")
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Ошибка бэкенда: {response.text}")
            
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Бэкенд недоступен")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")

@app.post("/chat")
async def chat_completion(request: ChatRequest):
    """Chat completion endpoint"""
    try:
        # Формируем запрос к OpenAI API
        openai_request = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream
        }
        
        response = await client.post(
            f"{BACKEND_URL}/v1/chat/completions",
            json=openai_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and result["choices"]:
                generated_text = result["choices"][0]["message"]["content"]
                return {
                    "response": generated_text,
                    "status": "success",
                    "messages": len(request.messages),
                    "timestamp": time.time()
                }
            else:
                raise HTTPException(status_code=500, detail="Неожиданный формат ответа")
        else:
            raise HTTPException(status_code=response.status_code, detail=f"Ошибка бэкенда: {response.text}")
            
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Бэкенд недоступен")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {str(e)}")

@app.post("/v1/chat/completions")
async def openai_proxy(request: Request):
    """Прокси для OpenAI API совместимости"""
    try:
        body = await request.body()
        
        response = await client.post(
            f"{BACKEND_URL}/v1/chat/completions",
            content=body,
            headers={"Content-Type": "application/json"}
        )
        
        return JSONResponse(
            content=response.json() if response.status_code == 200 else {"error": response.text},
            status_code=response.status_code
        )
        
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Бэкенд недоступен")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка прокси: {str(e)}")

@app.get("/info")
async def server_info():
    """Информация о сервере"""
    return {
        "server": "FastAPI Proxy for Text Generation WebUI",
        "version": "1.0.0",
        "proxy_mode": True,
        "backend": BACKEND_URL,
        "timestamp": time.time(),
        "features": [
            "OpenAI API Proxy",
            "Simple Generate Endpoint", 
            "Chat Completion",
            "Health Monitoring",
            "CORS Support"
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="FastAPI Proxy Server")
    parser.add_argument("--port", type=int, default=8000, help="Порт сервера")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Хост сервера")
    parser.add_argument("--backend", type=str, default="http://127.0.0.1:5000", help="URL бэкенда")
    args = parser.parse_args()
    
    global BACKEND_URL
    BACKEND_URL = args.backend
    
    print(f"🚀 Запуск FastAPI Proxy Server на {args.host}:{args.port}")
    print(f"📡 Проксирование к: {BACKEND_URL}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )

if __name__ == "__main__":
    main()