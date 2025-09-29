"""
Тестовый клиент для FastAPI сервера Text Generation WebUI
Демонстрирует использование различных API endpoints
"""

import requests
import json
import time
from typing import Dict, Any

class TextGenAPIClient:
    """Клиент для работы с FastAPI сервером"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья сервера"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e), "status": "unavailable"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получить информацию о модели"""
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def generate_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Генерация текста"""
        data = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 200),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 20),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.18),
            "stream": kwargs.get("stream", False)
        }
        
        try:
            response = self.session.post(f"{self.base_url}/generate", json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def chat_completion(self, messages: list, **kwargs) -> Dict[str, Any]:
        """Чат completion"""
        data = {
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 200),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": kwargs.get("stream", False)
        }
        
        try:
            response = self.session.post(f"{self.base_url}/chat", json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_settings(self) -> Dict[str, Any]:
        """Получить настройки"""
        try:
            response = self.session.get(f"{self.base_url}/settings")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def update_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Обновить настройки"""
        try:
            response = self.session.post(f"{self.base_url}/settings", json=settings)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def stop_generation(self) -> Dict[str, Any]:
        """Остановить генерацию"""
        try:
            response = self.session.post(f"{self.base_url}/stop")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

def main():
    """Демонстрация использования API"""
    print("🚀 Тестирование FastAPI сервера Text Generation WebUI")
    print("=" * 60)
    
    client = TextGenAPIClient()
    
    # Проверка здоровья
    print("\n1️⃣ Проверка здоровья сервера...")
    health = client.health_check()
    print(f"Результат: {health}")
    
    if health.get("status") != "healthy":
        print("❌ Сервер недоступен или модель не загружена")
        return
    
    # Информация о модели
    print("\n2️⃣ Информация о модели...")
    model_info = client.get_model_info()
    print(f"Модель: {model_info}")
    
    # Простая генерация текста
    print("\n3️⃣ Генерация текста...")
    prompt = "Расскажи интересный факт о космосе:"
    result = client.generate_text(
        prompt=prompt,
        max_tokens=150,
        temperature=0.8
    )
    
    if "error" not in result:
        print(f"Промпт: {prompt}")
        print(f"Ответ: {result['text']}")
        print(f"Токены: {result['tokens']}, Время: {result['generation_time']:.2f}с")
    else:
        print(f"Ошибка: {result['error']}")
    
    # Чат completion
    print("\n4️⃣ Чат completion...")
    messages = [
        {"role": "user", "content": "Привет! Как дела?"},
    ]
    
    chat_result = client.chat_completion(messages=messages, max_tokens=100)
    
    if "error" not in chat_result:
        if "choices" in chat_result and chat_result["choices"]:
            assistant_reply = chat_result["choices"][0]["message"]["content"]
            print(f"Пользователь: {messages[0]['content']}")
            print(f"Ассистент: {assistant_reply}")
            print(f"Использование токенов: {chat_result.get('usage', {})}")
        else:
            print("Нет ответа в chat completion")
    else:
        print(f"Ошибка чата: {chat_result['error']}")
    
    # Получение настроек
    print("\n5️⃣ Текущие настройки...")
    settings = client.get_settings()
    if "error" not in settings:
        key_settings = {k: v for k, v in settings.items() if k in 
                       ['temperature', 'top_p', 'top_k', 'max_tokens', 'repetition_penalty']}
        print(f"Основные настройки: {json.dumps(key_settings, indent=2, ensure_ascii=False)}")
    else:
        print(f"Ошибка получения настроек: {settings['error']}")
    
    print("\n✅ Тестирование завершено!")

if __name__ == "__main__":
    main()