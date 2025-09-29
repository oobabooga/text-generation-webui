#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
from datetime import datetime

print('🔍 ФИНАЛЬНАЯ ПРОВЕРКА СИСТЕМЫ')
print('=' * 50)
print(f'Время проверки: {datetime.now().strftime("%H:%M:%S")}')
print()

services = []

# 1. WebUI проверка
print('1️⃣ WebUI (порт 7860):')
try:
    r = requests.get('http://127.0.0.1:7860', timeout=8)
    if r.status_code == 200:
        print('   ✅ Статус: РАБОТАЕТ')
        print('   🌐 URL: http://127.0.0.1:7860')
        services.append('WebUI: ✅')
    else:
        print(f'   ❌ Ошибка статуса: {r.status_code}')
        services.append('WebUI: ❌')
except Exception as e:
    print(f'   ❌ Не отвечает: {str(e)[:50]}')
    services.append('WebUI: ❌')

print()

# 2. OpenAI API проверка с генерацией
print('2️⃣ OpenAI API (порт 5000):')
try:
    # Проверка health
    try:
        health_r = requests.get('http://127.0.0.1:5000/health', timeout=5)
        print(f'   🔹 Health check: {"OK" if health_r.status_code == 200 else "FAIL"}')
    except:
        print('   🔹 Health check: недоступен')
    
    # Тест генерации
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': [{'role': 'user', 'content': 'Привет! Как дела?'}],
        'max_tokens': 20,
        'temperature': 0.7
    }
    
    gen_r = requests.post(
        'http://127.0.0.1:5000/v1/chat/completions',
        headers={'Content-Type': 'application/json'},
        json=data,
        timeout=15
    )
    
    if gen_r.status_code == 200:
        result = gen_r.json()
        if 'choices' in result and result['choices']:
            message = result['choices'][0]['message']['content']
            print('   ✅ Статус: РАБОТАЕТ И ГЕНЕРИРУЕТ')
            print(f'   💬 Тестовый ответ: "{message[:60]}..."')
            print('   🌐 URL: http://127.0.0.1:5000')
            services.append('OpenAI API: ✅')
        else:
            print('   ⚠️  Отвечает, но неправильный формат')
            services.append('OpenAI API: ⚠️')
    else:
        print(f'   ❌ Ошибка генерации: {gen_r.status_code}')
        services.append('OpenAI API: ❌')
        
except Exception as e:
    print(f'   ❌ Не отвечает: {str(e)[:50]}')
    services.append('OpenAI API: ❌')

print()

# 3. FastAPI Proxy проверка
print('3️⃣ FastAPI Proxy (порт 8001):')
try:
    # Health check
    health_r = requests.get('http://127.0.0.1:8001/health', timeout=5)
    if health_r.status_code == 200:
        print('   🔹 Health check: OK')
        
        # Тест генерации через прокси
        data = {'prompt': 'Привет', 'max_tokens': 15}
        gen_r = requests.post('http://127.0.0.1:8001/generate', json=data, timeout=15)
        
        if gen_r.status_code == 200:
            result = gen_r.json()
            print('   ✅ Статус: РАБОТАЕТ И ГЕНЕРИРУЕТ')
            print(f'   💬 Тестовый ответ получен')
            print('   🌐 URL: http://127.0.0.1:8001')
            print('   📚 Документация: http://127.0.0.1:8001/docs')
            services.append('FastAPI Proxy: ✅')
        else:
            print(f'   ⚠️  Health OK, но генерация не работает: {gen_r.status_code}')
            services.append('FastAPI Proxy: ⚠️')
    else:
        print(f'   ❌ Health check failed: {health_r.status_code}')
        services.append('FastAPI Proxy: ❌')
        
except Exception as e:
    print(f'   ❌ Не отвечает: {str(e)[:50]}')
    services.append('FastAPI Proxy: ❌')

print()
print('📊 ИТОГОВЫЙ СТАТУС:')
print('=' * 30)
for service in services:
    print(f'   {service}')

working_count = sum(1 for s in services if '✅' in s)
print()
print(f'🎯 Работающих сервисов: {working_count}/3')

if working_count == 3:
    print('🎉 ВСЯ СИСТЕМА ПОЛНОСТЬЮ РАБОТАЕТ!')
elif working_count >= 2:
    print('⚠️  Система частично работает')
else:
    print('❌ Есть проблемы с системой')

print()
print('🔗 ДОСТУПНЫЕ ССЫЛКИ:')
print('   http://127.0.0.1:7860 - Веб-интерфейс (русский)')
print('   http://127.0.0.1:5000 - OpenAI совместимый API')
print('   http://127.0.0.1:8001 - FastAPI прокси')
print('   http://127.0.0.1:8001/docs - API документация')