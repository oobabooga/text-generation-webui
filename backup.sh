#!/bin/bash
# Скрипт резервного копирования рабочей конфигурации
# Text Generation Web UI v1.0.0
# Дата: 29.09.2025

echo "=== Резервное копирование Text Generation Web UI ==="
echo "Версия: v1.0.0"
echo "Дата: $(date)"
echo ""

# Создание директории для бэкапа
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Создание бэкапа в директории: $BACKUP_DIR"

# Критически важные файлы конфигурации
echo "📋 Копирование конфигурационных файлов..."
cp user_data/settings.yaml "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  user_data/settings.yaml не найден"
cp -r user_data/models-settings/ "$BACKUP_DIR/" 2>/dev/null || echo "⚠️  user_data/models-settings/ не найден"

# Документация проекта
echo "📚 Копирование документации..."
cp README_READY.md "$BACKUP_DIR/"
cp WORKING_CONFIG.md "$BACKUP_DIR/"
cp DEVELOPERS.md "$BACKUP_DIR/"
cp CHANGELOG.md "$BACKUP_DIR/"
cp VERSION "$BACKUP_DIR/"
cp start_working.bat "$BACKUP_DIR/"

# Git конфигурация
echo "🔧 Копирование git конфигурации..."
cp .gitignore "$BACKUP_DIR/"

# Создание архива
echo "📦 Создание архива..."
if command -v zip &> /dev/null; then
    zip -r "${BACKUP_DIR}.zip" "$BACKUP_DIR"
    rm -rf "$BACKUP_DIR"
    echo "✅ Резервная копия создана: ${BACKUP_DIR}.zip"
else
    echo "✅ Резервная копия создана в папке: $BACKUP_DIR"
    echo "💡 Для создания архива установите zip: sudo apt install zip"
fi

echo ""
echo "🎯 Резервная копия содержит:"
echo "   - Рабочие настройки (settings.yaml)"
echo "   - Настройки моделей"
echo "   - Полную документацию"
echo "   - Скрипты запуска"
echo ""
echo "📝 Примечание: Модели и логи НЕ включены в бэкап"
echo "   (скачайте модели отдельно из оригинальных источников)"
echo ""
echo "✨ Бэкап завершен успешно!"