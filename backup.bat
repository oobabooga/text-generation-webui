@echo off
REM Скрипт резервного копирования рабочей конфигурации
REM Text Generation Web UI v1.0.0
REM Дата: 29.09.2025

echo === Резервное копирование Text Generation Web UI ===
echo Версия: v1.0.0
echo Дата: %date% %time%
echo.

REM Создание директории для бэкапа
set BACKUP_DIR=backup_%date:~6,4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set BACKUP_DIR=%BACKUP_DIR: =0%
mkdir "%BACKUP_DIR%" 2>nul

echo Создание бэкапа в директории: %BACKUP_DIR%

REM Критически важные файлы конфигурации
echo 📋 Копирование конфигурационных файлов...
copy "user_data\settings.yaml" "%BACKUP_DIR%\" >nul 2>&1 || echo ⚠️  user_data\settings.yaml не найден
xcopy "user_data\models-settings" "%BACKUP_DIR%\models-settings\" /E /I /Q >nul 2>&1 || echo ⚠️  user_data\models-settings\ не найден

REM Документация проекта
echo 📚 Копирование документации...
copy "README_READY.md" "%BACKUP_DIR%\" >nul
copy "WORKING_CONFIG.md" "%BACKUP_DIR%\" >nul
copy "DEVELOPERS.md" "%BACKUP_DIR%\" >nul
copy "CHANGELOG.md" "%BACKUP_DIR%\" >nul
copy "VERSION" "%BACKUP_DIR%\" >nul
copy "start_working.bat" "%BACKUP_DIR%\" >nul

REM Git конфигурация
echo 🔧 Копирование git конфигурации...
copy ".gitignore" "%BACKUP_DIR%\" >nul

REM Создание архива (если доступен PowerShell)
echo 📦 Создание архива...
powershell -Command "Compress-Archive -Path '%BACKUP_DIR%' -DestinationPath '%BACKUP_DIR%.zip'" >nul 2>&1
if %errorlevel% equ 0 (
    rmdir /S /Q "%BACKUP_DIR%"
    echo ✅ Резервная копия создана: %BACKUP_DIR%.zip
) else (
    echo ✅ Резервная копия создана в папке: %BACKUP_DIR%
    echo 💡 Для создания архива требуется PowerShell 5.0+
)

echo.
echo 🎯 Резервная копия содержит:
echo    - Рабочие настройки (settings.yaml)
echo    - Настройки моделей
echo    - Полную документацию
echo    - Скрипты запуска
echo.
echo 📝 Примечание: Модели и логи НЕ включены в бэкап
echo    (скачайте модели отдельно из оригинальных источников)
echo.
echo ✨ Бэкап завершен успешно!
pause