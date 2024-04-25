# GGUF
### Требования:
- Python 3.10/3.11
- Git
- Скачанная модель в формате Transformers
  - Модели в этом формате представляют собой папку, содержащую несколько больших файлов `pytorch_model-XXXXX-of-XXXXX` с расшиернием `.bin` или `.safetensors`,
  а так же несколько `.json` файлов.
- Много свободного места на диске

## Windows
### Подготовка
- Откройте терминал в любой папке, где достаточно свободного места и выполните следующие команды:
  - `git clone --depth 1 https://github.com/ggerganov/llama.cpp.git`
  - `cd llama.cpp`
  - `python -m pip install -r requirements.txt`
- Переместите папку с весами модели в llama.cpp\models (для удобства, это не обязательно)
- Скачайте файл `w64devkit-fortran-<version>.zip` из https://github.com/skeeto/w64devkit/releases/latest и разархивируйте его в любом удобном месте.
Запустите `w64devkit.exe` и с помощью команды cd перейдите в папку с llama.cpp, например так: `cd "A:\LLM models\llama.cpp"`.
Запустите команду `make` и ждите завершения компиляции.
  

### Конвертирование
- Выполните команду `python convert.py models\<your model>\`.
  - Если получаете ошибку `TypeError: <model> must be converted with BpeVocab`, добавьте флаг `--vocab-type bpe` в конец команды.
Например `python convert.py models/Meta-Llama-3-8B/ --vocab-type bpe`.

После этого в папке вашей модели появится файл `ggml-model-f16.gguf` или `ggml-model-f32.gguf`.


### Квантирование
- В командной строке с открытой папкой llama.cpp выполните команду `.\quantize.exe .\<получившийся при конвертации файл .gguf> <метод квантирования>`.
  - Методы квантирования: `Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_0`, `Q4_K_S`, `Q4_K_M`, `Q5_0`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0`.
  Они перечислены в порядке возрастания точности. Чем выше точность, тем больше файл, больше требование к RAM/VRAM, ниже скорость генерации, но тем лучше качество вывода модели.
  Не рекомендуется использовать точность ниже `Q4_0`, рекомендуется `Q4_K_M`, `Q5_K_M` и `Q8_0`.
  - Пример: `.\quantize.exe .\models\Meta-Llama-3-8B\ggml-model-f32.gguf Q8_0`


## Linux
### Подготовка
- Выполните следующие команды в любой удобной директории, где достаточно много свободного места.
  - `git clone --depth 1 https://github.com/ggerganov/llama.cpp.git`
  - `python3 -m pip install -r requirements.txt`
  - `make`
- Переместите папку с весами модели в llama.cpp/models (для удобства, это не обязательно)
  

### Конвертирование
- Выполните команду `python convert.py models/<your model>/`.
  - Если получаете ошибку `TypeError: <model> must be converted with BpeVocab`, добавьте флаг `--vocab-type bpe` в конец команды.
Например `python convert.py models/Meta-Llama-3-8B/ --vocab-type bpe`.

После этого в папке вашей модели появится файл `ggml-model-f16.gguf` или `ggml-model-f32.gguf`.


### Квантирование
- В директории llama.cpp выполните команду `./quantize ./<получившийся при конвертации файл .gguf> <метод квантирования>`.
  - Методы квантирования: `Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_0`, `Q4_K_S`, `Q4_K_M`, `Q5_0`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0`.
  Они перечислены в порядке возрастания точности. Чем выше точность, тем больше файл, больше требование к RAM/VRAM, ниже скорость генерации, но тем лучше качество вывода модели.
  Не рекомендуется использовать точность ниже `Q4_0`, рекомендуется `Q4_K_M`, `Q5_K_M` и `Q8_0`.
  - Пример: `./quantize ./models/Meta-Llama-3-8B/ggml-model-f32.gguf Q8_0`
