# LLM* на ESP32-S3: 406 токенов в секунду

Llama-2 [stories260K](https://huggingface.co/karpathy/tinyllamas) (260 тыс. параметров, TinyStories) на микроконтроллере ESP32-S3.
Код к первой части статьи: int8-веса в SRAM, ассемблерные ядра на векторных инструкциях PIE, второе ядро и переписанное внимание.
*TinyLLM

| Режим | Что включено | tok/s |
|---|---|---|
| 0 | float32, PSRAM, 1 ядро | 22.5 |
| 1 | int8, PSRAM, 1 ядро | 55.3 |
| 2 | int8, флеш (XIP), 1 ядро | 53.3 |
| 3 | int8, внутренняя SRAM, 1 ядро | 71.7 |
| 4 | + векторные инструкции PIE | 175.2 |
| 5 | + второе ядро | 285.5 |
| 6 | + внимание v2 (обе головы GQA за один проход) | **406.1** |
| 7 | + целочисленное Q·K | 403.6 |

Средняя скорость на 128 токенах, плата ESP32-S3 с 2 МБ PSRAM. Каждый режим сверяется с симулятором на numpy: токены совпадают во всех режимах.

## Быстрый старт

Прошивка уже собрана (`firmware/prebuilt`), ESP-IDF не нужен. Подойдёт любая плата на ESP32-S3.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pyserial esptool

python esp_llm.py flash      # найдёт порт, определит тип PSRAM, соберёт и запишет модель
python esp_llm.py verify     # сверка прошивки с симулятором, ждём RESULT: PASS
python esp_llm.py bench      # скорость всех режимов
python esp_llm.py gen --prompt "Lily had a little dog"
```

Если плата после прошивки не загружается, используйте `python esp_llm.py flash --variant safe` (флеш в режиме DIO).
Если esptool не может подключиться: зажмите BOOT, нажмите RESET, отпустите BOOT.

Без платы: `python esp_llm.py analyze` сравнивает float32, int8, int6, int4 и полный конвейер прошивки по качеству.

## Состав

- `esp_llm.py` — эталон на numpy (совпадает с `run.c`), квантизатор, экспорт модели, прошивка, проверка, замеры.
- `firmware/main/llm.c` — движок: int8-веса и активации, int8 KV-кэш, слитые матрицы, быстрая экспонента.
- `firmware/main/mv_pie.S` — матрично-векторные ядра на `ee.vmulas.s8.accx`.
- `firmware/main/main.c` — размещение в памяти, второе ядро, протокол по serial.
- `stories260K.bin`, `tok512.bin` — модель и токенизатор из [llama2.c](https://github.com/karpathy/llama2.c).

## Пересборка прошивки

ESP-IDF 5.3: `python esp_llm.py flash --build --variant quad` (или `octal` для плат с 8 МБ PSRAM).

## Лицензия

MIT. Модель и токенизатор — MIT, Andrej Karpathy.
