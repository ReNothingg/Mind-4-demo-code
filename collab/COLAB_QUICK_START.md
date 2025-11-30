# Быстрый старт обучения Mind-4 в Google Colab

## Самый быстрый способ начать

### Копируйте этот код в Google Colab (colab.research.google.com)

**Ячейка 1: Установка и клонирование**

```python
# Установка PyTorch с CUDA поддержкой
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q

# Установка дополнительных пакетов
!pip install pyyaml tqdm -q

# Клонируем репо (или используйте git pull если уже есть)
!git clone https://github.com/ReNothingg/Mind-4-demo-code.git 2>/dev/null || (cd Mind-4-demo-code && git pull)

# Переходим в папку
%cd Mind-4-demo-code

# Проверяем GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Ячейка 2: Подготовка данных (выбирите один вариант)**

```python
# ===== ВАРИАНТ A: Загрузить свой файл =====
from google.colab import files
import shutil

print("Выберите файл train_data.txt")
uploaded = files.upload()

# Переместить в нужную папку
for filename in uploaded.keys():
    src = f"/content/{filename}"
    dst = "./train/train_data.txt"
    shutil.copy(src, dst)
    print(f"Файл скопирован: {dst}")

# Проверяем что скопилось
!ls -lh ./train/train_data.txt
```

```python
# ===== ВАРИАНТ B: Использовать Google Drive =====
from google.colab import drive
import shutil

# Монтируем Drive
drive.mount('/content/gdrive')

# Копируем данные
shutil.copy('/content/gdrive/My Drive/train_data.txt', './train/train_data.txt')
print("  Данные скопированы с Google Drive")
```

```python
# ===== ВАРИАНТ C: Создать тестовые данные =====
# Для теста создадим файл с примерами
test_data = """
The quick brown fox jumps over the lazy dog.
Machine learning is fascinating and powerful.
Python is great for scientific computing.
Neural networks learn patterns from data.
Transformers have revolutionized NLP.
Deep learning requires lots of data and compute.
GPUs make training much faster.
This is a training example for the Mind-4 model.
Artificial intelligence is transforming the world.
Language models can generate impressive text.
""" * 100  # Повторяем чтобы было побольше данных

with open('./train/train_data.txt', 'w') as f:
    f.write(test_data)

print(f"Создан тестовый файл ({len(test_data)} символов)")
```

**Ячейка 3: Малая конфигурация (для тестирования на T4)**

```python
import yaml

config_small = """
model:
  hidden_size: 512
  num_hidden_layers: 8
  num_attention_heads: 8
  num_key_value_heads: 2
  vocab_size: 50000
  head_dim: 64
  intermediate_size: 1024
  num_experts: 4
  experts_per_token: 2
  rope_theta: 10000.0
  sliding_window: 1024
  initial_context_length: 1024
  rope_scaling_factor: 1.0
  rope_ntk_alpha: 1.0
  rope_ntk_beta: 32.0
  swiglu_limit: 1.0

data:
  train_dataset: "./train/train_data.txt"
  val_dataset: null
  max_seq_length: 512
  batch_size: 2
  num_workers: 0
  pin_memory: false

training:
  num_epochs: 1
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0
  warmup_steps: 50
  logging_steps: 10
  eval_steps: 100
  save_steps: 100
  save_path: "./checkpoints/mind_epoch_{epoch}.pt"

  optimizer:
    type: "AdamW"
    lr: 1e-4
    betas: [0.9, 0.95]
    weight_decay: 0.01
    eps: 1e-8

hardware:
  device: "cuda"
  mixed_precision: "fp16"
  gradient_checkpointing: true
"""

with open('./config/train_small.yaml', 'w') as f:
    f.write(config_small)

print("  Конфигурация для маленькой модели сохранена")
```

**Ячейка 4: Запуск обучения**

```python
# Запускаем обучение
!python colab_train.py \
    --config config/train_small.yaml \
    --data ./train/train_data.txt \
    --max-samples 1000 \
    --dry-run  # Убирите флаг --dry-run для полного обучения
```

**Ячейка 5: Скачиваем результаты**

```python
from google.colab import files
import os

# Скачиваем чекпоинты
checkpoint_dir = './checkpoints'
for file in os.listdir(checkpoint_dir):
    if file.endswith('.pt'):
        files.download(os.path.join(checkpoint_dir, file))
        print(f"Скачан: {file}")
```

---

## Правильный порядок действий

1. **Ячейка 1**: Установка и проверка GPU
2. **Ячейка 2**: Загрузка данных (выберите один вариант A, B или C)
3. **Ячейка 3**: Создание конфигурации
4. **Ячейка 4**: Запуск обучения
5. **Ячейка 5**: Скачивание результатов

---

## ⚙️ Параметры для разных GPU

### Для T4 (бесплатный Colab, ~16 GB памяти)

```yaml
hidden_size: 512
num_hidden_layers: 8
batch_size: 2
max_seq_length: 512
```

### Для P100 (Pro Colab, ~40 GB)

```yaml
hidden_size: 768
num_hidden_layers: 12
batch_size: 4
max_seq_length: 1024
```

### Для A100 (Premium Colab, ~80 GB)

```yaml
hidden_size: 1024
num_hidden_layers: 16
batch_size: 8
max_seq_length: 2048
```

---

## 🐛 Решение проблем

| Проблема | Решение |
|----------|---------|
| CUDA out of memory | Уменьшите `batch_size` или `max_seq_length` |
| Медленное обучение | Убедитесь что используется GPU (`torch.cuda.is_available()`) |
| Модель не загружается | Проверьте что находитесь в папке `Mind-4-demo-code` |
| Нет данных | Загрузите файл `train_data.txt` в Colab (Ячейка 2) |

---

## Сохранение моделей

После обучения скачайте файлы из `./checkpoints/`:

```python
# Быстрое сохранение на Google Drive
!mkdir -p /content/gdrive/My\ Drive/mind-4-checkpoints
!cp ./checkpoints/*.pt /content/gdrive/My\ Drive/mind-4-checkpoints/
```

---

## Отслеживание прогресса

В процессе обучения вы будете видеть:

```
==================================================
Эпоха 1/1
==================================================
Обучение: 100%|██████| 500/500 [3:45<00:00, 2.23 batches/s]
Train Loss: 3.2145
```

Loss должен **убывать** - это хороший знак! 📈

---

## Следующие шаги после обучения

1. **Загрузите модель локально:**
   ```python
   import torch
   checkpoint = torch.load('mind_epoch_1.pt')
   model.load_state_dict(checkpoint['model_state_dict'])
   ```

2. **Используйте для генерации:**
   ```bash
   python model/generate.py ./checkpoints/mind_epoch_1.pt --prompt "Hello"
   ```

3. **Улучшайте архитектуру и данные:**
   - Больше слоев = лучше качество (но медленнее)
   - Качественные данные = лучший результат
   - Больше экспертов = лучше способность, но больше параметров

---

## Важные замечания

- **Бесплатный Colab отключается через 12 часов** - сохраняйте регулярно!
- **RAM обнуляется** при переподключении - заново запустите установку
- **Лучше использовать маленькие модели** при первом тесте
- **Экспериментируйте с параметрами** - каждый набор данных уникален
