# Руководство по обучению Mind-4 в Google Colab

## Пошаговая инструкция

### Шаг 1: Подготовка Google Colab

Создайте новый ноутбук на `colab.research.google.com` и скопируйте этот код в первую ячейку:

```python
# Установка зависимостей
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install peft transformers safetensors tqdm wandb pyyaml
!pip install triton  # опционально, для ускорения на некоторых GPU

# Клонирование репозитория
!git clone https://github.com/ReNothingg/Mind-4-demo-code.git
%cd Mind-4-demo-code
```

### Шаг 2: Подготовка данных для обучения

Colab может работать с данными несколькими способами:

#### Вариант A: Загрузите файл данных локально

```python
# Загрузите файл train_data.txt (максимум 256MB для бесплатного Colab)
from google.colab import files
uploaded = files.upload()

# Переместите файл в нужную папку
import shutil
shutil.move(list(uploaded.keys())[0], './train/train_data.txt')
```

#### Вариант B: Используйте Google Drive

```python
from google.colab import drive
drive.mount('/content/gdrive')

# Скопируйте данные с Drive
!cp /content/gdrive/My\ Drive/train_data.txt ./train/train_data.txt
!cp /content/gdrive/My\ Drive/val_data.txt ./evals/val_data.txt
```

#### Вариант C: Загрузите из облака (HuggingFace, S3, и т.д.)

```python
# Например, загрузить из HuggingFace Datasets
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Сохранить в файл
with open('./train/train_data.txt', 'w') as f:
    for item in dataset['train'][:10000]:
        f.write(item['text'] + '\n')
```

### Шаг 3: Обновление конфигурации

Для обучения в Colab отредактируйте `config/train.yaml`:

```python
import yaml

config = """
model:
  hidden_size: 768          # меньший размер для Colab
  num_hidden_layers: 12
  num_attention_heads: 12
  num_key_value_heads: 4
  vocab_size: 50304
  head_dim: 64
  intermediate_size: 2048
  num_experts: 4            # меньше экспертов
  experts_per_token: 2
  rope_theta: 10000.0
  sliding_window: 4096
  initial_context_length: 2048  # меньший контекст
  max_context_length: 8192
  rope_scaling_factor: 1.0
  rope_ntk_alpha: 1.0
  rope_ntk_beta: 32.0
  swiglu_limit: 1.0
  embedding_dropout: 0.1
  layer_norm_eps: 1e-5
  tie_word_embeddings: true

data:
  train_dataset: "train/train_data.txt"
  val_dataset: "evals/val_data.txt"
  tokenizer_path: "tokenizer/tokenizer.json"
  max_seq_length: 2048
  batch_size: 4            # маленький батч для памяти
  num_workers: 2
  pin_memory: false         # отключите на Colab

training:
  num_epochs: 1             # одна эпоха для теста
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0
  warmup_steps: 100
  logging_steps: 10
  eval_steps: 100
  save_steps: 500
  save_path: "./checkpoints/mind_epoch_{epoch}.ckpt"
  resume_from: null

  optimizer:
    type: "AdamW"
    lr: 1e-4
    betas: [0.9, 0.95]
    weight_decay: 0.01
    eps: 1e-8

  scheduler:
    type: "cosine_with_warmup"
    min_lr: 1e-6
    total_steps: 10000

  distributed: false
  world_size: 1

hardware:
  device: "cuda"
  mixed_precision: "bf16"  # или "fp16"
  gradient_checkpointing: true

eval:
  metrics:
    - perplexity
    - loss

logging:
  wandb_project: "mind"
  wandb_entity: "your_username"
  tensorboard: false
"""

with open('./config/train.yaml', 'w') as f:
    f.write(config)
```

### Шаг 4: Создание скрипта обучения

Создайте файл `train.py` в корне проекта:

```python
# Сохраните этот код в train.py
import os
import sys
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm

sys.path.insert(0, '/content/Mind-4-demo-code')

from model.torch.model import Model, ModelConfig
from model.torch.weights import Checkpoint
from model.torch.utils import setup_torch_seed

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_length=2048):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer

        with open(file_path, 'r', encoding='utf-8') as f:
            self.texts = f.readlines()

        # Фильтруем пустые строки
        self.texts = [t.strip() for t in self.texts if t.strip()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Токенизуем текст
        tokens = self.tokenizer.encode(text)[:self.max_seq_length]

        # Паддируем до max_seq_length
        if len(tokens) < self.max_seq_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(tokens))

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels
        }

def train_epoch(model, train_loader, optimizer, device, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0.0

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(train_loader, desc="Training")):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = model(input_ids)  # (batch_size, seq_length, vocab_size)

        # Вычисляем loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )

        # Нормируем по accumulation_steps
        loss = loss / gradient_accumulation_steps
        loss.backward()

        total_loss += loss.item()

        # Делаем шаг оптимизатора каждые N шагов
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if (step + 1) % 10 == 0:
            print(f"Step {step+1}, Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)

def main():
    # Загружаем конфиг
    with open('./config/train.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)

    # Создаем папки
    Path('./checkpoints').mkdir(exist_ok=True)

    # Установка seed
    setup_torch_seed(42)

    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    # Создаем конфиг модели
    model_config = ModelConfig(
        hidden_size=config_dict['model']['hidden_size'],
        num_hidden_layers=config_dict['model']['num_hidden_layers'],
        num_attention_heads=config_dict['model']['num_attention_heads'],
        num_key_value_heads=config_dict['model']['num_key_value_heads'],
        vocab_size=config_dict['model']['vocab_size'],
        head_dim=config_dict['model']['head_dim'],
        intermediate_size=config_dict['model']['intermediate_size'],
        num_experts=config_dict['model']['num_experts'],
        experts_per_token=config_dict['model']['experts_per_token'],
        rope_theta=config_dict['model']['rope_theta'],
        sliding_window=config_dict['model']['sliding_window'],
        initial_context_length=config_dict['model']['initial_context_length'],
        rope_scaling_factor=config_dict['model']['rope_scaling_factor'],
        rope_ntk_alpha=config_dict['model']['rope_ntk_alpha'],
        rope_ntk_beta=config_dict['model']['rope_ntk_beta'],
        swiglu_limit=config_dict['model']['swiglu_limit'],
    )

    # Создаем модель
    model = Model(model_config, device=device, dtype=torch.bfloat16)
    model = model.to(device)

    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

    # Создаем оптимизатор
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config_dict['training']['optimizer']['lr'],
        betas=tuple(config_dict['training']['optimizer']['betas']),
        weight_decay=config_dict['training']['optimizer']['weight_decay'],
    )

    # Загружаем датасет (простой вариант - используем имеющиеся данные)
    from model.torch.utils import get_tokenizer
    tokenizer = get_tokenizer()

    train_dataset = TextDataset(
        config_dict['data']['train_dataset'],
        tokenizer,
        max_seq_length=config_dict['data']['max_seq_length']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_dict['data']['batch_size'],
        shuffle=True,
        num_workers=0,  # Colab не поддерживает многопроцессность хорошо
        pin_memory=False
    )

    # Обучение
    num_epochs = config_dict['training']['num_epochs']
    for epoch in range(num_epochs):
        print(f"\nЭпоха {epoch+1}/{num_epochs}")
        avg_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            gradient_accumulation_steps=config_dict['training']['gradient_accumulation_steps']
        )
        print(f"Средний loss: {avg_loss:.4f}")

        # Сохраняем чекпоинт
        checkpoint_path = f"./checkpoints/mind_epoch_{epoch+1}.ckpt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Чекпоинт сохранен: {checkpoint_path}")

if __name__ == "__main__":
    main()
```

### Шаг 5: Запуск обучения в Colab

```python
# Запустите это в ячейке Colab
os.chdir('/content/Mind-4-demo-code')
exec(open('train.py').read())
```

### Шаг 6: Мониторинг и сохранение результатов

```python
# Скачайте чекпоинты обратно
from google.colab import files
files.download('./checkpoints/mind_epoch_1.ckpt')

# Или сохраните на Google Drive
!cp ./checkpoints/*.ckpt /content/gdrive/My\ Drive/mind_checkpoints/
```

## Оптимизация для бесплатного Colab (T4 GPU, 15GB памяти)

### Рекомендуемые параметры модели:

- `hidden_size`: 768-1024 (вместо 2880)
- `num_hidden_layers`: 12-16 (вместо 36)
- `num_attention_heads`: 12-16 (вместо 64)
- `vocab_size`: 32000-50000
- `batch_size`: 2-4
- `max_seq_length`: 1024-2048
- `gradient_accumulation_steps`: 2-4

### Включите gradient checkpointing:

```python
model.gradient_checkpointing_enable()
```

## Советы для успешного обучения

1. **Данные**: Используйте качественные текстовые данные (статьи, документацию, код)
2. **Learning Rate**: Начните с 1e-4 и экспериментируйте
3. **Batch Size**: Уменьшайте, если получаете OOM ошибку
4. **Сохранение**: Регулярно сохраняйте чекпоинты и скачивайте их
5. **Мониторинг**: Следите за loss - он должен убывать
6. **Время**: Свободный Colab отключается через 12 часов

## Решение проблем

### Ошибка: "CUDA out of memory"
- Уменьшите `batch_size`
- Уменьшите `max_seq_length`
- Включите `gradient_checkpointing`
- Используйте меньшую модель

### Ошибка: "Module not found"
- Проверьте, что находитесь в правильной директории: `%cd Mind-4-demo-code`
- Добавьте путь: `sys.path.insert(0, '/content/Mind-4-demo-code')`

### Медленное обучение
- Проверьте, что используется GPU: `torch.cuda.is_available()`
- Уменьшите `num_workers` (на Colab оставляйте 0)
- Увеличьте `gradient_accumulation_steps`

## Пример полного ноутбука

```python
# Cell 1: Установка
!pip install torch -q
!git clone https://github.com/ReNothingg/Mind-4-demo-code.git
%cd Mind-4-demo-code

# Cell 2: Импорты
import torch
import os

# Cell 3: Проверка GPU
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Cell 4: Подготовка данных
# (скопируйте код из Шага 2)

# Cell 5: Обновление конфига
# (скопируйте код из Шага 3)

# Cell 6: Обучение
# (скопируйте код из Шага 4 и запустите Шаг 5)
```

Удачи.
