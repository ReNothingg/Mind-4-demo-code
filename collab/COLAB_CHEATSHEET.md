# Шпаргалка по обучению Mind-4 в Google Colab

## Полный рабочий пример (скопируйте весь код ниже)

### Ячейка 1: Подготовка
```python
# Установка зависимостей
!pip install torch -q
!pip install pyyaml tqdm -q

# Клонирование репозитория
!git clone https://github.com/ReNothingg/Mind-4-demo-code.git 2>/dev/null
%cd Mind-4-demo-code

# Проверка GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Ячейка 2: Создание данных
```python
# Создаем тестовые данные для быстрого теста
training_text = """
Artificial Intelligence is transforming the world.
Machine Learning models learn from data patterns.
Neural networks are inspired by biological neurons.
Transformers have revolutionized natural language processing.
Deep learning requires significant computational resources.
GPUs accelerate training of neural networks.
Python is the preferred language for machine learning.
Data preprocessing is crucial for model success.
Feature engineering improves model performance.
Regularization prevents overfitting in models.
Batch normalization stabilizes training.
Dropout adds robustness to neural networks.
Attention mechanisms improve model focus.
Self-attention allows models to weigh importance.
Positional encoding provides sequence information.
Back propagation computes gradients efficiently.
Optimization algorithms adjust model parameters.
Learning rates affect convergence speed.
Momentum helps escape local minima.
Convergence occurs when loss stops decreasing.
""" * 100  # Повторяем 100 раз

# Сохраняем данные
with open('./train/train_data.txt', 'w') as f:
    f.write(training_text)

print(f" Создано {len(training_text)} символов тренировочных данных")
print(f" Файл сохранен в: ./train/train_data.txt")
```

### Ячейка 3: Конфигурация для T4
```python
import yaml

# Конфигурация для маленькой модели (T4 GPU)
config = {
    'model': {
        'hidden_size': 512,
        'num_hidden_layers': 8,
        'num_attention_heads': 8,
        'num_key_value_heads': 2,
        'vocab_size': 50000,
        'head_dim': 64,
        'intermediate_size': 1024,
        'num_experts': 4,
        'experts_per_token': 2,
        'rope_theta': 10000.0,
        'sliding_window': 1024,
        'initial_context_length': 1024,
        'rope_scaling_factor': 1.0,
        'rope_ntk_alpha': 1.0,
        'rope_ntk_beta': 32.0,
        'swiglu_limit': 1.0,
    },
    'data': {
        'train_dataset': './train/train_data.txt',
        'val_dataset': None,
        'max_seq_length': 512,
        'batch_size': 2,
        'num_workers': 0,
        'pin_memory': False,
    },
    'training': {
        'num_epochs': 1,
        'gradient_accumulation_steps': 2,
        'max_grad_norm': 1.0,
        'warmup_steps': 50,
        'logging_steps': 10,
        'eval_steps': 100,
        'save_steps': 100,
        'save_path': './checkpoints/mind_epoch_{epoch}.pt',
        'optimizer': {
            'type': 'AdamW',
            'lr': 1e-4,
            'betas': [0.9, 0.95],
            'weight_decay': 0.01,
            'eps': 1e-8,
        },
    },
    'hardware': {
        'device': 'cuda',
        'mixed_precision': 'fp16',
        'gradient_checkpointing': True,
    },
}

# Сохраняем конфигурацию
with open('./config/train_colab.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print("Конфигурация сохранена в config/train_colab.yaml")
```

### Ячейка 4: Запуск обучения
```python
import subprocess
import os

os.chdir('/content/Mind-4-demo-code')

# Запускаем обучение
result = subprocess.run([
    'python', 'colab_train.py',
    '--config', './config/train_colab.yaml',
    '--data', './train/train_data.txt',
    '--max-samples', '1000'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
```

### Ячейка 5: Скачивание результатов
```python
from google.colab import files
import os

# Список всех файлов в checkpoints
checkpoint_dir = './checkpoints'
if os.path.exists(checkpoint_dir):
    files_list = os.listdir(checkpoint_dir)
    print(f"Найдено файлов: {len(files_list)}")
    for file in files_list[:5]:  # Показываем первые 5
        filepath = os.path.join(checkpoint_dir, file)
        size = os.path.getsize(filepath) / 1024 / 1024
        print(f"   - {file} ({size:.1f} MB)")
        files.download(filepath)
else:
    print("Папка checkpoints не найдена")
```

---

## Различные конфигурации

### Для тестирования (очень маленькая, 2 минуты)
```yaml
model:
  hidden_size: 256
  num_hidden_layers: 4
  num_attention_heads: 4
data:
  batch_size: 1
  max_seq_length: 256
training:
  num_epochs: 1
```

### Для T4 (стандартный Colab)
```yaml
model:
  hidden_size: 512
  num_hidden_layers: 8
  num_attention_heads: 8
data:
  batch_size: 2
  max_seq_length: 512
training:
  num_epochs: 1
```

### Для P100 (Colab Pro)
```yaml
model:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
data:
  batch_size: 4
  max_seq_length: 1024
training:
  num_epochs: 2
```

### Для A100 (Colab Premium)
```yaml
model:
  hidden_size: 1024
  num_hidden_layers: 16
  num_attention_heads: 16
data:
  batch_size: 8
  max_seq_length: 2048
training:
  num_epochs: 3
```

---

## Загрузка данных из разных источников

### С Google Drive
```python
from google.colab import drive
drive.mount('/content/gdrive')

# Копируем файл
import shutil
shutil.copy('/content/gdrive/My Drive/my_data.txt',
            './train/train_data.txt')
```

### С локального компьютера (drag-drop)
```python
from google.colab import files
uploaded = files.upload()

import shutil
for filename in uploaded.keys():
    shutil.copy(filename, './train/train_data.txt')
```

### Из открытого источника
```python
import urllib.request

url = 'https://example.com/training_data.txt'
urllib.request.urlretrieve(url, './train/train_data.txt')
```

### Создание случайных данных
```python
import random
words = ['machine', 'learning', 'neural', 'network', 'data', 'model', 'training', 'inference']

data = ' '.join(random.choices(words, k=10000))
with open('./train/train_data.txt', 'w') as f:
    f.write(data)
```

---

## Мониторинг и отладка

### Проверка GPU
```python
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.mem_get_info()[1] / 1e9:.1f} GB total")
print(f"Memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
```

### Проверка памяти во время обучения
```python
# Добавьте в ячейку обучения:
import torch
print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
```

### Очистка памяти
```python
import torch
torch.cuda.empty_cache()
print("Memory cleared")
```

---

## Если что-то пошло не так

### "ModuleNotFoundError: No module named 'mind'"
```python
# Убедитесь что находитесь в правильной папке:
%cd /content/Mind-4-demo-code
import sys
sys.path.insert(0, '/content/Mind-4-demo-code')
```

### "CUDA out of memory"
```python
# Уменьшите параметры:
# 1. batch_size с 4 на 2
# 2. max_seq_length с 1024 на 512
# 3. hidden_size с 768 на 512
```

### "RuntimeError: Expected all tensors to be on the same device"
```python
# Убедитесь что модель на GPU:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Обучение очень медленное
```python
# Проверьте:
print(f"GPU: {torch.cuda.is_available()}")  # Должен быть True
print(torch.cuda.get_device_name(0))        # Должен показать GPU
```

---

## Экспорт и использование модели

### Сохранить модель
```python
import torch
torch.save(model.state_dict(), './my_model.pt')
```

### Загрузить модель
```python
import torch
from model.torch.model import Model, ModelConfig

# Создаем модель
model = Model(config, device='cuda')

# Загружаем веса
model.load_state_dict(torch.load('./my_model.pt'))
```

### Скачать на компьютер
```python
from google.colab import files
files.download('./my_model.pt')
```

### Загрузить на Google Drive для резервной копии
```python
!cp ./my_model.pt /content/gdrive/My\ Drive/Mind4_Backups/
```

---

## Советы и трюки

### Сохранение дополнительной памяти
```python
# Включите gradient checkpointing
model.gradient_checkpointing_enable()

# Используйте mixed precision
from torch.cuda.amp import autocast
with autocast():
    output = model(input)
```

### Ускорение обучения
```python
# Используйте больший batch size
batch_size = 8

# Увеличьте num_workers (но на Colab оставляйте 0)
num_workers = 0

# Используйте pin_memory=False на Colab
pin_memory = False
```

### Лучший learning rate
```python
# Экспериментируйте:
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4]
# Начните с 1e-4 и регулируйте в зависимости от результатов
```

---

## Полезные команды

```python
# Посмотреть параметры модели
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Посмотреть количество параметров
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Посмотреть размер файла
import os
size_mb = os.path.getsize('./my_model.pt') / 1024 / 1024
print(f"Model size: {size_mb:.1f} MB")

# Посмотреть содержимое папки
import os
print(os.listdir('./checkpoints'))
```

---

## Успешно запустил?

Если вы дошли сюда - поздравляем! Вы сделали больше чем большая часть населения мира!

### Следующие шаги:

1. **Экспериментируйте с параметрами** - попробуйте разные `hidden_size`, `learning_rate`
2. **Используйте больше данных** - качество данных = качество модели
3. **Тонкая настройка** - когда базовая модель работает хорошо
4. **Деплой** - загрузите модель на компьютер или в облако

---

**Вопросы? Проблемы?** → Смотрите `GOOGLE_COLAB_TRAINING_GUIDE.md`

**Быстро начать?** → Смотрите `COLAB_QUICK_START.md`

**Полный гайд?** → Смотрите `START_HERE.md`