# Готовый пример Google Colab ноутбука для обучения Mind-4

Скопируйте весь код этого файла в отдельные ячейки Google Colab (по одному `# ===` блоку на ячейку)

---

## ЯЧЕЙКА 1: Установка и проверка

```python
# Установка зависимостей
!pip install torch -q
!pip install pyyaml tqdm -q

# Проверка что установилось
import torch
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("GPU недоступна - обучение будет медленным!")
```

---

## ЯЧЕЙКА 2: Клонирование репозитория

```python
# Клонируем репозиторий
!git clone https://github.com/ReNothingg/Mind-4-demo-code.git 2>/dev/null

# Переходим в папку
import os
os.chdir('/content/Mind-4-demo-code')

# Проверяем что все скачалось
print("  Файлы в репозитории:")
for item in os.listdir('.')[:10]:
    print(f"   {item}")

# Добавляем путь для импортов
import sys
sys.path.insert(0, '/content/Mind-4-demo-code')
```

---

## ЯЧЕЙКА 3: Подготовка тренировочных данных

```python
# Создаем тренировочные данные
training_data = """
Artificial Intelligence is transforming industries.
Machine Learning enables computers to learn from data.
Neural Networks are inspired by biological brains.
Deep Learning uses multiple layers of abstraction.
Transformers have revolutionized NLP tasks.
Attention mechanisms help models focus on important parts.
Self-attention allows tokens to interact with each other.
RoPE provides positional information to transformers.
MoE allows dynamic routing to different experts.
Gradient descent optimizes model parameters.
Backpropagation computes gradients efficiently.
Optimization algorithms update weights iteratively.
Batch normalization stabilizes neural network training.
Dropout regularization prevents overfitting.
Residual connections enable deeper networks.
Layer normalization improves training stability.
Embeddings represent discrete tokens as vectors.
Tokenization breaks text into manageable pieces.
Context windows limit how much text models see.
Loss functions measure prediction errors.
Accuracy measures correct predictions.
Perplexity measures language model quality.
Cross-entropy loss is common for classification.
Learning rates control optimization step sizes.
Momentum helps escape local minima.
Adam optimizer adapts learning rates per parameter.
Warmup gradually increases learning rate.
Learning rate scheduling decreases rates over time.
Early stopping prevents training too long.
Validation splits evaluate model performance.
Generalization measures performance on unseen data.
Overfitting occurs when models memorize training data.
Underfitting means models are too simple.
Regularization reduces model complexity.
Data augmentation creates more training examples.
Preprocessing cleans and normalizes data.
Feature engineering creates useful representations.
Dimensionality reduction compresses high-dimensional data.
Clustering groups similar examples together.
Classification assigns examples to discrete categories.
Regression predicts continuous values.
Multi-task learning shares representations across tasks.
Transfer learning uses pre-trained models.
Fine-tuning adapts models to specific tasks.
Prompt engineering optimizes input queries.
Few-shot learning learns from few examples.
Zero-shot learning applies to unseen tasks.
Chain-of-thought improves reasoning capability.
Scaling laws predict performance with more compute.
Emergent abilities appear with scale.
Interpretability helps understand model decisions.
Alignment ensures models behave as intended.
Safety prevents harmful model behaviors.
Efficiency reduces computational requirements.
Inference generates predictions from trained models.
Sampling generates diverse predictions.
Greedy decoding selects highest probability tokens.
Beam search explores multiple hypotheses.
Temperature controls prediction randomness.
Top-k sampling limits vocabulary to k top choices.
Top-p nucleus sampling uses cumulative probability.
Repetition penalties discourage repeated outputs.
Length penalties discourage overly long outputs.
Diversity penalties encourage varied outputs.
Conditioning controls model behavior.
In-context learning from examples in prompts.
Instructions guide model behavior.
Role-playing makes models behave like characters.
Knowledge-grounded uses external information.
Retrieval-augmented uses search for context.
Multi-modal handles text and images.
Vision transformers apply transformers to images.
Multimodal fusion combines different modalities.
Cross-modal understanding connects modalities.
Zero-resource learning from unlabeled data.
Self-supervised learning from unlabeled data.
Contrastive learning maximizes similarity.
Metric learning learns similarity functions.
Siamese networks compare pairs of inputs.
Triplet loss encourages similarity for related items.
Domain adaptation transfers across domains.
Out-of-distribution detection identifies unfamiliar data.
Uncertainty quantification measures prediction confidence.
Bayesian deep learning uses probability distributions.
Ensemble methods combine multiple models.
Boosting trains models sequentially.
Bagging trains models on data subsets.
Stacking combines diverse model predictions.
Attention visualization shows what models focus on.
Saliency maps show important input regions.
Gradient-based explanations use input gradients.
Layer-wise relevance propagation traces predictions.
LIME explains with local approximations.
SHAP uses game theory for explanations.
Counterfactual explanations show minimal changes.
Causal inference reasons about cause and effect.
Confounding variables complicate inference.
Simpson's paradox shows importance of stratification.
Spurious correlations lack causal connection.
Confuse factors can mask true relationships.
Batch size affects convergence and memory.
Learning rate schedules adapt over training.
Weight initialization affects training dynamics.
Gradient clipping prevents exploding gradients.
Layer normalization normalizes activations.
Instance normalization normalizes per example.
Group normalization normalizes across channels.
Weight normalization reparameterizes weights.
Spectral normalization constrains weight matrices.
Dropout randomly disables activations.
DropConnect randomly disables connections.
Mixup blends training examples.
CutMix blends image regions.
Manifold mixup blends hidden representations.
Knowledge distillation transfers to smaller models.
Pruning removes unnecessary parameters.
Quantization reduces precision of values.
Sparsity focuses on important connections.
Lottery ticket hypothesis finds sparse subnetworks.
Neural architecture search finds optimal designs.
Automated machine learning automates ML pipelines.
Meta-learning learns to learn quickly.
Learning to learn adapts quickly.
Continual learning doesn't forget old tasks.
Catastrophic forgetting loses old knowledge.
Replay buffers store old data for review.
Elastic weight consolidation preserves important weights.
Progressive neural networks add new capacity.
Modular networks use specialized components.
Capsule networks model hierarchies.
Graph neural networks process graph data.
Message passing propagates information.
Graph attention uses attention on graphs.
Relational reasoning compares entities.
Scene graphs represent structured scenes.
Knowledge graphs store facts and relationships.
Entity linking connects mentions to entities.
Relation extraction finds relationships.
Coreference resolution links related mentions.
Semantic role labeling identifies argument roles.
Dependency parsing finds grammatical dependencies.
Constituency parsing finds syntactic trees.
Named entity recognition identifies entities.
Part-of-speech tagging labels word types.
Chunking identifies basic phrases.
Machine translation converts between languages.
Cross-lingual transfer shares knowledge across languages.
Multilingual models handle multiple languages.
Language identification detects spoken language.
Script identification detects writing system.
Code-switching handles multiple languages.
Transliteration converts between writing systems.
Speech recognition converts audio to text.
Speech synthesis converts text to audio.
Voice conversion changes speaker characteristics.
Speech enhancement improves audio quality.
Acoustic modeling maps audio to phones.
Language modeling predicts next tokens.
N-gram models use limited context.
Smoothing handles unseen n-grams.
Backoff uses shorter contexts gracefully.
Interpolation blends model predictions.
Cache language models use recent context.
Recency bias favors recent information.
Attention to discourse uses document structure.
Pragmatics considers speaker intentions.
Semantics considers word meanings.
Syntax considers grammatical structure.
Morphology considers word structure.
Phonology considers sound structure.
Corpus linguistics studies real language data.
Treebank provides annotated parse trees.
Lexicon stores word information.
Ontology encodes conceptual relationships.
Taxonomy arranges concepts hierarchically.
Thesaurus groups related words.
WordNet connects words semantically.
FrameNet maps semantic frames.
PropBank marks semantic predicates.
VerbNet groups verbs by behavior.
SenseNet disambiguates word meanings.
BabelNet maps meanings across languages.
Wikidata provides structured knowledge.
Freebase stored factual knowledge.
DBpedia extracts structured from Wikipedia.
NELL continuously learns facts.
Reasoning engines derive new facts.
Deductive reasoning uses logical rules.
Inductive reasoning generalizes from examples.
Abductive reasoning infers best explanation.
Commonsense reasoning uses world knowledge.
Temporal reasoning reasons about time.
Spatial reasoning reasons about space.
Planning reasons about actions.
Reinforcement learning learns from rewards.
Reward shaping guides learning.
Curriculum learning progresses in difficulty.
Active learning selects informative examples.
Human-in-the-loop incorporates human feedback.
Crowdsourcing collects data from many people.
Annotation guidelines standardize labeling.
Inter-annotator agreement measures consistency.
Cohen's kappa corrects for chance agreement.
Data quality affects model performance.
Measurement bias skews estimates.
Selection bias over-represents subgroups.
Reporting bias affects what is recorded.
Implicit bias creates unfairness.
Fairness means treating groups equally.
Bias-variance tradeoff balances error types.
Statistical significance tests hypothesis validity.
P-values measure evidence strength.
Confidence intervals bound parameters.
Bootstrapping estimates uncertainty.
Jackknifing estimates parameter variance.
Cross-validation estimates generalization.
Stratified sampling ensures representation.
Importance weighting adjusts for bias.
Reweighting corrects class imbalance.
Threshold tuning finds optimal decision boundary.
ROC curves show classifier tradeoffs.
AUC measures classification performance.
F1 score balances precision and recall.
Precision measures prediction accuracy.
Recall measures coverage of positive class.
Specificity measures coverage of negative class.
Sensitivity is same as recall.
Matthews correlation coefficient balances all classes.
Confusion matrix shows all prediction types.
True positives are correctly predicted positives.
False positives are incorrectly predicted positives.
True negatives are correctly predicted negatives.
False negatives are incorrectly predicted negatives.
Type I error is false positive rate.
Type II error is false negative rate.
Power is one minus type II error rate.
""" * 3  # Повторяем 3 раза чтобы было больше данных

# Создаем папку если нужно
os.makedirs('./train', exist_ok=True)

# Сохраняем данные
with open('./train/train_data.txt', 'w', encoding='utf-8') as f:
    f.write(training_data)

print(f"Создано {len(training_data):,} символов тренировочных данных")
print(f"Строк: {len(training_data.split(chr(10)))}")
print(f"Сохранено в: ./train/train_data.txt")

# Проверяем что файл создался
import os
size_mb = os.path.getsize('./train/train_data.txt') / 1024 / 1024
print(f"Размер файла: {size_mb:.2f} MB")
```

---

## ЯЧЕЙКА 4: Создание конфигурации

```python
import yaml

# Конфигурация для маленькой модели (подходит для T4)
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
        'gradient_accumulation_steps': 1,
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

# Создаем папку если нужно
os.makedirs('./config', exist_ok=True)

# Сохраняем конфигурацию
with open('./config/train_colab.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(" Конфигурация сохранена в: ./config/train_colab.yaml")
print("\nПараметры конфигурации:")
print(f"Model size: {config['model']['hidden_size']} hidden units")
print(f"Layers: {config['model']['num_hidden_layers']}")
print(f"Batch size: {config['data']['batch_size']}")
print(f"Max seq length: {config['data']['max_seq_length']}")
print(f"Learning rate: {config['training']['optimizer']['lr']}")
```

---

## ЯЧЕЙКА 5: Запуск обучения

```python
import subprocess
import os

os.chdir('/content/Mind-4-demo-code')

print("Запускаем обучение Mind-4...")
print("="*50)

# Запускаем скрипт обучения
result = subprocess.run([
    'python', 'colab_train.py',
    '--config', './config/train_colab.yaml',
    '--data', './train/train_data.txt',
], capture_output=False, text=True)

print("="*50)
print("Обучение завершено!")
```

---

## ЯЧЕЙКА 6: Проверка результатов

```python
import os
import torch

# Проверяем что обучилось
checkpoint_dir = './checkpoints'

if os.path.exists(checkpoint_dir):
    files_list = os.listdir(checkpoint_dir)
    print(f"  Найдено файлов: {len(files_list)}")

    for file in sorted(files_list):
        filepath = os.path.join(checkpoint_dir, file)
        size_mb = os.path.getsize(filepath) / 1024 / 1024
        print(f"   {file} ({size_mb:.1f} MB)")

        # Пытаемся загрузить чекпоинт и показать инфо
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                print(f"Model weights loaded, keys: {len(checkpoint['model_state_dict'])}")
            if 'epoch' in checkpoint:
                print(f"Epoch: {checkpoint['epoch']}")
        except Exception as e:
            print(f"Ошибка при загрузке: {e}")
else:
    print("Папка checkpoints не найдена")
```

---

## ЯЧЕЙКА 7: Скачивание модели

```python
from google.colab import files
import os

checkpoint_dir = './checkpoints'

if os.path.exists(checkpoint_dir):
    files_to_download = os.listdir(checkpoint_dir)

    if files_to_download:
        print(f"Скачиваем {len(files_to_download)} файлов...")

        for file in files_to_download[:3]:  # Скачиваем первые 3
            filepath = os.path.join(checkpoint_dir, file)
            size_mb = os.path.getsize(filepath) / 1024 / 1024

            if size_mb < 500:  # Только если меньше 500MB
                print(f"   Скачиваем {file} ({size_mb:.1f} MB)...")
                files.download(filepath)
            else:
                print(f"{file} слишком большой ({size_mb:.1f} MB), пропускаем")
    else:
        print("Нет файлов для скачивания")
else:
    print("Папка checkpoints не найдена")

print("Готово! Файлы загружены на компьютер")
```

---

## ЯЧЕЙКА 8: Опционально - Сохранение на Google Drive

```python
from google.colab import drive
import shutil
import os

# Монтируем Google Drive
drive.mount('/content/gdrive', force_remount=False)

# Создаем папку для бэкапов
backup_dir = '/content/gdrive/My Drive/Mind4_Backups'
os.makedirs(backup_dir, exist_ok=True)

# Копируем чекпоинты
checkpoint_dir = './checkpoints'
if os.path.exists(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        src = os.path.join(checkpoint_dir, file)
        dst = os.path.join(backup_dir, file)
        shutil.copy(src, dst)
        print(f"  Скопирован: {file}")

print(f"Все файлы сохранены на Google Drive: {backup_dir}")
```

---

## ДОПОЛНИТЕЛЬНО: Проверка памяти GPU

```python
import torch

def print_gpu_stats():
    if torch.cuda.is_available():
        print("GPU Statistics:")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")
    else:
        print("GPU not available")

print_gpu_stats()

# Очистить кэш если нужно
torch.cuda.empty_cache()
print("Cache cleared")
```

---

## ДОПОЛНИТЕЛЬНО: Сравнение конфигураций

```python
configs = {
    'T4 (Free)': {
        'hidden_size': 512,
        'batch_size': 2,
        'max_seq_length': 512,
        'estimated_memory': '~8 GB',
        'speed': '~1 batch/sec'
    },
    'P100 (Pro)': {
        'hidden_size': 768,
        'batch_size': 4,
        'max_seq_length': 1024,
        'estimated_memory': '~20 GB',
        'speed': '~0.5 batch/sec'
    },
    'A100 (Premium)': {
        'hidden_size': 1024,
        'batch_size': 8,
        'max_seq_length': 2048,
        'estimated_memory': '~40 GB',
        'speed': '~0.3 batch/sec'
    }
}

print("Рекомендуемые конфигурации для разных GPU:\n")
for gpu_name, config in configs.items():
    print(f"{gpu_name}:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
```

---

## Использование

1. Откройте https://colab.research.google.com
2. Создайте новый ноутбук
3. Скопируйте каждый код выше в отдельную ячейку (по одному блоку с `# ===`)
4. Запустите ячейки по порядку (Shift+Enter)
5. Ждите результатов!