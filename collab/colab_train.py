#!/usr/bin/env python3
"""
Простой скрипт для обучения Mind-4 модели в Google Colab
Использование: python colab_train.py --config config/train.yaml --data train/train_data.txt
"""

import os
import sys
import yaml
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from model.torch.model import Model, ModelConfig
except ImportError:
    logger.error("Не удалось импортировать Mind-4. Убедитесь, что находитесь в директории Mind-4-demo-code")
    sys.exit(1)


class SimpleTextDataset(Dataset):
    """Простой датасет для текстовых данных"""

    def __init__(self, file_path: str, max_seq_length: int = 2048, max_samples: Optional[int] = None):
        self.max_seq_length = max_seq_length

        # Читаем все данные
        with open(file_path, 'r', encoding='utf-8') as f:
            self.texts = f.readlines()

        self.texts = [t.strip() for t in self.texts if t.strip()]

        if max_samples:
            self.texts = self.texts[:max_samples]

        logger.info(f"Загружено {len(self.texts)} примеров из {file_path}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Простая токенизация (используем ord для символов)
        tokens = [ord(c) % 50000 for c in text][:self.max_seq_length]

        # Паддируем
        if len(tokens) < self.max_seq_length:
            tokens = tokens + [0] * (self.max_seq_length - len(tokens))

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels
        }


def load_config(config_path: str) -> dict:
    """Загружает конфигурацию из YAML файла"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, save_path: str):
    """Сохраняет чекпоинт модели"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Чекпоинт сохранен: {save_path}")


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   checkpoint_path: str, device: torch.device):
    """Загружает чекпоинт модели"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    logger.info(f"Чекпоинт загружен: {checkpoint_path}, эпоха: {epoch}")
    return epoch


def train_epoch(model: torch.nn.Module, train_loader: DataLoader,
               optimizer: torch.optim.Optimizer, device: torch.device,
               gradient_accumulation_steps: int = 1) -> float:
    """Обучает одну эпоху"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc="Обучение")
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(input_ids)

            # Loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0
            )
            loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()
        total_loss += loss.item()
        num_batches += 1

        # Optimizer step
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        pbar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model: torch.nn.Module, val_loader: DataLoader,
            device: torch.device) -> float:
    """Валидация модели"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Валидация"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=0
            )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Обучение Mind-4 в Google Colab")
    parser.add_argument('--config', type=str, default='config/train.yaml',
                       help='Путь к конфигурации')
    parser.add_argument('--data', type=str, default='train/train_data.txt',
                       help='Путь к файлу с обучающими данными')
    parser.add_argument('--val-data', type=str, default=None,
                       help='Путь к файлу с валидационными данными')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Путь для загрузки чекпоинта')
    parser.add_argument('--resume', action='store_true',
                       help='Продолжить обучение из чекпоинта')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Максимальное количество примеров (для теста)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Тестовый запуск (1 батч)')

    args = parser.parse_args()

    # Проверяем конфиг файл
    if not os.path.exists(args.config):
        logger.error(f"Конфиг-файл не найден: {args.config}")
        sys.exit(1)

    # Проверяем файл данных
    if not os.path.exists(args.data):
        logger.error(f"Файл данных не найден: {args.data}")
        sys.exit(1)

    config = load_config(args.config)

    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используем устройство: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Создаем папки
    Path(config['training']['save_path'].split('{')[0]).mkdir(exist_ok=True, parents=True)

    # Конфиг модели
    model_config = ModelConfig(
        hidden_size=config['model']['hidden_size'],
        num_hidden_layers=config['model']['num_hidden_layers'],
        num_attention_heads=config['model']['num_attention_heads'],
        num_key_value_heads=config['model']['num_key_value_heads'],
        vocab_size=config['model']['vocab_size'],
        head_dim=config['model']['head_dim'],
        intermediate_size=config['model']['intermediate_size'],
        num_experts=config['model']['num_experts'],
        experts_per_token=config['model']['experts_per_token'],
        rope_theta=config['model']['rope_theta'],
        sliding_window=config['model']['sliding_window'],
        initial_context_length=config['model']['initial_context_length'],
        rope_scaling_factor=config['model']['rope_scaling_factor'],
        rope_ntk_alpha=config['model']['rope_ntk_alpha'],
        rope_ntk_beta=config['model']['rope_ntk_beta'],
        swiglu_limit=config['model']['swiglu_limit'],
    )

    # Создаем модель
    logger.info("Создаем модель...")
    model = Model(model_config, device=device, dtype=torch.float16)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Количество параметров: {total_params:,}")

    # Оптимизатор
    optimizer_config = config['training']['optimizer']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config['lr'],
        betas=tuple(optimizer_config['betas']),
        weight_decay=optimizer_config['weight_decay'],
        eps=optimizer_config['eps'],
    )

    # Загружаем чекпоинт если нужно
    start_epoch = 0
    if args.resume and args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint, device)
        start_epoch += 1

    # Датасеты
    logger.info("Загружаем датасеты...")
    train_dataset = SimpleTextDataset(
        args.data,
        max_seq_length=config['data']['max_seq_length'],
        max_samples=args.max_samples
    )

    val_dataset = None
    if args.val_data and os.path.exists(args.val_data):
        val_dataset = SimpleTextDataset(
            args.val_data,
            max_seq_length=config['data']['max_seq_length'],
            max_samples=args.max_samples // 4 if args.max_samples else None
        )

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    # Обучение
    num_epochs = config['training']['num_epochs']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']

    logger.info(f"Начинаем обучение на {num_epochs} эпох...")

    best_val_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Эпоха {epoch+1}/{num_epochs}")
        logger.info(f"{'='*50}")

        # Обучение
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        logger.info(f"Train Loss: {train_loss:.4f}")

        # Валидация
        if val_loader:
            val_loss = validate(model, val_loader, device)
            logger.info(f"Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"Новый лучший результат! (loss: {val_loss:.4f})")

        # Сохранение чекпоинта
        if (epoch + 1) % config['training']['save_steps'] == 0:
            save_path = config['training']['save_path'].format(epoch=epoch+1)
            save_checkpoint(model, optimizer, epoch, save_path)

        if args.dry_run:
            logger.info("Dry-run завершен, выходим...")
            break

    logger.info("\nОбучение завершено!")

    # Финальное сохранение
    final_path = f"./checkpoints/mind_final.ckpt"
    save_checkpoint(model, optimizer, num_epochs - 1, final_path)
    logger.info(f"Финальная модель сохранена: {final_path}")


if __name__ == "__main__":
    main()
