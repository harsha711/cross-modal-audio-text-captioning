"""
Training Script for Audio Captioning Models
Handles all model levels with proper evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import numpy as np
from pathlib import Path


class ModelTrainer:
    """Unified trainer for all model types"""

    def __init__(self, model, vocab, device='cuda', model_name='baseline'):
        self.model = model.to(device)
        self.vocab = vocab
        self.device = device
        self.model_name = model_name
        self.idx_to_word = {v: k for k, v in vocab.items()}

        # Track metrics
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

    def train_epoch(self, train_loader, criterion, optimizer, clip_grad=5.0):
        """Single training epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Training {self.model_name}")
        for mel, cap in pbar:
            mel, cap = mel.to(self.device), cap.to(self.device)

            # Input: all tokens except last, Target: all tokens except first
            input_cap = cap[:, :-1]
            target_cap = cap[:, 1:]

            # Forward pass
            optimizer.zero_grad()
            logits = self.model(mel, input_cap)

            # Compute loss
            loss = criterion(
                logits.reshape(-1, len(self.vocab)),
                target_cap.reshape(-1)
            )

            # Backward pass
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / len(train_loader)

    def validate(self, val_loader, criterion):
        """Validation epoch"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for mel, cap in tqdm(val_loader, desc="Validating"):
                mel, cap = mel.to(self.device), cap.to(self.device)

                input_cap = cap[:, :-1]
                target_cap = cap[:, 1:]

                logits = self.model(mel, input_cap)
                loss = criterion(
                    logits.reshape(-1, len(self.vocab)),
                    target_cap.reshape(-1)
                )

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def generate_samples(self, eval_dataset, num_samples=5):
        """Generate captions for sample audios"""
        self.model.eval()
        results = []

        for i in range(min(num_samples, len(eval_dataset))):
            item = eval_dataset[i]

            # Get mel spectrogram
            mel = item['mel'].unsqueeze(0).to(self.device)

            # Generate
            with torch.no_grad():
                ids = self.model.generate(
                    mel,
                    max_len=30,
                    sos_idx=self.vocab['<sos>'],
                    eos_idx=self.vocab['<eos>']
                )

            # Decode
            words = []
            for idx in ids[0]:
                idx = idx.item()
                if idx == self.vocab['<eos>']:
                    break
                if idx not in [self.vocab['<pad>'], self.vocab['<sos>']]:
                    words.append(self.idx_to_word.get(idx, '<unk>'))

            generated = ' '.join(words)
            ground_truth = item['captions'][0]

            results.append({
                'generated': generated,
                'ground_truth': ground_truth
            })

        return results

    def fit(self, train_loader, val_loader, eval_dataset=None,
            num_epochs=30, learning_rate=5e-4, weight_decay=1e-4,
            patience=5, label_smoothing=0.1, save_dir='checkpoints'):
        """
        Full training loop with early stopping

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            eval_dataset: Evaluation dataset for generating samples
            num_epochs: Maximum number of epochs
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            label_smoothing: Label smoothing factor
            save_dir: Directory to save checkpoints
        """
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        # Setup
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.vocab['<pad>'],
            label_smoothing=label_smoothing
        )

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, verbose=True
        )

        # Training loop
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_loss = self.train_epoch(train_loader, criterion, optimizer)

            # Validate
            val_loss = self.validate(val_loader, criterion)

            # Update scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Track metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)

            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            print(f"LR:         {current_lr:.6f}")

            # Generate samples every 5 epochs
            if eval_dataset and (epoch + 1) % 5 == 0:
                print("\nSample Generations:")
                print("-" * 60)
                samples = self.generate_samples(eval_dataset, num_samples=3)
                for i, sample in enumerate(samples, 1):
                    print(f"{i}. Ground Truth: {sample['ground_truth']}")
                    print(f"   Generated:    {sample['generated']}")
                    print()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                self.save_model(save_dir / f'best_{self.model_name}.pth')
                print(f"✓ New best model saved!")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"\nEarly stopping after {epoch+1} epochs")
                    break

        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Validation Loss: {best_val_loss:.4f}")
        print("="*60)

        return self.history

    def save_model(self, path):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'history': self.history,
            'model_name': self.model_name
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Model loaded from {path}")
        return checkpoint


def train_single_model(model, vocab, train_loader, val_loader, eval_dataset,
                      model_name='baseline', device='cuda', **train_kwargs):
    """
    Train a single model with given hyperparameters

    Args:
        model: PyTorch model
        vocab: Vocabulary dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        eval_dataset: Evaluation dataset
        model_name: Name for saving/logging
        device: Device to train on
        **train_kwargs: Additional training arguments

    Returns:
        trainer: Trained ModelTrainer instance
    """
    trainer = ModelTrainer(
        model=model,
        vocab=vocab,
        device=device,
        model_name=model_name
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        eval_dataset=eval_dataset,
        **train_kwargs
    )

    return trainer


def train_all_models(models_config, vocab, train_loader, val_loader, eval_dataset, device='cuda'):
    """
    Train multiple models and compare results

    Args:
        models_config: Dictionary with model configurations
            {
                'model_name': {
                    'model': model_instance,
                    'epochs': 30,
                    'lr': 1e-3,
                    'weight_decay': 1e-5,
                    'label_smoothing': 0.0
                },
                ...
            }
        vocab: Vocabulary dictionary
        train_loader: Training data loader
        val_loader: Validation data loader
        eval_dataset: Evaluation dataset
        device: Device to train on

    Returns:
        results: Dictionary with training results for each model
    """
    results = {}

    for name, config in models_config.items():
        print(f"\n{'#'*60}")
        print(f"# Training: {name.upper()}")
        print(f"{'#'*60}\n")

        # Extract model and training params
        model = config.pop('model')

        # Create trainer
        trainer = ModelTrainer(
            model=model,
            vocab=vocab,
            device=device,
            model_name=name
        )

        # Train
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            eval_dataset=eval_dataset,
            **config
        )

        results[name] = {
            'trainer': trainer,
            'history': history,
            'best_val_loss': min(history['val_loss'])
        }

        # Save history
        with open(f'results_{name}.json', 'w') as f:
            json.dump({
                'history': history,
                'best_val_loss': results[name]['best_val_loss']
            }, f, indent=2)

    # Compare all models
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Model':<25} {'Best Val Loss':<15}")
    print("-"*60)

    for name, data in results.items():
        best_val = data['best_val_loss']
        print(f"{name:<25} {best_val:<15.4f}")

    return results


if __name__ == "__main__":
    print("ModelTrainer class ready!")
    print("\nExample usage:")
    print("""
    from src.models import create_model
    from src.trainer import ModelTrainer

    # Create model
    model = create_model('baseline', vocab_size=len(vocab))

    # Create trainer
    trainer = ModelTrainer(model, vocab, device='cuda', model_name='baseline')

    # Train
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        eval_dataset=eval_dataset,
        num_epochs=30,
        learning_rate=1e-3
    )
    """)
