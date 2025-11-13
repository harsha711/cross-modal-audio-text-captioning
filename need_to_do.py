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
from collections import defaultdict

# Import your models (from the previous artifact)
# from audio_caption_models import BaselineModel, ImprovedBaselineModel, AttentionModel, TransformerModel


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
            
            # Load and prepare mel
            mel = np.load(f"features/mel_eval/{item['fname']}.npy")
            mel = torch.FloatTensor(mel)
            
            # Pad/crop to 3000
            if mel.shape[2] < 3000:
                padding = torch.zeros(1, mel.shape[1], 3000 - mel.shape[2])
                mel = torch.cat([mel, padding], dim=2)
            else:
                mel = mel[:, :, :3000]
            
            mel = mel.unsqueeze(0).to(self.device)
            
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
            patience=5, label_smoothing=0.1):
        """
        Full training loop with early stopping
        """
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
                self.save_model(f'best_{self.model_name}.pth')
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
    
    def load_model(self, path):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def calculate_repetition_rate(caption):
    """Calculate how repetitive a caption is"""
    words = caption.split()
    if len(words) == 0:
        return 0
    unique_words = len(set(words))
    return 1 - (unique_words / len(words))


def evaluate_diversity(generated_captions):
    """Measure vocabulary diversity in generated captions"""
    all_words = []
    for cap in generated_captions:
        all_words.extend(cap.split())
    
    if len(all_words) == 0:
        return {'unique_words': 0, 'total_words': 0, 'diversity': 0}
    
    unique = len(set(all_words))
    total = len(all_words)
    
    return {
        'unique_words': unique,
        'total_words': total,
        'diversity': unique / total
    }


def evaluate_model(model, eval_dataset, vocab, device='cuda', num_samples=100):
    """
    Comprehensive model evaluation
    Returns metrics: repetition rate, diversity, etc.
    """
    model.eval()
    idx_to_word = {v: k for k, v in vocab.items()}
    
    generated_captions = []
    repetition_scores = []
    
    print(f"Evaluating on {num_samples} samples...")
    
    for i in tqdm(range(min(num_samples, len(eval_dataset)))):
        item = eval_dataset[i]
        
        # Load mel
        mel = np.load(f"features/mel_eval/{item['fname']}.npy")
        mel = torch.FloatTensor(mel)
        
        if mel.shape[2] < 3000:
            padding = torch.zeros(1, mel.shape[1], 3000 - mel.shape[2])
            mel = torch.cat([mel, padding], dim=2)
        else:
            mel = mel[:, :, :3000]
        
        mel = mel.unsqueeze(0).to(device)
        
        # Generate
        with torch.no_grad():
            ids = model.generate(
                mel,
                max_len=30,
                sos_idx=vocab['<sos>'],
                eos_idx=vocab['<eos>']
            )
        
        # Decode
        words = []
        for idx in ids[0]:
            idx = idx.item()
            if idx == vocab['<eos>']:
                break
            if idx not in [vocab['<pad>'], vocab['<sos>']]:
                words.append(idx_to_word.get(idx, '<unk>'))
        
        caption = ' '.join(words)
        generated_captions.append(caption)
        repetition_scores.append(calculate_repetition_rate(caption))
    
    # Calculate metrics
    diversity_metrics = evaluate_diversity(generated_captions)
    avg_repetition = np.mean(repetition_scores)
    
    results = {
        'num_samples': num_samples,
        'avg_repetition_rate': avg_repetition,
        'vocabulary_diversity': diversity_metrics['diversity'],
        'unique_words_used': diversity_metrics['unique_words'],
        'total_words_generated': diversity_metrics['total_words']
    }
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for key, value in results.items():
        print(f"{key:.<40} {value}")
    print("="*60)
    
    return results, generated_captions


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_all_models(train_loader, val_loader, eval_dataset, vocab, device='cuda'):
    """
    Train all model levels and compare results
    """
    from audio_caption_models import (
        BaselineModel, ImprovedBaselineModel, 
        AttentionModel, TransformerModel
    )
    
    results = {}
    
    # Configuration for each model
    configs = {
        'baseline': {
            'model': BaselineModel(len(vocab)),
            'epochs': 30,
            'lr': 1e-3,
            'weight_decay': 1e-5,
            'label_smoothing': 0.0
        },
        'improved_baseline': {
            'model': ImprovedBaselineModel(len(vocab)),
            'epochs': 30,
            'lr': 5e-4,
            'weight_decay': 1e-4,
            'label_smoothing': 0.1
        },
        'attention': {
            'model': AttentionModel(len(vocab)),
            'epochs': 35,
            'lr': 3e-4,
            'weight_decay': 1e-4,
            'label_smoothing': 0.1
        },
        'transformer': {
            'model': TransformerModel(len(vocab)),
            'epochs': 40,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'label_smoothing': 0.1
        }
    }
    
    for name, config in configs.items():
        print(f"\n{'#'*60}")
        print(f"# Training: {name.upper()}")
        print(f"{'#'*60}\n")
        
        # Create trainer
        trainer = ModelTrainer(
            model=config['model'],
            vocab=vocab,
            device=device,
            model_name=name
        )
        
        # Train
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            eval_dataset=eval_dataset,
            num_epochs=config['epochs'],
            learning_rate=config['lr'],
            weight_decay=config['weight_decay'],
            label_smoothing=config['label_smoothing']
        )
        
        # Evaluate
        print(f"\nFinal Evaluation for {name}:")
        eval_results, _ = evaluate_model(
            trainer.model, 
            eval_dataset, 
            vocab, 
            device, 
            num_samples=100
        )
        
        results[name] = {
            'history': history,
            'eval_metrics': eval_results
        }
        
        # Save results
        with open(f'results_{name}.json', 'w') as f:
            json.dump(results[name], f, indent=2)
    
    # Compare all models
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'Best Val Loss':<15} {'Repetition':<15} {'Diversity':<15}")
    print("-"*60)
    
    for name, data in results.items():
        best_val = min(data['history']['val_loss'])
        rep = data['eval_metrics']['avg_repetition_rate']
        div = data['eval_metrics']['vocabulary_diversity']
        print(f"{name:<20} {best_val:<15.4f} {rep:<15.4f} {div:<15.4f}")
    
    return results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage in your notebook:
    
    # 1. Load your data (you already have this)
    vocab = json.load(open("vocab.json"))
    # ... your train_loader, val_loader, eval_dataset ...
    
    # 2. Train a single model
    from audio_caption_models import BaselineModel
    
    model = BaselineModel(len(vocab))
    trainer = ModelTrainer(model, vocab, device='cuda', model_name='baseline')
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        eval_dataset=eval_dataset,
        num_epochs=30,
        learning_rate=1e-3
    )
    
    # 3. Evaluate
    results, captions = evaluate_model(
        trainer.model,
        eval_dataset,
        vocab,
        num_samples=100
    )
    
    # 4. Or train all models at once
    all_results = train_all_models(
        train_loader, val_loader, eval_dataset, vocab
    )
    """
    pass
