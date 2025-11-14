"""
Training script for Baseline model
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
from src.models import create_model
from src.dataset import create_dataloaders
from src.trainer import ModelTrainer
from src.utils import load_vocab, set_seed, get_device


def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Get device
    device = get_device()

    # Load vocabulary
    print("\nLoading vocabulary...")
    vocab = load_vocab('vocab.json')

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, eval_dataset = create_dataloaders(
        train_captions='data/train_captions.json',
        val_captions='data/val_captions.json',
        eval_captions='data/eval_captions.json',
        train_features_dir='features/mel/',
        val_features_dir='features/mel/',
        eval_features_dir='features/mel_eval/',
        vocab=vocab,
        batch_size=32,
        num_workers=4
    )

    # Create baseline model
    print("\nCreating baseline model...")
    model = create_model('baseline', vocab_size=len(vocab))

    # Count parameters
    from src.utils import count_parameters
    count_parameters(model)

    # Create trainer
    print("\nInitializing trainer...")
    trainer = ModelTrainer(
        model=model,
        vocab=vocab,
        device=device,
        model_name='baseline'
    )

    # Train
    print("\nStarting training...")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        eval_dataset=eval_dataset,
        num_epochs=30,
        learning_rate=1e-3,
        weight_decay=1e-5,
        patience=5,
        label_smoothing=0.0,
        save_dir='checkpoints'
    )

    # Save history
    print("\nSaving training history...")
    with open('results/baseline_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Final evaluation
    print("\nFinal evaluation...")
    from src.evaluation import evaluate_model
    results, captions, refs = evaluate_model(
        trainer.model,
        eval_dataset,
        vocab,
        device=device,
        num_samples=100
    )

    # Save results
    with open('results/baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nTraining complete!")
    print(f"Best model saved to: checkpoints/best_baseline.pth")
    print(f"Results saved to: results/baseline_results.json")


if __name__ == "__main__":
    main()
