"""
Utility functions for audio captioning project
Includes vocabulary building, visualization, and helper functions
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import torch


def build_vocab(captions_files, min_word_freq=2, special_tokens=None):
    """
    Build vocabulary from caption files

    Args:
        captions_files: List of paths to caption JSON files
        min_word_freq: Minimum word frequency to include in vocabulary
        special_tokens: List of special tokens (default: ['<pad>', '<sos>', '<eos>', '<unk>'])

    Returns:
        vocab: Dictionary mapping words to indices
        word_freq: Counter object with word frequencies
    """
    if special_tokens is None:
        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']

    # Count word frequencies
    word_freq = Counter()

    for captions_file in captions_files:
        with open(captions_file, 'r') as f:
            data = json.load(f)

        for item in data:
            for caption in item['captions']:
                words = caption.lower().strip().split()
                word_freq.update(words)

    # Build vocabulary
    vocab = {token: idx for idx, token in enumerate(special_tokens)}

    # Add words that meet frequency threshold
    idx = len(special_tokens)
    for word, freq in word_freq.most_common():
        if freq >= min_word_freq:
            vocab[word] = idx
            idx += 1

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Total unique words: {len(word_freq)}")
    print(f"Words with freq >= {min_word_freq}: {len(vocab) - len(special_tokens)}")

    return vocab, word_freq


def save_vocab(vocab, path):
    """Save vocabulary to JSON file"""
    with open(path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Vocabulary saved to {path}")


def load_vocab(path):
    """Load vocabulary from JSON file"""
    with open(path, 'r') as f:
        vocab = json.load(f)
    print(f"Vocabulary loaded from {path} (size: {len(vocab)})")
    return vocab


def plot_training_history(history, save_path=None):
    """
    Plot training history

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'learning_rate'
        save_path: Path to save plot (None = display only)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning rate plot
    axes[1].plot(history['learning_rate'], color='green', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_model_comparison(results_dict, metric='val_loss', save_path=None):
    """
    Compare multiple models

    Args:
        results_dict: Dictionary mapping model names to their history
        metric: Metric to plot ('val_loss' or 'train_loss')
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))

    for model_name, history in results_dict.items():
        if metric in history:
            plt.plot(history[metric], label=model_name, linewidth=2, marker='o', markersize=4)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'Model Comparison: {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()


def plot_evaluation_metrics(comparison_results, save_path=None):
    """
    Plot evaluation metrics for multiple models

    Args:
        comparison_results: Dictionary from compare_models function
        save_path: Path to save plot
    """
    models = list(comparison_results.keys())
    metrics = ['avg_repetition_rate', 'vocabulary_diversity', 'mean_caption_length']
    metric_names = ['Repetition Rate', 'Vocabulary Diversity', 'Average Caption Length']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [comparison_results[model]['metrics'][metric] for model in models]
        axes[i].bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)])
        axes[i].set_title(name, fontsize=12, fontweight='bold')
        axes[i].set_ylabel(name, fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics plot saved to {save_path}")
    else:
        plt.show()


def count_parameters(model):
    """
    Count trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        total_params: Total number of parameters
        trainable_params: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return total_params, trainable_params


def set_seed(seed=42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_device(prefer_cuda=True):
    """
    Get available device

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        device: torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")

    return device


def save_config(config, path):
    """
    Save configuration to JSON file

    Args:
        config: Dictionary with configuration
        path: Path to save
    """
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {path}")


def load_config(path):
    """
    Load configuration from JSON file

    Args:
        path: Path to config file

    Returns:
        config: Configuration dictionary
    """
    with open(path, 'r') as f:
        config = json.load(f)
    print(f"Configuration loaded from {path}")
    return config


def decode_caption(token_ids, idx_to_word, vocab):
    """
    Decode token IDs to caption string

    Args:
        token_ids: List or tensor of token IDs
        idx_to_word: Dictionary mapping indices to words
        vocab: Vocabulary dictionary (for special tokens)

    Returns:
        caption: Decoded caption string
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.cpu().numpy()

    words = []
    for idx in token_ids:
        idx = int(idx)
        if idx == vocab.get('<eos>', 2):
            break
        if idx not in [vocab.get('<pad>', 0), vocab.get('<sos>', 1)]:
            words.append(idx_to_word.get(idx, '<unk>'))

    return ' '.join(words)


def create_directory_structure(base_path='.'):
    """
    Create the required directory structure for the project

    Args:
        base_path: Base path for the project
    """
    base_path = Path(base_path)

    directories = [
        'src',
        'notebooks',
        'scripts',
        'configs',
        'features/mel',
        'features/mel_eval',
        'checkpoints',
        'results'
    ]

    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)

    print("Directory structure created successfully!")


def print_model_summary(models_dict):
    """
    Print summary of multiple models

    Args:
        models_dict: Dictionary mapping model names to model instances
    """
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    print(f"{'Model Name':<25} {'Total Params':<20} {'Trainable Params':<20}")
    print("-"*80)

    for name, model in models_dict.items():
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:<25} {total:>19,} {trainable:>19,}")

    print("="*80)


if __name__ == "__main__":
    print("Utility functions ready!")
    print("\nAvailable functions:")
    print("  - build_vocab: Build vocabulary from caption files")
    print("  - save_vocab / load_vocab: Save/load vocabulary")
    print("  - plot_training_history: Plot training curves")
    print("  - plot_model_comparison: Compare multiple models")
    print("  - plot_evaluation_metrics: Plot evaluation metrics")
    print("  - count_parameters: Count model parameters")
    print("  - set_seed: Set random seed for reproducibility")
    print("  - get_device: Get available device (CPU/GPU)")
    print("  - decode_caption: Decode token IDs to text")
    print("  - create_directory_structure: Create project directories")
