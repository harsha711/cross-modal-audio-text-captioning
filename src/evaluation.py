"""
Evaluation Metrics for Audio Captioning
Includes repetition rate, diversity metrics, and model evaluation
"""

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def calculate_repetition_rate(caption):
    """
    Calculate how repetitive a caption is

    Args:
        caption: String caption

    Returns:
        Repetition rate (0 = no repetition, 1 = all words repeated)
    """
    words = caption.split()
    if len(words) == 0:
        return 0
    unique_words = len(set(words))
    return 1 - (unique_words / len(words))


def evaluate_diversity(generated_captions):
    """
    Measure vocabulary diversity in generated captions

    Args:
        generated_captions: List of generated caption strings

    Returns:
        Dictionary with diversity metrics
    """
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


def calculate_caption_length_stats(generated_captions):
    """
    Calculate statistics about caption lengths

    Args:
        generated_captions: List of generated caption strings

    Returns:
        Dictionary with length statistics
    """
    lengths = [len(cap.split()) for cap in generated_captions]

    if len(lengths) == 0:
        return {
            'mean_length': 0,
            'std_length': 0,
            'min_length': 0,
            'max_length': 0
        }

    return {
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths)
    }


def evaluate_model(model, eval_dataset, vocab, device='cuda', num_samples=None):
    """
    Comprehensive model evaluation

    Args:
        model: Trained model
        eval_dataset: Evaluation dataset (should return dict with 'mel' and 'captions')
        vocab: Vocabulary dictionary
        device: Device to evaluate on
        num_samples: Number of samples to evaluate (None = all)

    Returns:
        results: Dictionary with evaluation metrics
        generated_captions: List of generated captions
    """
    model.eval()
    idx_to_word = {v: k for k, v in vocab.items()}

    generated_captions = []
    reference_captions = []
    repetition_scores = []

    num_samples = num_samples or len(eval_dataset)

    print(f"Evaluating on {num_samples} samples...")

    for i in tqdm(range(min(num_samples, len(eval_dataset)))):
        item = eval_dataset[i]

        # Get mel spectrogram
        mel = item['mel'].unsqueeze(0).to(device)

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
        reference_captions.append(item['captions'])
        repetition_scores.append(calculate_repetition_rate(caption))

    # Calculate metrics
    diversity_metrics = evaluate_diversity(generated_captions)
    length_stats = calculate_caption_length_stats(generated_captions)
    avg_repetition = np.mean(repetition_scores)

    results = {
        'num_samples': num_samples,
        'avg_repetition_rate': avg_repetition,
        'vocabulary_diversity': diversity_metrics['diversity'],
        'unique_words_used': diversity_metrics['unique_words'],
        'total_words_generated': diversity_metrics['total_words'],
        'mean_caption_length': length_stats['mean_length'],
        'std_caption_length': length_stats['std_length'],
        'min_caption_length': length_stats['min_length'],
        'max_caption_length': length_stats['max_length']
    }

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:.4f}")
        else:
            print(f"{key:.<40} {value}")
    print("="*60)

    return results, generated_captions, reference_captions


def compare_models(models_dict, eval_dataset, vocab, device='cuda', num_samples=None):
    """
    Compare multiple models on the same evaluation set

    Args:
        models_dict: Dictionary mapping model names to model instances
        eval_dataset: Evaluation dataset
        vocab: Vocabulary dictionary
        device: Device to evaluate on
        num_samples: Number of samples to evaluate

    Returns:
        comparison: Dictionary with results for each model
    """
    comparison = {}

    for model_name, model in models_dict.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()}")
        print(f"{'='*60}")

        results, captions, refs = evaluate_model(
            model, eval_dataset, vocab, device, num_samples
        )

        comparison[model_name] = {
            'metrics': results,
            'generated_captions': captions,
            'reference_captions': refs
        }

    # Print comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<20} {'Repetition':<15} {'Diversity':<15} {'Avg Length':<15}")
    print("-"*80)

    for name, data in comparison.items():
        metrics = data['metrics']
        print(f"{name:<20} {metrics['avg_repetition_rate']:<15.4f} "
              f"{metrics['vocabulary_diversity']:<15.4f} {metrics['mean_caption_length']:<15.2f}")

    print("="*80)

    return comparison


def get_sample_predictions(model, eval_dataset, vocab, device='cuda', num_samples=5):
    """
    Get sample predictions for qualitative analysis

    Args:
        model: Trained model
        eval_dataset: Evaluation dataset
        vocab: Vocabulary dictionary
        device: Device to use
        num_samples: Number of samples to generate

    Returns:
        List of dictionaries with generated and reference captions
    """
    model.eval()
    idx_to_word = {v: k for k, v in vocab.items()}
    samples = []

    for i in range(min(num_samples, len(eval_dataset))):
        item = eval_dataset[i]

        # Get mel
        mel = item['mel'].unsqueeze(0).to(device)

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

        samples.append({
            'fname': item['fname'],
            'generated': ' '.join(words),
            'references': item['captions']
        })

    return samples


def print_sample_predictions(samples, num_to_print=5):
    """
    Pretty print sample predictions

    Args:
        samples: List of sample dictionaries from get_sample_predictions
        num_to_print: Number of samples to print
    """
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)

    for i, sample in enumerate(samples[:num_to_print], 1):
        print(f"\nSample {i}: {sample['fname']}")
        print("-"*80)
        print(f"Generated:  {sample['generated']}")
        print(f"References:")
        for j, ref in enumerate(sample['references'], 1):
            print(f"  {j}. {ref}")
        print()

    print("="*80)


if __name__ == "__main__":
    print("Evaluation functions ready!")
    print("\nExample usage:")
    print("""
    from src.evaluation import evaluate_model, compare_models

    # Evaluate single model
    results, captions, refs = evaluate_model(
        model, eval_dataset, vocab, device='cuda', num_samples=100
    )

    # Compare multiple models
    models_dict = {
        'baseline': baseline_model,
        'attention': attention_model,
        'transformer': transformer_model
    }
    comparison = compare_models(models_dict, eval_dataset, vocab)

    # Get sample predictions
    samples = get_sample_predictions(model, eval_dataset, vocab, num_samples=10)
    print_sample_predictions(samples)
    """)
