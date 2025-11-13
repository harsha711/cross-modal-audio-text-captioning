"""
Evaluate and compare all trained models
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from src.models import create_model
from src.dataset import ClothoEvalDataset
from src.evaluation import compare_models, get_sample_predictions, print_sample_predictions
from src.utils import load_vocab, get_device, plot_evaluation_metrics


def main():
    # Get device
    device = get_device()

    # Load vocabulary
    print("\nLoading vocabulary...")
    vocab = load_vocab('vocab.json')

    # Load evaluation dataset
    print("\nLoading evaluation dataset...")
    eval_dataset = ClothoEvalDataset(
        captions_file='data/eval_captions.json',
        features_dir='features/mel_eval/',
        vocab=vocab
    )

    # Load trained models
    print("\nLoading trained models...")
    models_dict = {}

    model_names = ['baseline', 'improved_baseline', 'attention', 'transformer']

    for model_name in model_names:
        checkpoint_path = Path(f'checkpoints/best_{model_name}.pth')

        if checkpoint_path.exists():
            print(f"  Loading {model_name}...")

            # Create model
            model = create_model(model_name, vocab_size=len(vocab))

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            models_dict[model_name] = model
        else:
            print(f"  Checkpoint not found for {model_name}: {checkpoint_path}")

    if not models_dict:
        print("\nNo trained models found. Please train models first.")
        return

    # Compare all models
    print("\n" + "="*80)
    print("COMPARING ALL MODELS")
    print("="*80)

    comparison = compare_models(
        models_dict=models_dict,
        eval_dataset=eval_dataset,
        vocab=vocab,
        device=device,
        num_samples=100
    )

    # Save comparison results
    print("\nSaving comparison results...")
    results_to_save = {}
    for model_name, data in comparison.items():
        results_to_save[model_name] = data['metrics']

    with open('results/comparison_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)

    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_evaluation_metrics(comparison, save_path='results/comparison_plot.png')

    # Get sample predictions from each model
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)

    for model_name, model in models_dict.items():
        print(f"\n{model_name.upper()}:")
        print("-"*80)
        samples = get_sample_predictions(model, eval_dataset, vocab, device, num_samples=5)
        print_sample_predictions(samples, num_to_print=3)

    print("\nEvaluation complete!")
    print(f"Comparison results saved to: results/comparison_results.json")
    print(f"Comparison plot saved to: results/comparison_plot.png")


if __name__ == "__main__":
    main()
