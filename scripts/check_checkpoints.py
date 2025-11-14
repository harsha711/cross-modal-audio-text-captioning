"""
Utility script to check model checkpoints and their configurations
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.model_configs import get_checkpoint_info, infer_config_from_state_dict

def check_checkpoints(checkpoint_dir='checkpoints'):
    """Check all checkpoints in directory"""
    checkpoint_path = project_root / checkpoint_dir

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_path}")
        return

    checkpoint_files = list(checkpoint_path.glob('*.pth'))

    if not checkpoint_files:
        print(f"❌ No checkpoint files found in {checkpoint_path}")
        return

    print(f"Found {len(checkpoint_files)} checkpoint(s) in {checkpoint_dir}/\n")
    print("="*80)

    for i, cp_file in enumerate(sorted(checkpoint_files), 1):
        print(f"\n{i}. {cp_file.name}")
        print("-"*80)

        try:
            checkpoint = torch.load(cp_file, map_location='cpu')

            # Check for config
            if 'config' in checkpoint:
                print("✓ Has config saved")
                config = checkpoint['config']
                print("\nConfiguration:")
                for key, value in config.items():
                    print(f"  {key:<20} {value}")
            else:
                print("⚠️  No config in checkpoint, inferring from state dict...")
                state_dict = checkpoint['model_state_dict']
                vocab_size = state_dict['embedding.weight'].shape[0]
                config = infer_config_from_state_dict(state_dict, vocab_size)
                print("\nInferred configuration:")
                for key, value in config.items():
                    print(f"  {key:<20} {value}")

            # Check training info
            if 'epoch' in checkpoint:
                print(f"\nTraining epoch: {checkpoint['epoch']}")
            if 'loss' in checkpoint:
                print(f"Loss: {checkpoint['loss']:.4f}")

            # Model size
            state_dict = checkpoint['model_state_dict']
            total_params = sum(p.numel() for p in state_dict.values())
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Model size: {total_params * 4 / (1024**2):.2f} MB (float32)")

        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")

        print()

    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Check model checkpoints')
    parser.add_argument('--dir', '-d', default='checkpoints',
                       help='Checkpoint directory (default: checkpoints)')

    args = parser.parse_args()

    check_checkpoints(args.dir)
