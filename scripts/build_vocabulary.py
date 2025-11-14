"""
Build vocabulary from Clotho caption files
"""

import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.utils import build_vocab, save_vocab


def main():
    print("="*80)
    print("VOCABULARY BUILDER")
    print("="*80)

    # Check if caption files exist
    caption_files = [
        'data/train_captions.json',
        'data/val_captions.json'
    ]

    missing_files = [f for f in caption_files if not Path(f).exists()]

    if missing_files:
        print("\nError: Missing caption files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run: python scripts/prepare_clotho_with_aac.py first")
        return

    # Build vocabulary
    print("\nBuilding vocabulary from caption files...")
    vocab, word_freq = build_vocab(
        captions_files=caption_files,
        min_word_freq=2,
        special_tokens=['<pad>', '<sos>', '<eos>', '<unk>']
    )

    # Save vocabulary
    save_vocab(vocab, 'vocab.json')

    # Show statistics
    print("\n" + "="*80)
    print("VOCABULARY STATISTICS")
    print("="*80)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Total unique words: {len(word_freq)}")
    print(f"\nSpecial tokens:")
    print(f"  <pad>: {vocab['<pad>']}")
    print(f"  <sos>: {vocab['<sos>']}")
    print(f"  <eos>: {vocab['<eos>']}")
    print(f"  <unk>: {vocab['<unk>']}")
    print(f"\nMost common words:")
    for word, count in word_freq.most_common(20):
        print(f"  {word:<15} {count:>5}")

    print("\n✓ Vocabulary saved to: vocab.json")
    print("\nReady to train models!")


if __name__ == "__main__":
    main()
