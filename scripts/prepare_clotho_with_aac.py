"""
Prepare Clotho dataset using aac_datasets library
This will download and prepare the data automatically
"""

import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json
from tqdm import tqdm
import librosa

# Import aac_datasets
try:
    from aac_datasets import Clotho
except ImportError:
    print("Error: aac_datasets not installed")
    print("Please install: pip install aac-datasets")
    sys.exit(1)


def extract_mel_spectrogram(audio, sr=32000, n_mels=64, n_fft=2048, hop_length=512):
    """
    Extract mel spectrogram from audio

    Args:
        audio: Audio array
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length

    Returns:
        mel: Mel spectrogram (1, n_mels, time_steps)
    """
    # Extract mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Convert to log scale (dB)
    mel = librosa.power_to_db(mel, ref=np.max)

    # Add channel dimension
    mel = mel[np.newaxis, ...]  # (1, n_mels, time_steps)

    return mel


def prepare_clotho_split(dataset, output_mel_dir, output_json_path, split_name):
    """
    Process one split of Clotho dataset

    Args:
        dataset: Clotho dataset object from aac_datasets
        output_mel_dir: Directory to save mel spectrograms
        output_json_path: Path to save caption JSON
        split_name: Name of the split (train/val/test)
    """
    print(f"\n{'='*80}")
    print(f"Processing: {split_name.upper()}")
    print(f"{'='*80}")

    # Create output directory
    output_mel_dir.mkdir(parents=True, exist_ok=True)

    data = []
    processed = 0

    print(f"Total samples in {split_name}: {len(dataset)}")

    for idx in tqdm(range(len(dataset)), desc=f"Processing {split_name}"):
        try:
            item = dataset[idx]

            # Get audio and captions
            # aac_datasets returns: (audio, sample_rate, captions)
            audio = item['audio']
            sr = item['sr']
            captions = item['captions']
            fname = item.get('fname', f"{split_name}_{idx:05d}")

            # Resample if needed
            if sr != 32000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
                sr = 32000

            # Extract mel spectrogram
            mel = extract_mel_spectrogram(audio, sr=sr)

            # Save mel spectrogram
            mel_output_path = output_mel_dir / f"{fname}.npy"
            np.save(mel_output_path, mel)

            # Add to data
            data.append({
                'fname': fname,
                'captions': captions if isinstance(captions, list) else [captions]
            })

            processed += 1

        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            continue

    # Save caption JSON
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Processed: {processed} files")
    print(f"✓ Saved mel spectrograms to: {output_mel_dir}")
    print(f"✓ Saved captions to: {output_json_path}")

    return processed


def main():
    print("="*80)
    print("CLOTHO DATASET PREPARATION (using aac_datasets)")
    print("="*80)

    # Setup paths
    base_dir = Path('.')

    # Create clotho_data directory if it doesn't exist
    clotho_root = base_dir / 'clotho_data'
    clotho_root.mkdir(exist_ok=True)
    print(f"Using Clotho data directory: {clotho_root}")

    # Download and prepare Clotho dataset
    print("\nDownloading/Loading Clotho dataset...")
    print("This may take a while on first run...")

    try:
        # Load train split (development in Clotho)
        print("\nLoading training data...")
        print("  [This may take 5-15 minutes on first download...]")
        print("  [Downloading audio files and metadata from Zenodo...]")
        import time
        start_time = time.time()

        train_dataset = Clotho(
            root=str(clotho_root),
            subset='dev',  # development = training set
            download=True
        )

        elapsed = time.time() - start_time
        print(f"  ✓ Loaded in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

        # Load validation split
        print("\nLoading validation data...")
        val_dataset = Clotho(
            root=str(clotho_root),
            subset='val',
            download=True
        )

        # Load evaluation split
        print("\nLoading evaluation data...")
        eval_dataset = Clotho(
            root=str(clotho_root),
            subset='eval',
            download=True
        )

    except Exception as e:
        print(f"\nError loading Clotho dataset: {e}")
        print("\nMake sure aac_datasets is properly installed:")
        print("  pip install aac-datasets")
        return

    # Process each split
    splits = {
        'train': {
            'dataset': train_dataset,
            'mel_dir': base_dir / 'features' / 'mel',
            'json_path': base_dir / 'data' / 'train_captions.json'
        },
        'val': {
            'dataset': val_dataset,
            'mel_dir': base_dir / 'features' / 'mel',
            'json_path': base_dir / 'data' / 'val_captions.json'
        },
        'eval': {
            'dataset': eval_dataset,
            'mel_dir': base_dir / 'features' / 'mel_eval',
            'json_path': base_dir / 'data' / 'eval_captions.json'
        }
    }

    total_processed = 0

    for split_name, config in splits.items():
        processed = prepare_clotho_split(
            dataset=config['dataset'],
            output_mel_dir=config['mel_dir'],
            output_json_path=config['json_path'],
            split_name=split_name
        )
        total_processed += processed

    # Summary
    print("\n" + "="*80)
    print("PREPARATION COMPLETE")
    print("="*80)
    print(f"Total processed: {total_processed}")

    # Show data statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)

    for split_name, config in splits.items():
        if config['json_path'].exists():
            with open(config['json_path'], 'r') as f:
                data = json.load(f)
            print(f"\n{split_name.upper()}:")
            print(f"  - Samples: {len(data)}")
            if data:
                caption_lengths = [len(cap.split()) for item in data for cap in item['captions']]
                print(f"  - Total captions: {len(caption_lengths)}")
                print(f"  - Avg caption length: {np.mean(caption_lengths):.1f} words")
                print(f"  - Min/Max length: {np.min(caption_lengths)}/{np.max(caption_lengths)} words")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Build vocabulary:")
    print("   python scripts/build_vocabulary.py")
    print("\n2. Train models:")
    print("   python scripts/train_baseline.py")
    print("   python scripts/train_attention.py")
    print("   python scripts/train_transformer.py")
    print("\n3. Evaluate:")
    print("   python scripts/evaluate_all.py")


if __name__ == "__main__":
    main()
