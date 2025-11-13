"""
Dataset class for audio captioning
Handles loading mel spectrograms and captions
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path


class ClothoDataset(Dataset):
    """
    Dataset for Clotho audio captioning

    Args:
        captions_file: Path to JSON file with captions
        features_dir: Directory containing mel spectrograms (.npy files)
        vocab: Vocabulary dictionary mapping words to indices
        max_length: Maximum caption length (longer captions will be truncated)
        mel_length: Fixed length for mel spectrograms (3000 time steps)
    """

    def __init__(self, captions_file, features_dir, vocab, max_length=30, mel_length=3000):
        self.features_dir = Path(features_dir)
        self.vocab = vocab
        self.max_length = max_length
        self.mel_length = mel_length

        # Load captions
        with open(captions_file, 'r') as f:
            self.data = json.load(f)

        # Filter out samples without corresponding mel files
        self.data = [item for item in self.data
                     if (self.features_dir / f"{item['fname']}.npy").exists()]

        print(f"Loaded {len(self.data)} samples from {captions_file}")

    def __len__(self):
        return len(self.data)

    def text_to_sequence(self, text):
        """
        Convert text to sequence of token indices

        Args:
            text: String of words

        Returns:
            List of token indices
        """
        words = text.lower().strip().split()
        sequence = [self.vocab.get('<sos>', 1)]  # Start token

        for word in words[:self.max_length - 2]:  # Leave room for <sos> and <eos>
            sequence.append(self.vocab.get(word, self.vocab.get('<unk>', 3)))

        sequence.append(self.vocab.get('<eos>', 2))  # End token

        return sequence

    def pad_sequence(self, sequence):
        """
        Pad sequence to max_length

        Args:
            sequence: List of token indices

        Returns:
            Padded tensor of shape (max_length,)
        """
        if len(sequence) < self.max_length:
            sequence = sequence + [self.vocab.get('<pad>', 0)] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]

        return torch.LongTensor(sequence)

    def load_mel(self, fname):
        """
        Load and process mel spectrogram

        Args:
            fname: Filename (without extension)

        Returns:
            Mel spectrogram tensor of shape (1, 64, mel_length)
        """
        mel_path = self.features_dir / f"{fname}.npy"
        mel = np.load(mel_path)

        # Expected shape: (1, 64, T) where T is time dimension
        mel = torch.FloatTensor(mel)

        # Handle different input shapes
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # Add channel dimension

        # Pad or crop to fixed length
        if mel.shape[2] < self.mel_length:
            padding = torch.zeros(mel.shape[0], mel.shape[1], self.mel_length - mel.shape[2])
            mel = torch.cat([mel, padding], dim=2)
        else:
            mel = mel[:, :, :self.mel_length]

        return mel

    def __getitem__(self, idx):
        """
        Get a single sample

        Returns:
            mel: Mel spectrogram tensor (1, 64, mel_length)
            caption: Padded caption tensor (max_length,)
        """
        item = self.data[idx]

        # Load mel spectrogram
        mel = self.load_mel(item['fname'])

        # Randomly select one caption (Clotho has multiple captions per audio)
        caption_text = np.random.choice(item['captions'])

        # Convert to sequence and pad
        caption_seq = self.text_to_sequence(caption_text)
        caption = self.pad_sequence(caption_seq)

        return mel, caption


class ClothoEvalDataset(Dataset):
    """
    Evaluation dataset that returns all captions for each audio
    Used for proper evaluation with multiple references
    """

    def __init__(self, captions_file, features_dir, vocab, mel_length=3000):
        self.features_dir = Path(features_dir)
        self.vocab = vocab
        self.mel_length = mel_length

        # Load captions
        with open(captions_file, 'r') as f:
            self.data = json.load(f)

        # Filter out samples without corresponding mel files
        self.data = [item for item in self.data
                     if (self.features_dir / f"{item['fname']}.npy").exists()]

        print(f"Loaded {len(self.data)} samples for evaluation")

    def __len__(self):
        return len(self.data)

    def load_mel(self, fname):
        """Load and process mel spectrogram"""
        mel_path = self.features_dir / f"{fname}.npy"
        mel = np.load(mel_path)
        mel = torch.FloatTensor(mel)

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        if mel.shape[2] < self.mel_length:
            padding = torch.zeros(mel.shape[0], mel.shape[1], self.mel_length - mel.shape[2])
            mel = torch.cat([mel, padding], dim=2)
        else:
            mel = mel[:, :, :self.mel_length]

        return mel

    def __getitem__(self, idx):
        """
        Get a single sample with all reference captions

        Returns:
            dict with:
                - fname: filename
                - mel: mel spectrogram
                - captions: list of all reference captions
        """
        item = self.data[idx]

        return {
            'fname': item['fname'],
            'mel': self.load_mel(item['fname']),
            'captions': item['captions']
        }


def create_dataloaders(train_captions, val_captions, eval_captions,
                       train_features_dir, val_features_dir, eval_features_dir,
                       vocab, batch_size=32, num_workers=4,
                       max_length=30, mel_length=3000):
    """
    Create train, validation, and evaluation dataloaders

    Args:
        train_captions: Path to training captions JSON
        val_captions: Path to validation captions JSON
        eval_captions: Path to evaluation captions JSON
        train_features_dir: Directory with training mel spectrograms
        val_features_dir: Directory with validation mel spectrograms
        eval_features_dir: Directory with evaluation mel spectrograms
        vocab: Vocabulary dictionary
        batch_size: Batch size for training/validation
        num_workers: Number of workers for data loading
        max_length: Maximum caption length
        mel_length: Fixed mel spectrogram length

    Returns:
        train_loader, val_loader, eval_dataset
    """

    # Create datasets
    train_dataset = ClothoDataset(
        train_captions, train_features_dir, vocab, max_length, mel_length
    )

    val_dataset = ClothoDataset(
        val_captions, val_features_dir, vocab, max_length, mel_length
    )

    eval_dataset = ClothoEvalDataset(
        eval_captions, eval_features_dir, vocab, mel_length
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, eval_dataset


if __name__ == "__main__":
    # Test dataset
    print("Testing dataset...")

    # Example vocab (you would load this from vocab.json)
    vocab = {
        '<pad>': 0,
        '<sos>': 1,
        '<eos>': 2,
        '<unk>': 3,
        'a': 4,
        'sound': 5,
        'of': 6,
        'water': 7,
    }

    # Example usage
    # dataset = ClothoDataset(
    #     captions_file='data/train_captions.json',
    #     features_dir='features/mel/',
    #     vocab=vocab
    # )

    # mel, caption = dataset[0]
    # print(f"Mel shape: {mel.shape}")
    # print(f"Caption shape: {caption.shape}")
    # print(f"Caption: {caption}")

    print("Dataset class ready!")
