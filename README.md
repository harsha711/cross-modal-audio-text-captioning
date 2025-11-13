# Cross-Model Audio Text Captioning

A comprehensive audio captioning system implementing multiple model architectures (Baseline, Improved Baseline, Attention, and Transformer) for generating natural language descriptions of audio content.

## Features

- **4 Model Architectures**:
  - **Baseline Model**: Simple CNN encoder + LSTM decoder
  - **Improved Baseline**: Deeper CNN + BiLSTM encoder
  - **Attention Model**: Bahdanau attention mechanism
  - **Transformer Model**: Full transformer architecture

- **Complete Training Pipeline**: Automated training, validation, and evaluation
- **Comprehensive Evaluation**: Repetition rate, diversity metrics, caption quality
- **Easy-to-use API**: Simple interfaces for training and inference
- **Modular Design**: Clean separation of models, data, training, and evaluation

## Project Structure

```
audio_captioning/
│
├── notebooks/              # Jupyter notebooks for experiments
│   └── baseline_experiments.ipynb
│
├── src/                    # Main source code
│   ├── __init__.py        # Package initialization
│   ├── models.py          # All model architectures
│   ├── dataset.py         # Dataset classes and data loaders
│   ├── trainer.py         # Training logic
│   ├── evaluation.py      # Evaluation metrics
│   └── utils.py           # Utility functions
│
├── scripts/               # Training and evaluation scripts
│   ├── train_baseline.py
│   ├── train_attention.py
│   ├── train_transformer.py
│   └── evaluate_all.py
│
├── configs/               # Configuration files
│   ├── baseline_config.yaml
│   ├── attention_config.yaml
│   └── transformer_config.yaml
│
├── features/              # Mel spectrograms
│   ├── mel/              # Training/validation features
│   └── mel_eval/         # Evaluation features
│
├── checkpoints/           # Saved model checkpoints
├── results/              # Training results and plots
├── vocab.json            # Vocabulary file
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cross-model-audio-text-captioning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place mel spectrograms in `features/mel/` and `features/mel_eval/`
   - Create caption JSON files in `data/` directory
   - Build vocabulary using the utility function

## Quick Start

### Building Vocabulary

```python
from src.utils import build_vocab, save_vocab

vocab, word_freq = build_vocab(
    captions_files=['data/train_captions.json', 'data/val_captions.json'],
    min_word_freq=2
)
save_vocab(vocab, 'vocab.json')
```

### Training a Model

```python
from src.models import create_model
from src.dataset import create_dataloaders
from src.trainer import ModelTrainer
from src.utils import load_vocab, get_device

# Load vocabulary
vocab = load_vocab('vocab.json')
device = get_device()

# Create dataloaders
train_loader, val_loader, eval_dataset = create_dataloaders(
    train_captions='data/train_captions.json',
    val_captions='data/val_captions.json',
    eval_captions='data/eval_captions.json',
    train_features_dir='features/mel/',
    val_features_dir='features/mel/',
    eval_features_dir='features/mel_eval/',
    vocab=vocab,
    batch_size=32
)

# Create and train model
model = create_model('baseline', vocab_size=len(vocab))
trainer = ModelTrainer(model, vocab, device=device, model_name='baseline')

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    eval_dataset=eval_dataset,
    num_epochs=30,
    learning_rate=1e-3
)
```

### Using Training Scripts

Train individual models:
```bash
python scripts/train_baseline.py
python scripts/train_attention.py
python scripts/train_transformer.py
```

Evaluate all models:
```bash
python scripts/evaluate_all.py
```

### Generating Captions

```python
import torch
from src.models import create_model
from src.utils import load_vocab, decode_caption

# Load model
vocab = load_vocab('vocab.json')
model = create_model('baseline', vocab_size=len(vocab))
checkpoint = torch.load('checkpoints/best_baseline.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and prepare mel spectrogram
mel = torch.FloatTensor(mel_data).unsqueeze(0)  # (1, 1, 64, 3000)

# Generate caption
with torch.no_grad():
    caption_ids = model.generate(mel, max_len=30, sos_idx=vocab['<sos>'], eos_idx=vocab['<eos>'])

# Decode to text
idx_to_word = {v: k for k, v in vocab.items()}
caption = decode_caption(caption_ids[0], idx_to_word, vocab)
print(f"Generated caption: {caption}")
```

## Model Architectures

### 1. Baseline Model
- **Encoder**: 3-layer CNN (64→128→256 channels)
- **Decoder**: 2-layer LSTM
- **Parameters**: ~10M
- **Best for**: Quick prototyping, baseline comparisons

### 2. Improved Baseline Model
- **Encoder**: 4-layer CNN + BiLSTM for temporal modeling
- **Decoder**: 2-layer LSTM with audio context concatenation
- **Parameters**: ~15M
- **Best for**: Better audio understanding without attention

### 3. Attention Model
- **Encoder**: CNN + BiLSTM producing sequence of features
- **Attention**: Bahdanau (additive) attention mechanism
- **Decoder**: 2-layer LSTM with attention context
- **Parameters**: ~18M
- **Best for**: Focus on relevant audio segments

### 4. Transformer Model
- **Encoder**: 3-layer transformer encoder
- **Decoder**: 3-layer transformer decoder
- **Parameters**: ~25M
- **Best for**: State-of-the-art performance, parallel processing

## Evaluation Metrics

The system provides comprehensive evaluation:

- **Repetition Rate**: Measures caption repetitiveness
- **Vocabulary Diversity**: Unique words / total words ratio
- **Caption Length Statistics**: Mean, std, min, max lengths
- **Model Comparison**: Side-by-side comparison of all models

## Configuration

Each model has a YAML configuration file in `configs/`:

```yaml
model:
  name: baseline
  params:
    embed_dim: 256
    hidden_dim: 512

training:
  num_epochs: 30
  learning_rate: 1e-3
  batch_size: 32

data:
  train_captions: data/train_captions.json
  vocab_path: vocab.json
```

## API Reference

### Models
```python
from src.models import create_model

model = create_model(
    model_type='baseline',  # 'baseline', 'improved_baseline', 'attention', 'transformer'
    vocab_size=5000,
    **model_kwargs
)
```

### Training
```python
from src.trainer import ModelTrainer

trainer = ModelTrainer(model, vocab, device='cuda', model_name='baseline')
history = trainer.fit(train_loader, val_loader, eval_dataset, num_epochs=30)
```

### Evaluation
```python
from src.evaluation import evaluate_model, compare_models

# Single model
results, captions, refs = evaluate_model(model, eval_dataset, vocab)

# Multiple models
comparison = compare_models(models_dict, eval_dataset, vocab)
```

## Data Format

### Caption JSON Format
```json
[
  {
    "fname": "audio_001",
    "captions": [
      "a dog barking in the distance",
      "barking dog can be heard",
      "a canine is barking"
    ]
  }
]
```

### Mel Spectrogram Format
- Shape: `(1, 64, 3000)` - (channels, mel_bins, time_steps)
- Format: `.npy` files
- Naming: `{fname}.npy`

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:
```bibtex
@software{audio_captioning_2024,
  title={Cross-Model Audio Text Captioning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/audio-captioning}
}
```

## Acknowledgments

- Built with PyTorch
- Inspired by image captioning and audio processing research
- Thanks to the Clotho dataset creators

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]
