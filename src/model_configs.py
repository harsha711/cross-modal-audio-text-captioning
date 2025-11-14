"""
Model Configuration Helper
Stores and loads model configurations to ensure consistent model creation
"""
import json
import os.path as osp


# Default configurations for each model type
MODEL_CONFIGS = {
    'baseline_small': {
        'model_type': 'baseline',
        'vocab_size': None,  # Will be set from vocab
        'embed_dim': 128,
        'hidden_dim': 256,
        'num_layers': 1
    },
    'baseline': {
        'model_type': 'baseline',
        'vocab_size': None,
        'embed_dim': 256,
        'hidden_dim': 512,
        'num_layers': 2
    },
    'improved_baseline': {
        'model_type': 'improved_baseline',
        'vocab_size': None,
        'embed_dim': 256,
        'hidden_dim': 512,
        'num_layers': 2
    },
    'attention_small': {
        'model_type': 'attention',
        'vocab_size': None,
        'embed_dim': 256,
        'hidden_dim': 512,
        'num_layers': 2
    },
    'attention': {
        'model_type': 'attention',
        'vocab_size': None,
        'embed_dim': 256,
        'hidden_dim': 512,
        'num_layers': 2
    },
    'transformer_small': {
        'model_type': 'transformer',
        'vocab_size': None,
        'd_model': 256,
        'nhead': 4,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'dim_feedforward': 512,
        'dropout': 0.1
    },
    'transformer': {
        'model_type': 'transformer',
        'vocab_size': None,
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dim_feedforward': 2048,
        'dropout': 0.1
    }
}


def get_model_config(model_name, vocab_size):
    """
    Get model configuration by name

    Args:
        model_name: Name of the model (e.g., 'baseline_small', 'transformer_small')
        vocab_size: Size of vocabulary

    Returns:
        Configuration dictionary
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name].copy()
    config['vocab_size'] = vocab_size
    return config


def save_model_with_config(model, optimizer, epoch, loss, save_path, config):
    """
    Save model checkpoint with configuration

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
        config: Model configuration dictionary
    """
    import torch

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'epoch': epoch,
        'loss': loss,
        'config': config
    }

    torch.save(checkpoint, save_path)


def load_model_from_checkpoint(checkpoint_path, vocab_size, device='cpu'):
    """
    Load model from checkpoint with automatic configuration detection

    Args:
        checkpoint_path: Path to checkpoint file
        vocab_size: Vocabulary size
        device: Device to load model on

    Returns:
        model: Loaded model
        config: Model configuration used
        checkpoint: Full checkpoint dictionary
    """
    import torch
    from src.models import create_model

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try to get config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        config['vocab_size'] = vocab_size  # Update vocab size
        print(f"✓ Loaded config from checkpoint: {config}")
    else:
        # Infer config from state dict
        print("⚠️ No config in checkpoint, inferring from state dict...")
        config = infer_config_from_state_dict(
            checkpoint['model_state_dict'],
            vocab_size
        )
        print(f"✓ Inferred config: {config}")

    # Create model with correct config
    model_type = config.pop('model_type')
    model = create_model(model_type, **config)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Add model_type back to config for reference
    config['model_type'] = model_type

    return model, config, checkpoint


def infer_config_from_state_dict(state_dict, vocab_size):
    """
    Infer model configuration from state dict

    Args:
        state_dict: Model state dictionary
        vocab_size: Vocabulary size

    Returns:
        Configuration dictionary
    """
    config = {'vocab_size': vocab_size}

    # Check for transformer
    if 'transformer.encoder.layers.0.self_attn.in_proj_weight' in state_dict:
        # Transformer model
        config['model_type'] = 'transformer'

        # Get d_model from embedding
        config['d_model'] = state_dict['embedding.weight'].shape[1]

        # Count encoder layers
        encoder_layers = sum(1 for k in state_dict.keys()
                           if k.startswith('transformer.encoder.layers.') and
                           k.endswith('.self_attn.in_proj_weight'))
        config['num_encoder_layers'] = encoder_layers

        # Count decoder layers
        decoder_layers = sum(1 for k in state_dict.keys()
                           if k.startswith('transformer.decoder.layers.') and
                           k.endswith('.self_attn.in_proj_weight'))
        config['num_decoder_layers'] = decoder_layers

        # Get nhead from attention projection
        d_model = config['d_model']
        in_proj_weight = state_dict['transformer.encoder.layers.0.self_attn.in_proj_weight']
        # in_proj_weight shape is (3*d_model, d_model) for multihead attention
        config['nhead'] = 8 if d_model == 512 else 4  # Heuristic

        # Get feedforward dim
        config['dim_feedforward'] = state_dict['transformer.encoder.layers.0.linear1.weight'].shape[0]
        config['dropout'] = 0.1

    # Check for attention model
    elif 'attention.W_encoder.weight' in state_dict:
        # Attention model
        config['model_type'] = 'attention'
        config['embed_dim'] = state_dict['embedding.weight'].shape[1]
        config['hidden_dim'] = state_dict['attention.W_encoder.weight'].shape[1]

        # Count decoder LSTM layers
        num_layers = sum(1 for k in state_dict.keys() if 'decoder_lstm.weight_ih_l' in k)
        config['num_layers'] = num_layers

    # Check for improved baseline (has temporal_lstm)
    elif 'temporal_lstm.weight_ih_l0' in state_dict:
        # Improved baseline
        config['model_type'] = 'improved_baseline'
        config['embed_dim'] = state_dict['embedding.weight'].shape[1]

        # Get hidden dim from decoder LSTM
        # decoder_lstm input is embed_dim + hidden_dim
        decoder_input_size = state_dict['decoder_lstm.weight_ih_l0'].shape[1]
        config['hidden_dim'] = decoder_input_size - config['embed_dim']

        # Count decoder LSTM layers
        num_layers = sum(1 for k in state_dict.keys() if 'decoder_lstm.weight_ih_l' in k)
        config['num_layers'] = num_layers

    else:
        # Baseline model
        config['model_type'] = 'baseline'
        config['embed_dim'] = state_dict['embedding.weight'].shape[1]
        config['hidden_dim'] = state_dict['encoder_projection.weight'].shape[0]

        # Count decoder LSTM layers
        num_layers = sum(1 for k in state_dict.keys() if 'decoder_lstm.weight_ih_l' in k)
        config['num_layers'] = num_layers

    return config


def get_checkpoint_info(checkpoint_path):
    """
    Get information about a checkpoint without loading the full model

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary with checkpoint information
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    info = {
        'has_config': 'config' in checkpoint,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'loss': checkpoint.get('loss', 'unknown')
    }

    if 'config' in checkpoint:
        info['config'] = checkpoint['config']
    else:
        # Try to infer basic info
        state = checkpoint['model_state_dict']
        info['vocab_size'] = state['embedding.weight'].shape[0]
        info['embed_dim'] = state['embedding.weight'].shape[1]

    return info


if __name__ == "__main__":
    # Print available configurations
    print("Available model configurations:")
    print("=" * 60)

    for name, config in MODEL_CONFIGS.items():
        print(f"\n{name}:")
        for key, value in config.items():
            if value is not None:
                print(f"  {key}: {value}")
