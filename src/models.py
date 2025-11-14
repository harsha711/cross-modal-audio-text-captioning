"""
Audio Captioning Model Architectures
Implements 4 progressively complex models for audio captioning
"""
import sys, os
project_root = os.path.abspath("..")
sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# SAMPLING UTILITIES FOR DIVERSE GENERATION
# ============================================================================

def sample_with_temperature(logits, temperature=1.0, top_k=0, top_p=0.9):
    """
    Sample from logits with temperature, top-k, and top-p (nucleus) filtering

    Args:
        logits: (batch, vocab_size) raw model outputs
        temperature: Sampling temperature (higher = more random)
        top_k: If > 0, only sample from top k tokens
        top_p: If > 0, nucleus sampling - sample from smallest set with cumulative prob >= p

    Returns:
        next_token: (batch, 1) sampled token indices
    """
    # Apply temperature
    logits = logits / temperature

    # Apply top-k filtering
    if top_k > 0:
        top_k_values = torch.topk(logits, top_k)[0][..., -1, None]
        logits = torch.where(logits < top_k_values, torch.full_like(logits, -float('Inf')), logits)

    # Apply top-p (nucleus) filtering
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = 0
        # Scatter sorted tensors back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, -float('Inf'))

    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

    return next_token


# ============================================================================
# LEVEL 1: BASELINE MODEL
# ============================================================================

class BaselineModel(nn.Module):
    """
    Simple baseline: CNN encoder + LSTM decoder
    No attention mechanism

    IMPROVEMENTS:
    - Uses Average Pooling instead of Max Pooling to preserve temporal info
    - Keeps temporal dimension for better audio representation
    - Projects to sequence for LSTM processing
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()

        # Audio Encoder: CNN to process mel spectrograms with AVERAGE pooling
        self.encoder = nn.Sequential(
            # Input: (batch, 1, 64, 3000) - mel spectrogram
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),  # -> (batch, 64, 32, 1500) - AVGPOOL preserves info better

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),  # -> (batch, 128, 16, 750)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d((2, 2)),  # -> (batch, 256, 8, 375)
        )

        # Project to sequence representation (keep temporal dimension)
        # We'll use the time dimension (375) as sequence length
        self.encoder_projection = nn.Linear(256 * 8, hidden_dim)  # Per-timestep projection

        # Text Decoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.decoder_lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def encode_audio(self, mel):
        """Encode mel spectrogram to sequence representation preserving temporal info"""
        # mel: (batch, 1, 64, 3000)
        features = self.encoder(mel)  # (batch, 256, 8, 375)

        # Reshape to preserve temporal dimension
        batch_size = features.size(0)
        features = features.permute(0, 3, 1, 2)  # (batch, 375, 256, 8)
        features = features.reshape(batch_size, 375, -1)  # (batch, 375, 256*8)

        # Project each timestep
        audio_encoding = self.encoder_projection(features)  # (batch, 375, hidden_dim)

        # Global average pooling to get single vector for LSTM initialization
        audio_context = audio_encoding.mean(dim=1)  # (batch, hidden_dim)

        return audio_context

    def forward(self, mel, captions):
        """
        Forward pass for training
        mel: (batch, 1, 64, 3000)
        captions: (batch, seq_len) - input caption tokens
        """
        batch_size = mel.size(0)

        # Encode audio
        audio_encoding = self.encode_audio(mel)  # (batch, hidden_dim)

        # Initialize LSTM hidden state with audio encoding
        h0 = audio_encoding.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)

        # Decode captions
        embedded = self.embedding(captions)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.decoder_lstm(embedded, (h0, c0))  # (batch, seq_len, hidden_dim)
        logits = self.output_projection(lstm_out)  # (batch, seq_len, vocab_size)

        return logits

    def generate(self, mel, max_len=30, sos_idx=1, eos_idx=2, temperature=1.0, top_k=0, top_p=0.9):
        """
        Generate caption for given audio with sampling for diversity

        Args:
            mel: Audio mel spectrogram
            max_len: Maximum caption length
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
            temperature: Sampling temperature (higher = more random, lower = more deterministic)
            top_k: If > 0, only sample from top k tokens
            top_p: If > 0, nucleus sampling - sample from smallest set of tokens with cumulative prob >= p
        """
        self.eval()
        batch_size = mel.size(0)

        # Encode audio
        audio_encoding = self.encode_audio(mel)
        h = audio_encoding.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = torch.zeros_like(h)

        # Start with <sos> token
        input_token = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=mel.device)
        generated = []

        for _ in range(max_len):
            embedded = self.embedding(input_token)
            lstm_out, (h, c) = self.decoder_lstm(embedded, (h, c))
            logits = self.output_projection(lstm_out[:, -1, :])  # (batch, vocab_size)

            # Sample using temperature and nucleus sampling
            next_token = sample_with_temperature(logits, temperature, top_k, top_p)
            generated.append(next_token)

            # Stop if <eos> token
            if (next_token == eos_idx).all():
                break

            input_token = next_token

        return torch.cat(generated, dim=1)


# ============================================================================
# LEVEL 2: IMPROVED BASELINE
# ============================================================================

class ImprovedBaselineModel(nn.Module):
    """
    Improved baseline with:
    - Deeper CNN encoder
    - Bidirectional LSTM for better audio encoding
    - Residual connections
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()

        # Deeper Audio Encoder - Using AvgPool to preserve temporal info
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),  # -> (batch, 512, 4, 187)
        )

        # Bidirectional LSTM for temporal modeling
        self.temporal_lstm = nn.LSTM(
            512 * 4,
            hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )

        # Text Decoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.decoder_lstm = nn.LSTM(
            embed_dim + hidden_dim,  # Concatenate with audio context
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def encode_audio(self, mel):
        """Encode mel spectrogram"""
        batch_size = mel.size(0)

        # CNN features
        features = self.encoder(mel)  # (batch, 512, 4, 187)

        # Reshape for LSTM: (batch, time_steps, features)
        features = features.permute(0, 3, 1, 2)  # (batch, 187, 512, 4)
        features = features.reshape(batch_size, features.size(1), -1)  # (batch, 187, 512*4)

        # Temporal modeling with BiLSTM
        lstm_out, (h, c) = self.temporal_lstm(features)  # (batch, 187, hidden_dim)

        # Use mean pooling as audio context
        audio_context = lstm_out.mean(dim=1)  # (batch, hidden_dim)

        return audio_context

    def forward(self, mel, captions):
        """Forward pass for training"""
        batch_size = mel.size(0)
        seq_len = captions.size(1)

        # Encode audio
        audio_context = self.encode_audio(mel)  # (batch, hidden_dim)

        # Initialize decoder hidden state
        h0 = audio_context.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)

        # Embed captions and concatenate with audio context
        embedded = self.embedding(captions)  # (batch, seq_len, embed_dim)
        audio_context_expanded = audio_context.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_input = torch.cat([embedded, audio_context_expanded], dim=-1)
        decoder_input = self.dropout(decoder_input)

        # Decode
        lstm_out, _ = self.decoder_lstm(decoder_input, (h0, c0))
        logits = self.output_projection(lstm_out)

        return logits

    def generate(self, mel, max_len=30, sos_idx=1, eos_idx=2, temperature=1.0, top_k=0, top_p=0.9):
        """Generate caption with sampling for diversity"""
        self.eval()
        batch_size = mel.size(0)

        audio_context = self.encode_audio(mel)
        h = audio_context.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = torch.zeros_like(h)

        input_token = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=mel.device)
        generated = []

        for _ in range(max_len):
            embedded = self.embedding(input_token)
            decoder_input = torch.cat([embedded, audio_context.unsqueeze(1)], dim=-1)
            lstm_out, (h, c) = self.decoder_lstm(decoder_input, (h, c))
            logits = self.output_projection(lstm_out[:, -1, :])

            # Sample using temperature and nucleus sampling
            next_token = sample_with_temperature(logits, temperature, top_k, top_p)
            generated.append(next_token)

            if (next_token == eos_idx).all():
                break

            input_token = next_token

        return torch.cat(generated, dim=1)


# ============================================================================
# LEVEL 3: ATTENTION MODEL
# ============================================================================

class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.W_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        """
        encoder_outputs: (batch, seq_len, hidden_dim)
        decoder_hidden: (batch, hidden_dim)
        """
        # decoder_hidden: (batch, 1, hidden_dim)
        decoder_hidden = decoder_hidden.unsqueeze(1)

        # Compute attention scores
        score = self.V(torch.tanh(
            self.W_encoder(encoder_outputs) + self.W_decoder(decoder_hidden)
        ))  # (batch, seq_len, 1)

        # Attention weights
        attention_weights = F.softmax(score, dim=1)  # (batch, seq_len, 1)

        # Context vector
        context = (attention_weights * encoder_outputs).sum(dim=1)  # (batch, hidden_dim)

        return context, attention_weights.squeeze(-1)


class AttentionModel(nn.Module):
    """
    Attention-based model:
    - CNN + BiLSTM encoder (produces sequence of features)
    - Attention mechanism to focus on relevant audio parts
    - LSTM decoder with attention context
    """
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()

        # Audio Encoder - Using AvgPool instead of MaxPool to preserve temporal info
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )

        self.encoder_lstm = nn.LSTM(
            512 * 4,
            hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )

        # Attention
        self.attention = BahdanauAttention(hidden_dim)

        # Text Decoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.decoder_lstm = nn.LSTM(
            embed_dim + hidden_dim,  # Embed + attention context
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.output_projection = nn.Linear(hidden_dim * 2, vocab_size)  # LSTM + context
        self.dropout = nn.Dropout(0.3)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def encode_audio(self, mel):
        """Encode audio to sequence of features"""
        batch_size = mel.size(0)

        # CNN
        features = self.encoder_cnn(mel)  # (batch, 512, 4, 187)

        # Reshape for LSTM
        features = features.permute(0, 3, 1, 2)
        features = features.reshape(batch_size, features.size(1), -1)

        # BiLSTM
        encoder_outputs, _ = self.encoder_lstm(features)  # (batch, 187, hidden_dim)

        return encoder_outputs

    def forward(self, mel, captions):
        """Forward pass with attention"""
        batch_size = mel.size(0)
        seq_len = captions.size(1)

        # Encode audio
        encoder_outputs = self.encode_audio(mel)  # (batch, time_steps, hidden_dim)

        # Initialize decoder
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=mel.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=mel.device)

        # Embed captions
        embedded = self.embedding(captions)  # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)

        outputs = []
        for t in range(seq_len):
            # Attention context using current decoder state
            context, _ = self.attention(encoder_outputs, h[-1])  # (batch, hidden_dim)

            # LSTM input: embedding + attention context
            lstm_input = torch.cat([embedded[:, t:t+1, :], context.unsqueeze(1)], dim=-1)
            lstm_out, (h, c) = self.decoder_lstm(lstm_input, (h, c))

            # Output projection from LSTM output + context
            output = torch.cat([lstm_out.squeeze(1), context], dim=-1)
            outputs.append(self.output_projection(output))

        return torch.stack(outputs, dim=1)  # (batch, seq_len, vocab_size)

    def generate(self, mel, max_len=30, sos_idx=1, eos_idx=2, temperature=1.0, top_k=0, top_p=0.9):
        """Generate with attention and sampling for diversity"""
        self.eval()
        batch_size = mel.size(0)

        encoder_outputs = self.encode_audio(mel)

        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=mel.device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=mel.device)

        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=mel.device)
        generated = []

        for _ in range(max_len):
            # Attention
            context, _ = self.attention(encoder_outputs, h[-1])

            # Embed and decode
            embedded = self.embedding(input_token).unsqueeze(1)
            lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
            lstm_out, (h, c) = self.decoder_lstm(lstm_input, (h, c))

            # Output
            output = torch.cat([lstm_out.squeeze(1), context], dim=-1)
            logits = self.output_projection(output)

            # Sample using temperature and nucleus sampling
            next_token = sample_with_temperature(logits, temperature, top_k, top_p).squeeze(1)
            generated.append(next_token.unsqueeze(1))

            if (next_token == eos_idx).all():
                break

            input_token = next_token

        return torch.cat(generated, dim=1)


# ============================================================================
# LEVEL 4: TRANSFORMER MODEL
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """
    Full transformer model:
    - CNN for initial feature extraction
    - Transformer encoder for audio
    - Transformer decoder for text generation
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # Audio Encoder CNN - Using AvgPool to preserve temporal info
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
        )

        self.audio_projection = nn.Linear(512 * 4, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Text embedding and output
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.output_projection = nn.Linear(d_model, vocab_size)

        self.d_model = d_model
        self.vocab_size = vocab_size

    def encode_audio(self, mel):
        """Encode audio to transformer-ready features"""
        batch_size = mel.size(0)

        # CNN
        features = self.encoder_cnn(mel)  # (batch, 512, 4, 187)

        # Reshape to sequence
        features = features.permute(0, 3, 1, 2)  # (batch, 187, 512, 4)
        features = features.reshape(batch_size, features.size(1), -1)  # (batch, 187, 512*4)

        # Project to d_model
        features = self.audio_projection(features)  # (batch, 187, d_model)
        features = self.pos_encoder(features)

        return features

    def generate_square_subsequent_mask(self, sz, device):
        """Generate mask for causal attention"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, mel, captions):
        """Forward pass"""
        # Encode audio
        audio_features = self.encode_audio(mel)  # (batch, audio_len, d_model)

        # Embed captions
        embedded = self.embedding(captions) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)  # (batch, seq_len, d_model)

        # Create causal mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(captions.size(1), mel.device)

        # Transformer
        output = self.transformer(
            src=audio_features,
            tgt=embedded,
            tgt_mask=tgt_mask
        )  # (batch, seq_len, d_model)

        # Project to vocabulary
        logits = self.output_projection(output)  # (batch, seq_len, vocab_size)

        return logits

    def generate(self, mel, max_len=30, sos_idx=1, eos_idx=2, temperature=1.0, top_k=0, top_p=0.9):
        """Generate caption autoregressively with sampling for diversity"""
        self.eval()
        batch_size = mel.size(0)

        # Encode audio
        audio_features = self.encode_audio(mel)

        # Start with <sos>
        generated = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=mel.device)

        for _ in range(max_len - 1):
            # Embed current sequence
            embedded = self.embedding(generated) * math.sqrt(self.d_model)
            embedded = self.pos_encoder(embedded)

            # Create mask
            tgt_mask = self.generate_square_subsequent_mask(generated.size(1), mel.device)

            # Decode
            output = self.transformer(
                src=audio_features,
                tgt=embedded,
                tgt_mask=tgt_mask
            )

            # Get next token using sampling
            logits = self.output_projection(output[:, -1, :])
            next_token = sample_with_temperature(logits, temperature, top_k, top_p)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have <eos>
            if (next_token == eos_idx).all():
                break

        return generated[:, 1:]  # Remove <sos>


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_type, vocab_size, **kwargs):
    """
    Factory function to create models

    Args:
        model_type: 'baseline', 'improved_baseline', 'attention', 'transformer'
        vocab_size: Size of vocabulary
        **kwargs: Additional model-specific arguments
    """
    models = {
        'baseline': BaselineModel,
        'improved_baseline': ImprovedBaselineModel,
        'attention': AttentionModel,
        'transformer': TransformerModel
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](vocab_size, **kwargs)


if __name__ == "__main__":
    # Quick test
    vocab_size = 1000
    batch_size = 4
    mel = torch.randn(batch_size, 1, 64, 3000)
    captions = torch.randint(0, vocab_size, (batch_size, 20))

    print("Testing models...")
    for model_type in ['baseline', 'improved_baseline', 'attention', 'transformer']:
        print(f"\n{model_type.upper()}")
        model = create_model(model_type, vocab_size)
        logits = model(mel, captions)
        print(f"  Input: {mel.shape}, Captions: {captions.shape}")
        print(f"  Output: {logits.shape}")

        # Test generation
        generated = model.generate(mel, max_len=10, sos_idx=1, eos_idx=2)
        print(f"  Generated: {generated.shape}")
