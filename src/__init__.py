"""
Audio Captioning Package
Cross-model audio text captioning system with multiple architectures
"""

from .models import (
    BaselineModel,
    ImprovedBaselineModel,
    AttentionModel,
    TransformerModel,
    create_model
)

from .dataset import (
    ClothoDataset,
    ClothoEvalDataset,
    create_dataloaders
)

from .trainer import (
    ModelTrainer,
    train_single_model,
    train_all_models
)

from .evaluation import (
    calculate_repetition_rate,
    evaluate_diversity,
    evaluate_model,
    compare_models,
    get_sample_predictions,
    print_sample_predictions
)

from .utils import (
    build_vocab,
    save_vocab,
    load_vocab,
    plot_training_history,
    plot_model_comparison,
    plot_evaluation_metrics,
    count_parameters,
    set_seed,
    get_device,
    decode_caption
)

__version__ = '1.0.0'
__author__ = 'Audio Captioning Team'

__all__ = [
    # Models
    'BaselineModel',
    'ImprovedBaselineModel',
    'AttentionModel',
    'TransformerModel',
    'create_model',

    # Dataset
    'ClothoDataset',
    'ClothoEvalDataset',
    'create_dataloaders',

    # Training
    'ModelTrainer',
    'train_single_model',
    'train_all_models',

    # Evaluation
    'calculate_repetition_rate',
    'evaluate_diversity',
    'evaluate_model',
    'compare_models',
    'get_sample_predictions',
    'print_sample_predictions',

    # Utils
    'build_vocab',
    'save_vocab',
    'load_vocab',
    'plot_training_history',
    'plot_model_comparison',
    'plot_evaluation_metrics',
    'count_parameters',
    'set_seed',
    'get_device',
    'decode_caption',
]
