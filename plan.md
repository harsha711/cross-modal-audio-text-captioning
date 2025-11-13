audio_captioning/
│
├── notebooks/
│ ├── 01_data_exploration.ipynb # Your current notebook (data loading)
│ ├── 02_baseline_experiments.ipynb # Test baseline model
│ ├── 03_attention_experiments.ipynb # Test attention model
│ └── 04_results_analysis.ipynb # Compare all results
│
├── src/
│ ├── **init**.py
│ ├── models.py # All model architectures
│ ├── dataset.py # ClothoDataset class
│ ├── trainer.py # ModelTrainer class
│ ├── evaluation.py # Evaluation metrics
│ └── utils.py # Helper functions
│
├── scripts/
│ ├── train_baseline.py # Train baseline model
│ ├── train_attention.py # Train attention model
│ ├── train_transformer.py # Train transformer
│ └── evaluate_all.py # Compare all models
│
├── configs/
│ ├── baseline_config.yaml # Hyperparameters for baseline
│ ├── attention_config.yaml # Hyperparameters for attention
│ └── transformer_config.yaml # Hyperparameters for transformer
│
├── features/ # Your existing features
│ ├── mel/
│ └── mel_eval/
│
├── checkpoints/ # Saved models
│ ├── baseline_best.pth
│ ├── attention_best.pth
│ └── transformer_best.pth
│
├── results/ # Experiment results
│ ├── baseline_results.json
│ ├── attention_results.json
│ └── comparison_plots.png
│
├── vocab.json # Your vocabulary
├── requirements.txt # Dependencies
└── README.md # Project documentation
