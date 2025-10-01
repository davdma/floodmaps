# Model Training

### Wandb Integration
All training runs automatically log to [Weights & Biases](https://wandb.ai) for visualization and comparison.

**Logged Metrics:**
- Training/validation loss and accuracy
- Precision, recall, F1-score, accuracy, IoU, confusion matrices per epoch
- Learning rate scheduling
- Model parameters and memory usage

**Displays Model Results:**
- Input channels (RGB, SAR, ancillary data)
- Ground truth and model predictions
- False positive/negative analysis
- Reference imagery (TCI, NLCD)

### Pre-requisites

Before training a new model ensure preprocessed data exists, or run [preprocessing](../preprocess/README.md) scripts.