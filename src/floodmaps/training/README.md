# Model Training

### Wandb Integration
All training runs automatically log to [Weights & Biases](https://wandb.ai) for visualization and comparison.

**Logged Metrics:**
- Training/validation loss and accuracy
- Precision, recall, F1-score, accuracy, IoU, confusion matrices per epoch
- Learning rate scheduling
- Model parameters and memory usage
- Final saved model val/test metrics logged to summary dictionary

**Displays Model Results:**
- Input channels (RGB, SAR, ancillary data)
- Ground truth and model predictions
- False positive/negative analysis
- Reference imagery (TCI, NLCD)

### Pre-requisites

Before training a new model ensure preprocessed data exists, or run [preprocessing](../preprocess/README.md) scripts.

---

## VAE/CVAE Posterior Collapse Monitoring

Applies to scripts:
- `train_multi_cvae_ddp.py` - DDP CVAE training
- `train_multi_cvae.py` - Single-GPU CVAE training
- `train_multi_vae.py` - Single-GPU VAE training

### Why Monitor for Posterior Collapse?

Posterior collapse is a critical failure mode in Variational Autoencoders where the decoder learns to ignore the latent variable z entirely. When this happens:

1. **The model degenerates**: The VAE becomes equivalent to a deterministic autoencoder, losing its generative capabilities
2. **Latent space becomes meaningless**: The learned representations carry no useful information
3. **KL divergence vanishes**: The approximate posterior q(z|x) collapses to the prior p(z) = N(0, I)

For SAR despeckling, posterior collapse means the CVAE/VAE decoder produces outputs based solely on the conditioning signal (noisy input), completely ignoring the latent code. This defeats the purpose of the variational architecture and may limit model expressiveness.

### Monitored Metrics

Metrics ending with `^` in wandb are **critical warning indicators** that require immediate attention.

| Metric | Description | Healthy Range | Collapse Warning |
|--------|-------------|---------------|------------------|
| `AU_strict^` | Active Units (Var[mu] > 1e-2) | > 50% of latent_dim | < 10% |
| `AU_lenient^` | Active Units (Var[mu] > 1e-3) | > 70% of latent_dim | < 20% |
| `median_KL^` | Median per-sample KL divergence | > 1.0 | < 0.1 |
| `frac_KL_small^` | Fraction of samples with KL < 1e-3 | < 0.1 | > 0.5 |
| `ablation_delta^` | L(z=0) - L(z=mu) reconstruction diff | > 0.01 | ~ 0 |
| `log_var_min/max` | Raw log variance extrema | [-6, 2] | Outside [-10, 10] |

### Interpreting the Metrics

**Active Units (AU)**: Counts latent dimensions where the variance of the posterior mean across samples exceeds a threshold. Low AU means most dimensions have collapsed to outputting the same value regardless of input. Uses Welford's streaming algorithm for memory efficiency.

**Median KL**: More robust than mean KL since it's not skewed by outliers. If median KL approaches zero, most samples have posteriors identical to the prior.

**Frac(KL < 1e-3)**: Directly measures what fraction of training samples have effectively zero KL. High values indicate widespread collapse across the dataset.

**Ablation Delta**: The definitive test - run the decoder with z=mu (actual posterior mean) and z=0 (ablated). If the reconstruction loss is the same (delta ~ 0), the decoder is literally ignoring z. Computed on the first validation/test batch each epoch.

**Log Variance Extrema**: Track for numerical instability. Extremely negative log_var (variance near 0) or extremely positive (exploding variance) indicates training problems. Values are tracked before the [-6, 6] clamping applied during loss computation.