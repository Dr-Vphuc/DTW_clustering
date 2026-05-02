# DTW Clustering — Model Fitting Guide

## Data format

All models expect a NumPy array of shape `(N, d, T)`:

| Symbol | Meaning |
|--------|---------|
| `N` | number of samples |
| `d` | number of variates (channels) — use `d=1` for univariate |
| `T` | sequence length (time steps) |

```python
import numpy as np

X = np.random.randn(200, 3, 50).astype(np.float32)
# 200 samples, 3 channels, 50 time steps
```

Univariate series of shape `(N, T)` must be reshaped:

```python
X = X[:, np.newaxis, :]   # (N, T) → (N, 1, T)
```

---

## Classical models

All classical models follow the same pattern:

```python
model = ModelClass(n_clusters=K, ...)
model.fit(X)
labels = model.labels_
```

The `metric` parameter accepts `"dtw"` (default), `"soft_dtw"`, or any
`BaseDistance` instance. Extra keyword arguments are forwarded to the metric.

---

### KMedoidsDTW

```python
from dtw_clustering.models import KMedoidsDTW

model = KMedoidsDTW(
    n_clusters=4,
    metric="dtw",        # "dtw" | "soft_dtw" | BaseDistance
    init="kmedoids++",   # "kmedoids++" | "random"
    max_iter=300,
    n_jobs=-1,           # parallel pairwise distance computation
    random_state=0,
    # metric_kwargs forwarded to the distance, e.g. gamma=0.1 for soft_dtw
)
model.fit(X)
print(model.labels_)           # (N,)
print(model.medoid_indices_)   # indices of the K medoids in X
print(model.inertia_)          # sum of distances to nearest medoid
```

---

### KMeansDTW

Uses DBA (DTW Barycenter Averaging) to compute cluster centres.

```python
from dtw_clustering.models import KMeansDTW

model = KMeansDTW(
    n_clusters=4,
    metric="dtw",
    init="dtw_kmeans++",   # "dtw_kmeans++" | "random"
    max_iter=100,
    tol=0.0,
    dba_max_iter=30,       # inner DBA iterations per M-step
    n_jobs=-1,
    random_state=0,
)
model.fit(X)
print(model.labels_)           # (N,)
print(model.cluster_centers_)  # (K, d, T) — DBA barycenters
print(model.inertia_)
print(model.n_iter_)           # number of outer K-Means iterations run
```

---

### AgglomerativeDTW

```python
from dtw_clustering.models import AgglomerativeDTW

model = AgglomerativeDTW(
    n_clusters=4,
    metric="dtw",
    linkage_method="complete",  # "single" | "complete" | "average" | "ward"
    n_jobs=-1,
    random_state=0,
)
model.fit(X)
print(model.labels_)          # (N,)
print(model.linkage_matrix_)  # (N-1, 4) — SciPy linkage format
```

---

### SpectralClusteringDTW

Converts the pairwise DTW distance matrix to a Gaussian affinity matrix
then runs spectral embedding + K-Means.

```python
from dtw_clustering.models import SpectralClusteringDTW

model = SpectralClusteringDTW(
    n_clusters=4,
    metric="dtw",
    sigma=None,   # RBF bandwidth; None → median heuristic
    n_init=10,    # K-Means restarts in the spectral step
    n_jobs=-1,
    random_state=0,
)
model.fit(X)
print(model.labels_)           # (N,)
print(model.affinity_matrix_)  # (N, N) Gaussian affinity
print(model.embedding_)        # (N, K) spectral embedding
print(model.sigma_)            # bandwidth used
```

---

### AffinityPropagationDTW

Does **not** require specifying `n_clusters` in advance — the number of
clusters is determined by the algorithm.

```python
from dtw_clustering.models import AffinityPropagationDTW

model = AffinityPropagationDTW(
    preference="median",  # "median" | "min" | float | array (N,)
    metric="dtw",
    sigma=None,           # RBF bandwidth for the affinity; None → median
    damping=0.5,          # message-passing damping factor in [0.5, 1)
    max_iter=200,
    convits=15,           # convergence window (iterations without change)
    n_jobs=-1,
    random_state=0,
)
model.fit(X)
print(model.labels_)                  # (N,)
print(model.n_clusters_)              # number of clusters found
print(model.cluster_centers_indices_) # indices of exemplars in X
```

---

## Deep models

All deep models share:

1. A `TimeSeriesAutoencoder` backbone (Conv1D encoder/decoder).
2. A joint pretrain phase (metric loss + reconstruction loss).
3. A fine-tuning phase specific to each model.

### Building the autoencoder

```python
from dtw_clustering.models import TimeSeriesAutoencoder

ae = TimeSeriesAutoencoder(
    input_channels=3,      # d — number of variates
    seq_len=50,            # T — sequence length
    latent_dim=32,         # embedding dimension
    conv_channels=[32, 64, 128],  # encoder channel sizes (default)
    kernel_size=3,
    dropout_rate=0.2,
)
```

`TimeSeriesAutoencoder` instances are **not** shared between models — create
one per model.

---

### DEC — Deep Embedded Clustering

Fine-tuning discards the decoder and optimises encoder + cluster centres
via KL divergence (soft assignment → auxiliary target).

```python
from dtw_clustering.models import DEC, TimeSeriesAutoencoder

ae = TimeSeriesAutoencoder(input_channels=3, seq_len=50, latent_dim=32)
model = DEC(
    n_clusters=4,
    autoencoder=ae,
    update_interval=5,   # target distribution P refresh interval (batches)
    alpha=1.0,           # Student's-t degrees of freedom
    random_state=0,
)
model.fit(
    X,
    pretrain_epochs=100,
    pretrain_lr=1e-3,
    epochs=200,          # fine-tuning epochs
    lr=1e-4,
    batch_size=256,
    sdtw_gamma=0.1,
    verbose=True,
)
print(model.labels_)     # (N,)
```

---

### IDEC — Improved DEC

Keeps the decoder during fine-tuning so reconstruction loss prevents
encoder distortion.

```python
from dtw_clustering.models import IDEC, TimeSeriesAutoencoder

ae = TimeSeriesAutoencoder(input_channels=3, seq_len=50, latent_dim=32)
model = IDEC(
    n_clusters=4,
    autoencoder=ae,
    update_interval=5,
    alpha=1.0,
    gamma=0.1,           # weight of clustering loss vs reconstruction loss
    random_state=0,
)
model.fit(
    X,
    pretrain_epochs=100,
    pretrain_lr=1e-3,
    epochs=200,
    lr=1e-4,
    batch_size=256,
    sdtw_gamma=0.1,
    verbose=True,
)
print(model.labels_)
```

---

### DECS — DEC with Sample Stability

Augments inputs during pretrain (jitter + magnitude scaling) and uses a
stability-based loss instead of KL divergence during fine-tuning.

```python
from dtw_clustering.models import DECS, TimeSeriesAutoencoder

ae = TimeSeriesAutoencoder(input_channels=3, seq_len=50, latent_dim=32)
model = DECS(
    n_clusters=4,
    autoencoder=ae,
    update_interval=5,
    alpha=1.0,
    jitter_std=0.05,     # additive Gaussian noise for augmentation
    scale_std=0.1,       # multiplicative scaling noise for augmentation
    lambda_s=0.5,        # weight of stability penalty
    random_state=0,
)
model.fit(
    X,
    pretrain_epochs=100,
    pretrain_lr=1e-3,
    epochs=200,
    lr=1e-4,
    batch_size=256,
    sdtw_gamma=0.1,
    verbose=True,
)
print(model.labels_)
```

---

### DSC — Deep Subspace Clustering

**Transductive**: learns a self-expressive N×N matrix C tied to the
training set. Cannot predict on new samples. Runs full-batch (no
`batch_size` argument).

```python
from dtw_clustering.models import DSC, TimeSeriesAutoencoder

ae = TimeSeriesAutoencoder(input_channels=3, seq_len=50, latent_dim=32)
model = DSC(
    n_clusters=4,
    autoencoder=ae,
    lambda1=1.0,     # sparsity / Frobenius regularisation on C
    lambda2=0.5,     # self-expressiveness reconstruction weight
    reg="L1",        # "L1" | "L2" regularisation on C
    random_state=0,
)
model.fit(
    X,
    pretrain_epochs=100,
    pretrain_lr=1e-3,
    epochs=1000,              # full-batch fine-tuning steps
    lr=1e-3,
    sdtw_gamma=0.1,
    sdtw_gamma_finetune=0.1,  # γ for fine-tuning reconstruction
    verbose=True,
)
print(model.labels_)   # (N,) — transductive, only valid for training X
```

---

### DAMIC — Deep Autoencoding Mixture for Imaging Clustering

Trains K independent cluster decoders and a routing network (ClusterNet)
in multiple phases.

```python
from dtw_clustering.models import DAMIC, TimeSeriesAutoencoder

ae = TimeSeriesAutoencoder(input_channels=3, seq_len=50, latent_dim=32)
model = DAMIC(
    n_clusters=4,
    autoencoder=ae,
    ce_warmup_epochs=10,     # ClusterNet warm-up epochs (Phase 4a)
    random_state=0,
)
model.fit(
    X,
    pretrain_epochs=100,
    pretrain_lr=1e-3,
    cluster_ae_epochs=50,    # per-cluster decoder training (Phase 3)
    cluster_ae_lr=1e-3,
    ce_warmup_lr=1e-3,
    epochs=100,              # joint fine-tuning (Phase 4b)
    lr=1e-3,
    batch_size=256,
    sdtw_gamma=0.1,
    sdtw_gamma_finetune=0.1,
    verbose=True,
)
print(model.labels_)         # (N,)
```

---

## Common patterns

### Pretraining separately

Pretrain once, save weights, reload for fine-tuning runs:

```python
model.pretrain(
    X,
    epochs=100,
    lr=1e-3,
    batch_size=256,
    sdtw_gamma=0.1,
    metric_target_gamma=0.05,  # γ for computing N×N DTW target matrix
    lambda_metric=1.0,         # metric loss weight
    lambda_recon=0.5,          # reconstruction loss weight
    n_jobs=-1,                 # parallel DTW matrix computation
    verbose=True,
)
model.save_weights("checkpoints/ae_pretrained.pt")
```

Load on a later run to skip pretraining:

```python
model.load_weights("checkpoints/ae_pretrained.pt")
# _pretrained is now True; fit() will not re-run pretrain
model.fit(X, pretrain_epochs=0, epochs=200, ...)
```

### Supplying a precomputed distance matrix

For large N, compute the target matrix once and reuse:

```python
from dtw_clustering.distances import soft_dtw_distance_matrix

D = soft_dtw_distance_matrix(X, gamma=0.05, n_jobs=-1)  # (N, N)

model.pretrain(X, target_matrix=D, epochs=100, ...)
```

### fit_predict shortcut

All models expose `fit_predict()`:

```python
labels = model.fit_predict(X, ...)   # equivalent to fit() then labels_
```

### Choosing `n_jobs`

`n_jobs` in classical models controls pairwise distance parallelism.  
`n_jobs` in `pretrain` controls only the target DTW matrix computation —
the neural network training itself runs on the device (CPU/CUDA).

### GPU support

Deep models move the autoencoder to CUDA automatically if available.
To force a device:

```python
model = DEC(n_clusters=4, autoencoder=ae, device="cuda:0")
model = DEC(n_clusters=4, autoencoder=ae, device="cpu")
```
