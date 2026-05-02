"""Spectral Clustering with DTW-based affinity matrix.

Standard spectral clustering assumes a Euclidean similarity kernel.  This
module replaces that kernel with a Gaussian kernel applied to the **DTW
distance matrix**, converting dissimilarity into similarity:

.. math::

    W_{ij} = \\begin{cases}
        \\exp\\!\\left(-\\dfrac{\\mathrm{DTW}(s_i, s_j)^2}{2\\sigma^2}\\right)
            & i \\neq j \\\\
        0 & i = j
    \\end{cases}

The K-Means step that partitions the spectral embedding operates in
Euclidean space (sklearn ``KMeans``), since the embedding vectors are
ordinary R^k vectors, not time series.

Algorithm
---------
1. Compute the pairwise DTW distance matrix **D**.
2. Build the affinity matrix **W** via the Gaussian kernel above.
   The bandwidth σ is estimated automatically from the data (median
   heuristic) unless supplied by the user.
3. Construct the normalised symmetric graph Laplacian:
   **L** = **I** − **D**^{−½} **W** **D**^{−½},
   where **D** = diag(**W** 1) is the degree matrix.
4. Compute the *k* eigenvectors corresponding to the *k* smallest
   eigenvalues of **L** (Fiedler-vector embedding).
5. Row-normalise the embedding matrix to the unit sphere.
6. Apply ``sklearn.cluster.KMeans`` on the normalised embedding.

References
----------
Shi & Malik (2000). "Normalised cuts and image segmentation."
    *IEEE TPAMI*, 22(8), 888–905.

Ng, Jordan & Weiss (2002). "On spectral clustering: Analysis and an
    algorithm."  *NeurIPS 14*.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans

from dtw_clustering.distances.base import BaseDistance
from dtw_clustering.models.base import BaseClusterer


class SpectralClusteringDTW(BaseClusterer):
    """Spectral clustering using a DTW-based Gaussian affinity matrix.

    Parameters
    ----------
    n_clusters : int
        Number of clusters *k*.
    metric : str or BaseDistance
        Distance metric used to build the affinity matrix.
        Defaults to ``"dtw"``.
    sigma : float or None
        Bandwidth of the Gaussian kernel (controls how quickly affinity
        decays with DTW distance).  If ``None`` (default), σ is set to the
        median of all non-zero pairwise DTW distances — the *median
        heuristic*.
    n_init : int
        Number of K-Means restarts on the spectral embedding (passed to
        ``sklearn.cluster.KMeans``).
    n_jobs : int
        Parallel workers for distance matrix computation.
    random_state : int or None
    **metric_kwargs
        Forwarded to the metric constructor (e.g. ``constraint="sakoe_chiba"``,
        ``window=10``).

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster index of each sample (set after :meth:`fit`).
    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        The Gaussian-kernel affinity matrix **W** built from DTW distances.
    embedding_ : ndarray of shape (n_samples, n_clusters)
        Row-normalised spectral embedding used as input to K-Means.
    sigma_ : float
        Bandwidth value actually used (equals ``sigma`` if supplied, or
        the median-heuristic estimate otherwise).

    Examples
    --------
    >>> from dtw_clustering.models.classical import SpectralClusteringDTW
    >>> model = SpectralClusteringDTW(n_clusters=4, sigma=None)
    >>> labels = model.fit_predict(X)
    """

    def __init__(
        self,
        n_clusters: int = 2,
        metric: Union[str, BaseDistance] = "dtw",
        sigma: Optional[float] = None,
        n_init: int = 10,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        **metric_kwargs,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            metric=metric,
            n_jobs=n_jobs,
            random_state=random_state,
            **metric_kwargs,
        )
        if sigma is not None and sigma <= 0:
            raise ValueError("sigma must be a strictly positive float, or None for auto")
        self.sigma = sigma

        self.n_init = n_init

        self.affinity_matrix_: Optional[np.ndarray] = None
        self.embedding_: Optional[np.ndarray] = None
        self.sigma_: Optional[float] = None
        self._X_fit: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Affinity construction
    # ------------------------------------------------------------------

    def _build_affinity(self, D: np.ndarray) -> tuple[np.ndarray, float]:
        """Convert DTW distance matrix to Gaussian-kernel affinity matrix.

        Implements:

        .. math::

            W_{ij} = \\exp\\!\\left(-\\frac{D_{ij}^2}{2\\sigma^2}\\right), \\quad
            W_{ii} = 0

        Parameters
        ----------
        D : ndarray of shape (n, n)
            Symmetric pairwise DTW distance matrix (diagonal is zero).

        Returns
        -------
        W : ndarray of shape (n, n)
        sigma : float
            The bandwidth value used.
        """
        if self.sigma is not None:
            sigma = float(self.sigma)
        else:
            # Median heuristic: σ = median of all non-diagonal distances.
            off_diag = D[~np.eye(len(D), dtype=bool)]
            positive = off_diag[off_diag > 0]
            sigma = float(np.median(positive)) if len(positive) > 0 else 1.0

        W = np.exp(-(D ** 2) / (2.0 * sigma ** 2))
        np.fill_diagonal(W, 0.0)      # enforce W_ii = 0 (no self-loops)
        return W, sigma

    # ------------------------------------------------------------------
    # Normalised Laplacian
    # ------------------------------------------------------------------

    @staticmethod
    def _normalised_laplacian(W: np.ndarray) -> np.ndarray:
        """Compute the symmetric normalised graph Laplacian.

        .. math::

            \\mathbf{L} = \\mathbf{I} - \\mathbf{D}^{-1/2}
                          \\mathbf{W} \\mathbf{D}^{-1/2}

        where **D** = diag(**W** **1**) is the degree matrix.

        Isolated nodes (degree 0) are handled by clamping their inverse-sqrt
        degree to 0, leaving the corresponding row/column of **L** as the
        identity row.
        """
        d = W.sum(axis=1)                                # (n,)
        d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
        # L = I - D^{-1/2} W D^{-1/2}  (avoid forming full diagonal matrices)
        W_norm = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]
        L = np.eye(len(W)) - W_norm
        return L

    # ------------------------------------------------------------------
    # Spectral embedding
    # ------------------------------------------------------------------

    def _spectral_embedding(self, L: np.ndarray) -> np.ndarray:
        """Extract the *k* eigenvectors of the smallest eigenvalues.

        Uses :func:`scipy.linalg.eigh` (symmetric eigendecomposition) for
        numerical stability and to avoid computing the full spectrum.

        Parameters
        ----------
        L : ndarray of shape (n, n) — symmetric normalised Laplacian.

        Returns
        -------
        V : ndarray of shape (n, k) — row-normalised embedding.
        """
        k = self.n_clusters
        n = len(L)

        # eigh returns eigenvalues in ascending order; we want indices 0..k-1.
        _, V = eigh(L, subset_by_index=[0, min(k - 1, n - 1)])  # (n, k)

        # Row-normalise to the unit sphere (standard for normalised Laplacian).
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V = V / np.where(norms > 0, norms, 1.0)
        return V

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "SpectralClusteringDTW":
        """Compute spectral clustering.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timesteps[, n_features])

        Returns
        -------
        self
        """
        X = np.asarray(X)
        n = len(X)
        if n < self.n_clusters:
            raise ValueError(
                f"n_samples={n} is smaller than n_clusters={self.n_clusters}"
            )

        # Step 1 — DTW distance matrix
        D = self._compute_distance_matrix(X)             # (n, n)

        # Step 2 — Gaussian-kernel affinity
        W, sigma = self._build_affinity(D)

        # Step 3 — Normalised Laplacian
        L = self._normalised_laplacian(W)

        # Step 4–5 — Spectral embedding (k eigenvectors, row-normalised)
        V = self._spectral_embedding(L)                  # (n, k)

        # Step 6 — K-Means on the embedding (Euclidean space)
        km = KMeans(
            n_clusters=self.n_clusters,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        labels = km.fit_predict(V)

        self.labels_ = labels.astype(int)
        self.affinity_matrix_ = W
        self.embedding_ = V
        self.sigma_ = sigma
        self._X_fit = X
        self._D_fit = D                                  # kept for predict()
        return self

    # ------------------------------------------------------------------
    # Cluster centres and predict
    # ------------------------------------------------------------------

    def _cluster_centers(self) -> np.ndarray:
        """Return the per-cluster medoid (the training sample with minimum
        mean DTW distance to the other cluster members).

        Unlike K-Means, spectral clusters have no natural synthetic centre;
        the medoid is the most representative real sample.
        """
        self._check_is_fitted()
        X = self._X_fit
        D = self._D_fit
        centers = []
        for k in range(self.n_clusters):
            idx = np.where(self.labels_ == k)[0]
            if len(idx) == 0:
                # Degenerate — fall back to the global medoid.
                centers.append(X[int(np.argmin(D.sum(axis=1)))])
            elif len(idx) == 1:
                centers.append(X[idx[0]])
            else:
                # Intra-cluster distance submatrix; pick minimum mean-distance row.
                D_sub = D[np.ix_(idx, idx)]
                local_medoid = idx[int(np.argmin(D_sub.mean(axis=1)))]
                centers.append(X[local_medoid])
        return np.stack(centers, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new samples via nearest-neighbour lookup in DTW space.

        Each new sample is assigned to the cluster of its nearest training
        sample.  This is the standard inductive extension for transductive
        spectral clustering.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_timesteps[, n_features])

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype int
        """
        self._check_is_fitted()
        # D_new[i, j] = DTW(X_new[i], X_train[j])
        D_new = self._compute_distance_matrix(X, Y=self._X_fit)   # (m, n_train)
        nearest_train = np.argmin(D_new, axis=1)                   # (m,)
        return self.labels_[nearest_train].astype(int)
