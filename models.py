import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
try:
    from pyod.models.hbos import HBOS
except ImportError:
    HBOS = None  # If PyOD is not installed, handle accordingly

def run_isolation_forest(X, contamination=0.1):
    """Run Isolation Forest and return a boolean mask of outliers."""
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)
    labels = model.predict(X)  # 1 for inliers, -1 for outliers
    return (labels == -1)

def run_lof(X, contamination=0.1):
    """Run Local Outlier Factor and return a boolean mask of outliers."""
    model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
    labels = model.fit_predict(X)  # 1 for inliers, -1 for outliers
    return (labels == -1)

def run_dbscan(X, eps=0.5, min_samples=5):
    """Run DBSCAN and return a boolean mask of outlier points (noise)."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)  # cluster labels, -1 is noise
    return (labels == -1)

def run_hbos(X, contamination=0.1):
    """Run HBOS (Histogram-Based Outlier Score) and return a boolean mask of outliers."""
    if HBOS is None:
        raise ImportError("HBOS model not available. Please install PyOD to use HBOS.")
    model = HBOS(contamination=contamination)
    model.fit(X)
    # PyOD models: labels_ attribute (0 = inlier, 1 = outlier) after fit
    if hasattr(model, 'labels_'):
        labels = model.labels_
    else:
        labels = model.predict(X)
    return (labels == 1)

def detect_outliers_simple(X):
    """
    Simple detection mode: use a single outlier detection model.
    Here we use Isolation Forest for a quick detection.
    """
    return run_isolation_forest(X, contamination=0.1)

def detect_outliers_balanced(X):
    """
    Balanced detection mode: combine multiple models (Isolation Forest, LOF, HBOS) via majority vote.
    An outlier is flagged only if at least two of the three detectors agree.
    """
    mask_if = run_isolation_forest(X, contamination=0.1)
    mask_lof = run_lof(X, contamination=0.1)
    # If HBOS is available, use it; otherwise treat HBOS votes as all False (no effect)
    if HBOS is not None:
        mask_hbos = run_hbos(X, contamination=0.1)
    else:
        mask_hbos = np.zeros(X.shape[0], dtype=bool)
    # Majority vote: outlier if at least 2 out of 3 detectors flag it
    votes = mask_if.astype(int) + mask_lof.astype(int) + mask_hbos.astype(int)
    return (votes >= 2)

def detect_outliers_complex(X):
    """
    Complex detection mode: use an ensemble of multiple models (Isolation Forest, LOF, HBOS, DBSCAN).
    An outlier is flagged if **any** of the detectors marks it as outlier (union of all).
    This mode is more sensitive and may detect a broader set of anomalies.
    """
    mask_if = run_isolation_forest(X, contamination=0.1)
    mask_lof = run_lof(X, contamination=0.1)
    mask_dbscan = run_dbscan(X, eps=0.5, min_samples=5)
    if HBOS is not None:
        mask_hbos = run_hbos(X, contamination=0.1)
    else:
        mask_hbos = np.zeros(X.shape[0], dtype=bool)
    # Union of all outlier flags: True if any model flagged the point
    outlier_mask = mask_if | mask_lof | mask_dbscan | mask_hbos
    return outlier_mask
