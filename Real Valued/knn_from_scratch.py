# knn_from_scratch.py
from typing import Literal
import numpy as np

def pairwise_distances(X: np.ndarray, Y: np.ndarray, metric: Literal['euclidean','manhattan']='euclidean') -> np.ndarray:
    """
    Compute pairwise distances between rows of X (n x d) and rows of Y (m x d).
    Returns (n x m) array where entry [i, j] is distance between X[i] and Y[j].
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    if metric == 'euclidean':
        # Use (x-y)^2 = x^2 + y^2 - 2xy
        XX = np.sum(X**2, axis=1)[:, None]
        YY = np.sum(Y**2, axis=1)[None, :]
        D2 = XX + YY - 2 * (X @ Y.T)
        D2 = np.maximum(D2, 0.0)
        return np.sqrt(D2)
    elif metric == 'manhattan':
        # careful with memory for very large inputs
        return np.sum(np.abs(X[:, None, :] - Y[None, :, :]), axis=2)
    else:
        raise ValueError("Unsupported metric")

class KNNClassifierScratch:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def predict(self, X: np.ndarray, k: int = 3, metric: str = 'euclidean', weights: str = 'uniform') -> np.ndarray:
        X = np.asarray(X)
        D = pairwise_distances(X, self.X_train, metric=metric)  # (n_test, n_train)
        idx = np.argsort(D, axis=1)[:, :k]                       # (n_test, k)
        neigh_dists = np.take_along_axis(D, idx, axis=1)
        preds = []
        for i in range(X.shape[0]):
            labs = self.y_train[idx[i]]
            dists = neigh_dists[i]
            if weights == 'uniform':
                vals, counts = np.unique(labs, return_counts=True)
                maxc = np.max(counts)
                candidates = vals[counts == maxc]
                if len(candidates) == 1:
                    preds.append(candidates[0])
                else:
                    # tie-break: choose label with smallest average neighbor distance
                    avg = {c: np.mean(dists[labs == c]) for c in candidates}
                    preds.append(min(avg, key=avg.get))
            elif weights == 'distance':
                eps = 1e-8
                w = 1.0 / (dists + eps)
                score = {}
                for lab, wt in zip(labs, w):
                    score[lab] = score.get(lab, 0.0) + wt
                # choose label with highest weighted score; break ties by label numeric value
                best = max(score.items(), key=lambda x: (x[1], -x[0]))[0]
                preds.append(best)
            else:
                raise ValueError("weights must be 'uniform' or 'distance'")
        return np.array(preds)

class KNNRegressorScratch:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def predict(self, X: np.ndarray, k: int = 3, metric: str = 'euclidean', weights: str = 'uniform') -> np.ndarray:
        X = np.asarray(X)
        D = pairwise_distances(X, self.X_train, metric=metric)
        idx = np.argsort(D, axis=1)[:, :k]
        neigh_dists = np.take_along_axis(D, idx, axis=1)
        preds = []
        for i in range(X.shape[0]):
            targets = self.y_train[idx[i]]
            dists = neigh_dists[i]
            if weights == 'uniform':
                preds.append(np.mean(targets))
            elif weights == 'distance':
                eps = 1e-8
                w = 1.0 / (dists + eps)
                preds.append(np.sum(w * targets) / np.sum(w))
            else:
                raise ValueError("weights must be 'uniform' or 'distance'")
        return np.array(preds)
