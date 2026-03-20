import numpy as np

class PCA_scratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_  = None
        self.mean_        = None
        self.explained_variance_ratio_ = None
      
    def fit(self, X):
        n_samples = X.shape[0]
        self.mean_ = np.mean(X, axis=0)
        X_c = X - self.mean_
        C_small = X_c @ X_c.T / (n_samples - 1)
        eigenvalues_small, eigenvectors_small = np.linalg.eigh(C_small)
        eigenvectors = X_c.T @ eigenvectors_small  # (4096, 320)
        norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
        norms[norms == 0] = 1
        eigenvectors = eigenvectors / norms
        eigenvalues  = eigenvalues_small[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        eigenvalues = np.maximum(eigenvalues, 0)
        self.components_ = eigenvectors[:, :self.n_components] 
        total_var = np.sum(eigenvalues)
        if total_var > 0:
            self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_var
        else:
            self.explained_variance_ratio_ = np.zeros(self.n_components)
        return self
      
    def transform(self, X):
        X_c = X - self.mean_
        return X_c @ self.components_
      
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class KNN_scratch:
    def __init__(self, k=5, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _compute_distances(self, X):
        if self.metric == 'euclidean':
            X_sq   = np.sum(X**2, axis=1, keepdims=True)
            Xtr_sq = np.sum(self.X_train**2, axis=1, keepdims=True).T
            dot    = X @ self.X_train.T
            dist_sq = np.maximum(X_sq + Xtr_sq - 2*dot, 0)
            return np.sqrt(dist_sq)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X[:, None, :] - self.X_train[None, :, :]), axis=2)
        else:
            raise ValueError(f"Metric '{self.metric}' chưa được hỗ trợ")

    def predict(self, X):
        distances = self._compute_distances(X)
        k_nearest_idx = np.argsort(distances, axis=1)[:, :self.k]
        predictions = []
        for i in range(len(X)):
            k_labels = self.y_train[k_nearest_idx[i]]
            unique, counts = np.unique(k_labels, return_counts=True)
            predictions.append(unique[np.argmax(counts)])
        return np.array(predictions)
