# PCA - Phân tích thành phần chính (cài đặt từ đầu, không dùng thư viện)
# Eigenfaces dùng cho nhận dạng khuôn mặt
#
# Các bước toán học:
#   1. Tính khuôn mặt trung bình: mean = (1/N) * sum(x_i)
#   2. Trừ mean: X_centered = X - mean
#   3. Ma trận hiệp phương sai: C = (1/N) * X_centered.T @ X_centered
#      (Trick: dùng X @ X.T khi N < d (số mẫu < số chiều), rồi chuyển eigenvectors)
#   4. Phân rã trị riêng: C @ v = lambda * v
#   5. Sắp xếp eigenvectors theo eigenvalue giảm dần
#   6. Chiếu xuống: Z = X_centered @ W  (W: d x k, top-k eigenvectors)
#
# Ghi chú:
#   - Eigenvalues = lượng phương sai giải thích bởi mỗi thành phần
#   - Eigenvectors (eigenfaces) = các hướng chính trong không gian khuôn mặt
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
