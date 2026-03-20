# KNN - K Láng giềng gần nhất (cài đặt từ đầu, không dùng thư viện)
# Bộ phân loại cho các embedding được chiếu qua PCA
#
# Toán học cốt lõi:
#   Độ đo khoảng cách:
#     Euclidean:   d(x,y) = sqrt(sum((x_i - y_i)^2))
#     Cosine:      d(x,y) = 1 - (x.y)/(||x|| * ||y||)
#     Manhattan:   d(x,y) = sum(|x_i - y_i|)
#
#   Thuật toán:
#     1. Với mỗi mẫu test x:
#        - Tính d(x, x_i) cho tất cả mẫu huấn luyện x_i
#        - Chọn k láng giềng gần nhất theo khoảng cách nhỏ nhất
#        - Dự đoán nhãn bằng bỏ phiếu đa số trong k láng giềng
#
#   Siêu tham số:
#     - k: số láng giềng
#     - độ đo khoảng cách
import numpy as np

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
