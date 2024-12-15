import numpy as np
from collections import defaultdict

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # Tính khoảng cách từ x đến tất cả các điểm trong tập huấn luyện
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # Lấy chỉ số của k điểm gần nhất
        k_indices = np.argsort(distances)[:self.k]

        # Lấy nhãn của k điểm gần nhất
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_distances = [distances[i] for i in k_indices]

        # Weighted voting thay vì Distance voting
        label_weights = defaultdict(float)

        for label, dist in zip(k_nearest_labels, k_nearest_distances):
            if dist == 0:
                # Nếu khoảng cách bằng 0, trả về trực tiếp nhãn này
                return label
            label_weights[label] += 1 / dist

        # Trả về nhãn có trọng số cao nhất
        return max(label_weights, key=label_weights.get)
