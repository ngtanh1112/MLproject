import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, K):
        self.K = K  # Số cụm 
    
    def kmeans_init_centers(self, X):
        # Lấy k hàng của X làm center ngẫu nhiên
        # return X[np.random.choice(X.shape[0], self.K, replace=False)]
        return X.iloc[np.random.choice(X.shape[0], self.K, replace=False)]    
    
    def kmeans_assign_labels(self, X, centers):
        # Tính khoảng cách từ các điểm dữ liệu tới các center
        D = cdist(X, centers)
        # Trả về index của center gần nhất
        return np.argmin(D, axis=1)
    
    def kmeans_update_centers(self, X, labels):
        centers = np.zeros((self.K, X.shape[1]))

        for k in range(self.K):
            # Lấy các chỉ số dữ liệu có nhãn k
            # Xk = X[labels == k, :]
            Xk = X.iloc[labels == k, :].values  # Sử dụng iloc và .values để lấy mảng numpy
            centers[k, :] = np.mean(Xk, axis=0)
        return centers

    
    def has_converged(self, centers, new_centers):
        # Kiểm tra xem các center hiện tại và mới có giống nhau không
        return set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers])
    
    def kmeans(self, X):
        # Khởi tạo các centers ngẫu nhiên
        centers = [self.kmeans_init_centers(X)]

        labels = []
        it = 0
        
        while True:
            # Gán nhãn cho các điểm dữ liệu
            labels.append(self.kmeans_assign_labels(X, centers[-1]))
            # Cập nhật các centers mới
            new_centers = self.kmeans_update_centers(X, labels[-1])

            # Kiểm tra điều kiện dừng 
            if self.has_converged(centers[-1], new_centers):
                break
            centers.append(new_centers)
            it += 1
            
        return centers, labels, it

    def display(self, X, centers, labels):
            plt.figure(figsize=(8, 6))
            plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', s=30)
            plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100, label='Centers')
            plt.title('KMeans Clustering Result')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            plt.show()