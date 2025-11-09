from __future__ import print_function 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)

def kmeans_init_centroids(X, K):
    return X[np.random.choice(X.shape[0], K, replace=False)]

def kmeans_assign_centroids(X, K):
    D = cdist(X, K)
    return np.argmin(D, axis=1)

def kmeans_update_centers(X, label, K): 
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        X_label = X[label == k, :]
        if X_label.shape[0] > 0:  # Kiểm tra xem cụm có điểm nào không
            centers[k, :] = np.mean(X_label, axis=0)
    return centers

def has_converged(centers, new_centers, tol=1e-6):
    return np.allclose(centers, new_centers, rtol=0.0, atol=tol)

def main_kmeans(X, K):
    centers = [kmeans_init_centroids(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_centroids(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers, tol=1e-6):
            break
        centers.append(new_centers)
        it += 1
    return centers, labels, it

# Dữ liệu huấn luyện ban đầu
X_train = np.r_[  # r_ dùng để ghép dọc nhiều mảng
    np.random.randn(100, 2) + [0, 0],   # cụm 1
    np.random.randn(100, 2) + [5, 5],   # cụm 2
    np.random.randn(100, 2) + [0, 5]    # cụm 3
]
K = 3
centers, labels, it = main_kmeans(X_train, K)

# Vẽ kết quả trên dữ liệu huấn luyện
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=labels[-1])
plt.scatter(centers[-1][:, 0], centers[-1][:, 1], 
           marker='x', s=200, c='red', label='centers')
plt.legend()
plt.title(f'K-Means trên dữ liệu huấn luyện, {it} vòng lặp')

# Tạo dữ liệu kiểm thử
np.random.seed(42)  
X_test = np.r_[
    np.random.randn(50, 2) + [0, 0],   # cụm 1
    np.random.randn(50, 2) + [5, 5],   # cụm 2
    np.random.randn(50, 2) + [0, 5]    # cụm 3
]


centers_test, labels_test, it_test = main_kmeans(X_test, K)


plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=labels_test[-1])
plt.scatter(centers_test[-1][:, 0], centers_test[-1][:, 1], 
           marker='x', s=200, c='red', label='centers')
plt.legend()
plt.title(f'K-Means trên dữ liệu kiểm thử, {it_test} vòng lặp')

plt.tight_layout()
plt.show()


