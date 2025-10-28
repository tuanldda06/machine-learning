import numpy as np                  
import matplotlib.pyplot as plt       

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))  


def update_centroids(X, labels, K):
    centroids = np.zeros((K, X.shape[1]))  
    for k in range(K):                    
        points_in_cluster = X[labels == k]  
        if len(points_in_cluster) > 0:     
            centroids[k] = np.mean(points_in_cluster, axis=0)  
        else:                              
            centroids[k] = X[np.random.choice(len(X))]  
    return centroids                      
