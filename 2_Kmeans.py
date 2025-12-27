from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(10)


def kmeans_innit(X,K) :
    return X[np.random.choice(X.shape[0],K , replace = False)] # chọn ra K điểm ngẫu nhiên bất kì thuộc tập X làm tâm ban đầu 

def kmeans_assign(X, centers):
    D = cdist(X, centers)
    return np.argmin(D, axis=1) #dùng axis=1 ở np.argmin vì ta cần lấy min theo từng hàng (mỗi hàng là một điểm), để tìm tâm gần nhất cho MỖI điểm .Còn axis = 0 thì nó so sánh theo cột của từng cái một với nhau , nên không đúng

def kmeans_update_centroids(X,label , K ) :
    centers = np.zeros((K,X.shape[1])) #  khởi tạo tâm ban đầu sẽ là 1 tập rỗng bằng 0 , và có số chiều = số chiều của X , do có K cụm nên số hàng bằng số K luôn 
    for k in range(K) :
        Xk = X[label == k ,:]#Lấy tất cả điểm nằm trong cụm k → mỗi điểm gồm đầy đủ tọa độ (tức toàn bộ cột).
        if Xk.shape[0] > 0:
            centers[k, :] = np.mean(Xk , axis = 0) # nghĩa là lấy trung bình của tất cả các điểm nằm trong cụm theo chiều cột , nghĩa là trung bình của tất cả phần từ cột x và trung bình của tất cả phần từ cột y
        else:
            centers[k, :] = X[np.random.randint(0, X.shape[0])]# nếu cụm rỗng, random lại tâm 
    return centers

def has_converge(center , new_center , tol = 1e-6) :
    return np.allclose(center , new_center,rtol = 0.0 , atol = tol)

def main_kmeans(X,K) :
    center= kmeans_innit(X,K)# khởi tạo tâm ban đầu
    
    while True : # đảm bảo khi thuật toán đúng 
        labels = kmeans_assign(X,center) # lúc này hàm sẽ trả về label( hay index) của cái vị trí tâm khởi tạo ban đầu mà gần mỗi điểm dữ liệu thuộc X nhất
        new_center = kmeans_update_centroids(X,labels , K ) # ứng với từng label thì ta sẽ  lấy trung bình của tất cả các điểm gần index nhất theo cột tức là trung bình tất cả theo cột x và cột y
        if has_converge(center , new_center , tol = 1e-6):
            break
        center = new_center
        
    return center ,labels

X = np.r_[
    np.random.randn(1000,2) + [5,5],
    np.random.randn(1000,2) + [0,0],
    np.random.randn(1000,2) + [0,5],
    np.random.randn(1000,2) + [5,10],
]
#np.random.randn(1000, 2) → 1000 điểm, mỗi điểm có 2 chiều (x, y).Ghép 4 cụm lại bằng np.r_ → tổng cộng 4000 điểm, vẫn chỉ 2 chiều
K = 4
center , label = main_kmeans(X,K)
SEE = 0
for i in range(len(X)) :
    diff = X[i] - center[label[i]]# nó như kiểu tính độ dài AB khi mà biết A(a,b) và B(x,y)
    SEE += np.sum(diff **2)
print("SEE =", SEE)

plt.scatter(X[:,0] ,X[:,1] , c = label , s= 1)
plt.scatter(center[:,0] , center[:,1] ,c='red', marker ='X' ,s = 200)
plt.show()
