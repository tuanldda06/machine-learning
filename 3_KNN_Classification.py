import numpy as np
from collections import Counter 

def euclid_distance(x1,x2) :
    return np.sqrt(np.sum((x1-x2)**2))

def knn_classification(x_train , y_train , x_test , k = 4) :
    distance = []
    for i in range(len(x_train)) :
        dist = euclid_distance(x_train[i],x_test)
        distance.append((dist , y_train[i])) # append 1 tuple (khoảng cách, nhãn) nên cần 2 dấu ()
    distance.sort(key=lambda x : x[0])
    neighbors = distance[:k]

    output = [neighbor[1] for neighbor in neighbors]
    predict = Counter(output).most_common(1)[0][0]
#counter sẽ đi đếm số lần xuất hiện của từng nhãn trong output , most_common(1) thì trả về danh sách 1 phần tử, phần tử đó là cặp (nhãn, số_lần_đếm) xuất hiện nhiều nhất ,  [0] : lấy phần tử đầu tiên trong danh sách và [0] tiếp theo: lấy phần tử đầu tiên trong tuple
    return predict

def accuracy(x_train , y_train , x_test , y_test , k = 4) :
    correct = 0
    for i in range(len(x_train)) :
        predict = knn_classification(x_train , y_train , x_test , k )
        if predict == y_test[i] :
            correct += 1
    return correct/len(x_train)

np.random.seed(10)
N = 10000
X = np.random.randint(0,101,size = (N,2)) # ở đây kiểu dữ liệu sẽ là 2 chiều , như kiểu một cột là điểm toán , cột còn lại là điểm văn
def make_label(x) :
    s = x[0] + x[1]
    if s 