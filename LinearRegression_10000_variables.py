import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(14)
m = 100000000
X = np.random.randint(150 ,201, size = (m,1)).astype(float)# viết thành cột sẵn rồi không cần T
y =0.55 * X -25 + np.random.randn(m, 1) * 8

one = np.ones((m ,1))
Xbar = np.concatenate((one ,X) ,axis =1)


print('Normal Equations')
start = time.time()
A = np.dot(Xbar.T , Xbar)
b = np.dot(Xbar.T , y)
w = np.dot(np.linalg.pinv(A) , b)

normal_time = time.time() - start
print(f'Time of normal equation : {normal_time:.5f}s')

print('Gradient Descent')
start = time.time()
w = np.random.randn(2, 1) * 0.01
lrn_rate = 0.00001
epoach = 1000
for i in range(epoach) :
    y_pred = np.dot(Xbar , w)
    loss = y_pred - y
    gradient =(1/m) * np.dot(Xbar.T , loss)
    w = w -lrn_rate * gradient
normal_time = time.time() - start
print(f'Time of gradient descent : {normal_time:.5f}s')

