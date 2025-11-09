import numpy as np
import matplotlib.pyplot as plt

X = np.array([[178, 167,189,174,173,165,187,189 ,190 ,167]]).T
y = np.array([70 ,65,78,62,65,76,59,76,67,78])

one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one , X) , axis =1)

A = np.dot(Xbar.T , Xbar)
b = np.dot(Xbar.T , y)
w = np.dot(np.linalg.pinv(A) , b )

w_0 , w_1 = w[0], w[1]# dùng shape kiểm tra xem kết quả là 1D hay 2D
plt.scatter(X , y , color ='blue' , label = 'Dữ liệu thật')
x_line = np.array([[X.min()] , [X.max()]])  
y_line = w_0 + w_1 * x_line

X_test = np.array([[160],[168], [170], [180], [185],[190]])
y_pred = w_0 + w_1 * X_test
plt.scatter(X_test, y_pred, color='green', label='Dự đoán mới')

plt.plot(x_line , y_line , color = 'red' , label = 'Du doan')
plt.xlabel('Chieu cao')
plt.ylabel('Can nang')
plt.title('Linear Regression : Normal Equation')
plt.legend()
plt.show()