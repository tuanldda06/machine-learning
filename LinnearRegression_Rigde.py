import numpy as np
import matplotlib.pyplot as plt

X = np.array([[174, 167, 189, 171, 173, 165, 187]]).T       
y = np.array([70, 65, 78, 62, 69, 76, 70]).reshape(-1, 1)   


one = np.ones((X.shape[0], 1))                              
Xbar = np.concatenate((one, X), axis=1)                     


w = np.random.randn(Xbar.shape[1], 1) * 0.01                
lrn_rate = 0.00001
epoach = 10000
m = Xbar.shape[0]
lambda_penalty = 10.0

for i in range(epoach):
    y_pred = np.dot(Xbar, w)                                
    error = y - y_pred
    gradient_mse = (-2 / m) * np.dot(Xbar.T, error)         
    gradient_l2 = np.zeros_like(w)
    gradient_l2[1:] = (2 * lambda_penalty / m) * w[1:]
    gradient = gradient_mse + gradient_l2
    w = w - lrn_rate * gradient

w_0, w_1 = w[0, 0], w[1, 0]
print(f"w_0 (bias) = {w_0:.4f}")
print(f"w_1 (slope) = {w_1:.6f}")

X_test = np.array([[160],[165], [170], [180], [185],[190]])
y_pred = w_0 + w_1 * X_test
plt.scatter(X_test, y_pred, color='green', label='Dự đoán mới')

plt.scatter(X, y, color='blue', label='Dữ liệu thực')
x_line = np.array([[X.min()], [X.max()]])
y_line = w_0 + w_1 * x_line
plt.plot(x_line, y_line, color='red', label=f'Ridge (λ={lambda_penalty})')
plt.xlabel('Chiều cao (cm)')
plt.ylabel('Cân nặng (kg)')
plt.title('Ridge Regression - Gradient Descent (np.dot)')
plt.legend()
plt.show()
