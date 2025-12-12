import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
N = 1000
X = np.random.randint(150 , 190 , size = (N ,1))
y = 0.5 * X + 10

idx = np.random.permutation(N)
split = int(0.8 * N)
train , test = idx[:split], idx[split:]
X_train , y_train = X[train] , y[train]
X_test , y_test = X[test] ,y[test]

one = np.ones((X_train.shape[0],1))# Do phải gắn 1 vào để khi mà nhân thì mới có w0 , nên nó bắt buộc phải bằng số hàng
Xbar = np.concatenate((one,X_train),axis = 1)

w = np.random.randn(Xbar.shape[1],1) * 0.01 # do w phải có cùng số chiều với Xbar , dùng gradient descent bắt buộc phải có giá trị của w ban đầu , ở đây ta khời tạo w theo chuẩn N(0,1)
lrn_rate = 1e-6          #  GIẢM learning rate
epochs = 200000          # số vòng lặp vừa phải
m = X_train.shape[0]     # dùng số mẫu TRAIN, không phải toàn bộ X
lamda = 8.0       


for _ in range(epochs) :
    y_pred = Xbar @ w
    y_pred_1 = y_pred - y_train
    gradient_1 = (1/m) * (Xbar.T @ y_pred_1)
    gradient_2 = np.zeros_like(w)
    gradient_2[1:] = (lamda / m) * w[1:]# phạt từ w1 , không phạt w0(bias)
    gradient = gradient_1 + gradient_2
    w = w - lrn_rate * gradient

w_0 , w_1 = w[0,0] , w[1,0]

y_train_final = Xbar @ w
mse = np.mean((y_train_final-y_train)**2)
mae = np.mean(np.abs(y_train_final-y_train))

one_test = np.ones((X_test.shape[0],1))# Do phải gắn 1 vào để khi mà nhân thì mới có w0 , nên nó bắt buộc phải bằng số hàng
Xbar_test = np.concatenate((one_test,X_test),axis = 1)
y_test_final = Xbar_test @ w

mse_test = np.mean((y_test_final-y_test)**2)
mae_test = np.mean(np.abs(y_test_final-y_test))

print(f'w0 = {w_0}')
print(f'w1 = {w_1}')

print(f'MSE = {mse}')
print(f'MAE = {mae}')

print(f'MSE test = {mse_test}')
print(f'MAE test = {mae_test}')

plt.scatter(X_train[:200] , y_train[:200] , color = 'red',label = "Train")
x_line = np.array([[X.min()],[X.max()]])
y_line = w_0 + w_1 * x_line
plt.scatter(X_test[:200] , y_test[:200] , color = 'blue',label = "Test")
plt.plot(x_line , y_line , label = 'Fit for train')
plt.xlabel('Chieu cao')
plt.ylabel('Can nang')
plt.legend()
plt.show()



