import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
N = 100000
X = np.random.randint(150 , 190 , size = (N,1))
y = 0.5 * X + 11

idx = np.random.permutation(N)
split = int(0.8*N)
train , test = idx[:split] , idx[split:]
X_train , y_train = X[train] , y[train]
X_test , y_test = X[test] , y[test]

one_train = np.ones((X_train.shape[0],1))
Xbar = np.concatenate((one_train , X_train), axis = 1)

w = np.random.randn(Xbar.shape[1],1) * 0.01 # do w phải có cùng số chiều với Xbar , dùng gradient descent bắt buộc phải có giá trị của w ban đầu , ở đây ta khời tạo w theo chuẩn N(0,1) 
lrn_rate = 1e-6
m = X_train.shape[0]
epochs = 10000000
for _ in range(epochs) :
    y_pred = X_train @ w
    loss = y_pred - y_train
    gradient = (1/m) * (Xbar.T @ loss)
    w = w - lrn_rate * gradient

w_0 , w_1 = w[0,0], w[1,0]
y_train_final = Xbar @ w
mse = np.mean((y_train_final-y_train)**2)
mae = np.mean(np.abs(y_train_final-y_train))


one_test = np.ones((X_test.shape[0],1))
Xbar_test = np.concatenate((one_test , X_test), axis = 1)
y_test_final = Xbar_test @ w

mse_test = np.mean((y_test_final-y_test)**2)
mae_test = np.mean(np.abs(y_test_final-y_test))

print(f'w0 = {w_0}')
print(f'w1 = {w_1}')
print(f'MSE = {mse}')
print(f'MAE = {mae}')
print(f'MSE test = {mse_test}')
print(f'MAE test = {mae_test}')

plt.scatter(X_test[:200] , y_test[:200] , color = 'red',label = 'Test')
x_line = np.array([[X.min()],[X.max()]])
y_line = w_0 + w_1 * x_line
plt.scatter(X_train[:200],y_train[:200],color = 'green',label = 'Train')
plt.plot(x_line , y_line , label = 'Fit for train')
plt.xlabel('Chieu cao')
plt.ylabel('Can nang')
plt.legend()
plt.show()
