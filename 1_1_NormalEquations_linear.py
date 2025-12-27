import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
N = 1000000
X = np.random.randint(160 , 190 , size = (N,1))
y = X * 0.5 +11
idx = np.random.permutation(N)
split = int(0.8*N)
train , test = idx[:split] , idx[split:]
X_train , y_train = X[train] , y[train]
X_test , y_test = X[test] , y[test]



one = np.ones((X_train.shape[0],1))
Xbar =np.concatenate((one,X_train),axis = 1)

A = Xbar.T @ Xbar
b = Xbar.T @ y_train
w = np.linalg.pinv(A) @ b

w_0 , w_1 = w[0,0] , w[1,0]
y_train_final = Xbar @ w
mse = np.mean((y_train_final-y_train)**2)
mae = np.mean(np.abs(y_train_final-y_train))


one_test = np.ones((X_test.shape[0],1))
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



plt.scatter(X_test[:500] , y_test[:500] , color = 'red' , label = 'Test')
x_line = np.array([[X.min()],[X.max()]])
y_line= w_0 + w_1 * x_line
plt.scatter(X_train[:500] , y_train[:500] , color = 'green' , label = 'Train')
plt.plot(x_line , y_line , label = 'Fit for train')
plt.xlabel('Chieu cao')
plt.ylabel('Can nang')
plt.legend()
plt.show()
