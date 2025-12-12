import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
N = 1000000
X = np.random.randint(150 , 190 , size = (N , 1))  # số nguyên trong khoảng [a, b)
y = X * 0.4 + 11

# Chia train / test
idx = np.random.permutation(N)
split = int(N*0.8)
train , test = idx[:split] , idx[split:]
X_train , y_train = X[train] , y[train]
X_test  , y_test  = X[test]  , y[test]

# Thêm cột 1 để học w0 (bias)
one = np.ones((X_train.shape[0],1))
Xbar = np.concatenate((one , X_train), axis=1)

# Khởi tạo w nhỏ quanh 0
w = np.random.randn(Xbar.shape[1],1) * 0.01   # np.random.randn ~ phân phối chuẩn N(0,1)
lrn_rate = 5e-6                               # không chuẩn hóa thì để lr nhỏ cho an toàn
m = X_train.shape[0]
epochs = 5000000
lamda = 8.0                                   # hệ số phạt L1

for _ in range(epochs):
    y_pred = Xbar @ w               # dự đoán
    loss = y_pred - y_train         # sai số

    # gradient MSE
    gradient_1 = (1/m) * (Xbar.T @ loss)

    # gradient L1: lamda * sign(w), KHÔNG phạt w0
    gradient_2 = np.zeros_like(w)
    gradient_2[1:] = (lamda/m) * np.sign(w[1:])

    # tổng gradient
    gradient = gradient_1 + gradient_2

    # cập nhật w
    w = w - lrn_rate * gradient

w_0 , w_1 = w[0,0] , w[1,0]

# Train error
y_train_final = Xbar @ w
mse = np.mean((y_train_final - y_train)**2)
mae = np.mean(np.abs(y_train_final - y_train))

# Test error
one_test = np.ones((X_test.shape[0],1))
Xbar_test = np.concatenate((one_test , X_test), axis=1)
y_test_final = Xbar_test @ w

mse_test = np.mean((y_test_final - y_test)**2)
mae_test = np.mean(np.abs(y_test_final - y_test))

print(f"w0 = {w_0}")
print(f"w1 = {w_1}")
print(f"MSE train = {mse}")
print(f"MAE train = {mae}")
print(f"MSE test  = {mse_test}")
print(f"MAE test  = {mae_test}")

# Vẽ dữ liệu GỐC + đường fit
plt.scatter(X_test[:100],  y_test[:100] ,  color='green', label='Test')
plt.scatter(X_train[:100], y_train[:100], color='red',   label='Train')

x_line = np.array([[X.min()], [X.max()]])   # chiều cao thật (cm)
y_line = w_0 + w_1 * x_line                # mô hình y = w0 + w1 * X (vì không chuẩn hóa)

plt.plot(x_line, y_line, label='Fit for train')
plt.xlabel('Chieu cao')
plt.ylabel('Can nang')
plt.legend()
plt.show()

