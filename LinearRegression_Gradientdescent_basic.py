import numpy as np
import matplotlib.pyplot as plt

X = np.array([[174, 167,189,171,173,165,187]]).T
y = np.array([70 ,65,78,62,69,76,70]).reshape(-1, 1)

one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one , X) , axis =1)

w = np.random.randn(Xbar.shape[1], 1) * 0.01
lrn_rate = 0.00001
epoach = 10000
m = Xbar.shape[0]
for i in range(epoach) :
    y_pred = np.dot(Xbar, w)
    loss = y_pred - y
    gradient = (1/m) * np.dot(Xbar.T , loss)
    w = w - lrn_rate * gradient

w_0 , w_1 = w[0,0] , w[1,0] # do ket qua la mang 2 chieu

plt.scatter(X , y , color = 'blue' , label = 'Du lieu that')
x_line = np.array([[X.min()] , [X.max()]])
y_line = w_0 + w_1 * x_line
plt.plot(x_line , y_line , color = 'green' , label = 'Du lieu huan luyen')
plt.xlabel('Chieu cao')
plt.ylabel('Can nang')
plt.title('Linear Regression: Gradient Descent')
plt.legend()
plt.show()