import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model
from sklearn.linear_model import SGDRegressor, Ridge

np.random.seed(42)

X = 2*np.random.rand(100, 1)
y = 4+3*X + np.random.rand(100, 1)

# linear regression
plt.scatter(X, y)

X_b = np.c_[np.ones((100, 1)), X]

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)
x_hat = np.arange(0, 2, 0.2)
y_hat = theta_best[0]+x_hat*theta_best[1]
plt.plot(x_hat, y_hat)
plt.show()

linReg = linear_model.LinearRegression()
linReg.fit(X, y)
predit = linReg.intercept_, linReg.coef_
print(predit)

# Batch gradient descend

eta = 0.1  # learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2,1)

for iteration in range(n_iterations):
    MSE = 2.0/m*X_b.T.dot(X_b.dot(theta)-y)
    theta = theta-eta*MSE

print(theta)

# Stochastic Gradient Descend

SGD = SGDRegressor(n_iter=1000, penalty=None, eta0=0.1)
SGD.fit(X, y.ravel())
print("SGD", SGD.intercept_, SGD.coef_)

# Ridge Regression Cost Function

ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
print("Ridge", ridge_reg.intercept_, ridge_reg.coef_)

sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(X, y.ravel())
print(sgd_reg.intercept_, sgd_reg.coef_)