import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, kernel="linear",
                 degree=2, gamma=1.0):
        self.lr = learning_rate
        self.lambda_param = lambda_param # regularization
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.kernel = kernel
        self.degree = degree  # Degree for polynomial kernel
        self.gamma = gamma    # Gamma for RBF kernel

    def fit(self, X, y):
        self.X_train = X
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                if self.kernel == "linear":
                    condition = y_[idx] * (self.w.T @ x_i - self.b) >= 1
                elif self.kernel == "rbf":
                    condition = y_[idx] * (self._rbf_kernel(x_i, x_i) - self.b) >= 1
                elif self.kernel == "poly":
                    condition = y_[idx] * (self._poly_kernel(x_i, x_i) - self.b) >= 1

                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    if self.kernel == "linear":
                        self.w -= self.lr * (
                            2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                        )
                    elif self.kernel == "rbf":
                        self.w -= self.lr * (
                            2 * self.lambda_param * self.w - y_[idx] * self._rbf_kernel(x_i, x_i)
                        )
                    elif self.kernel == "poly":
                        self.w -= self.lr * (
                            2 * self.lambda_param * self.w - y_[idx] * self._poly_kernel(x_i, x_i)
                        )
                    self.b -= self.lr * y_[idx]

    def _rbf_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def _poly_kernel(self, x1, x2):
        return (np.dot(x1, x2) + 1) ** self.degree


    def predict(self, X):
        if self.kernel == "linear":
            approx = np.dot(X, self.w) - self.b
        elif self.kernel == "rbf":
            approx = np.array([self._predict_rbf(x) - self.b for x in X])
        elif self.kernel == "poly":
            approx = np.array([self._predict_poly(x) - self.b for x in X])
        self.noise_index = abs(approx) < 1
        return np.sign(approx)

    def _predict_rbf(self, x):
        return self._rbf_kernel(self.w.T, x)

    def _predict_poly(self, x):
        return self._poly_kernel(self.w.T, x)

