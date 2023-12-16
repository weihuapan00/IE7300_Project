import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the feature
        self.left = left            # Left subtree (Node)
        self.right = right          # Right subtree (Node)
        self.value = value          # Predicted value (for leaf nodes)

# Define a basic decision tree regressor
class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or np.all(y == y[0]):
            return Node(value=np.mean(y))

        n_features = X.shape[1]
        m, n = X.shape
        best_mse = float('inf')
        best_feature, best_threshold = None, None
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = ~left_indices
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                mse = self._mean_squared_error(y[left_indices], y[right_indices])
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    best_threshold = threshold

        if best_mse == float('inf'):
            return Node(value=np.mean(y))

        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_tree, right=right_tree)

    def _mean_squared_error(self, left_labels, right_labels):
        m, n = len(left_labels), len(right_labels)
        mse_left = np.mean((left_labels - np.mean(left_labels)) ** 2)
        mse_right = np.mean((right_labels - np.mean(right_labels)) ** 2)
        return (m / (m + n)) * mse_left + (n / (m + n)) * mse_right

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)

# Create and fit the Random Forest regressor
class RandomForestRegressor:
    def __init__(self, n_estimators=10, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            X_sampled = X[sample_indices]
            y_sampled = y[sample_indices]

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_sampled, y_sampled)
            self.models.append(tree)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)