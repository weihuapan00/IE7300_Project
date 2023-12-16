import numpy as np
import pandas as pd

class MyLogisticRegression:
    
    def __init__(self, epochs = 10000, threshold=1e-3,
                 regularization=None,alpha=0.01) -> None:
        
        # Initialize parameters and flags for the model.
        self.epochs = epochs
        self.threshold = threshold
        self.regularization = regularization
        self.alpha = alpha
    
    
    def train(self, X, y, batch_size=64, lr=1e-3, seed=11, verbose=False):
        """
        Train the model using stochastic gradient descent.
        """
        # Set seed for reproducibility.
        np.random.seed(seed) 
        
        # Define the unique classes and their corresponding indices.
        self.classes = np.unique(y)
        self.class_labels = {c: i for i, c in enumerate(self.classes)}
        
        # Add bias term to the features.
        X = self.add_bias(X)
        
        # Convert labels into one-hot encoded format.
        y = self.one_hot(y)
        
        # Initialize weights matrix with zeros.
        self.loss = []
        self.weights = np.zeros(shape=(len(self.classes), X.shape[1]))
        
        # Start the training process.
        self.fit_data(X, y, batch_size, lr, verbose)
        return self
    
    
    
    def fit_data(self, X, y, batch_size, lr, verbose):
        """
        Fit the data using stochastic gradient descent.
        """
        i = 0
        while (not self.epochs or i < self.epochs):
            # Compute and store the cross-entropy loss.
            self.loss.append(self.cross_entropy(y, self.predict_(X)))
            
            # Randomly select a batch of data.
            idx = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[idx], y[idx]
            
            # Calculate the error between predicted and true values.
            error = y_batch - self.predict_(X_batch)
            
            # Update the weights based on the error and learning rate.
            update = lr * np.dot(error.T, X_batch)

            # Apply regularization if specified.
            if self.regularization == 'Ridge':
                update += self.alpha * self.weights
            elif self.regularization == 'Lasso':
                update += self.alpha * np.sign(self.weights)
            elif self.regularization == 'Elastic Net':
                update_w += self.alpha * (self.weights + np.sign(self.weights))

            self.weights += update

            # Stop training if updates are smaller than a threshold.
            if np.abs(update).max() < self.threshold: 
                break
            
            # Print training accuracy every 1000 iterations if verbose is True.
            if i % 1000 == 0 and verbose: 
                print(' Training Accuracy at {} iterations is {}'.format(i, self.evaluate_(X, y)))
            i += 1
        
    def predict(self, X):
        return self.predict_(self.add_bias(X))
    
    def predict_(self, X):
        pre_vals = np.dot(X, self.weights.T).reshape(-1,len(self.classes))
        return self.softmax(pre_vals)
    
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1,1)

    def predict_classes(self, X):
        self.probs_ = self.predict(X)
        return np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))
  
    def add_bias(self,X):
        return np.insert(X, 0, 1, axis=1)
  
    def one_hot(self, y):
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]
    
    def score(self, X, y):
        '''
        Accuracy metric
        '''
        return round(np.mean(self.predict_classes(X).reshape(-1,1) == y),3)
    
    def evaluate_(self, X, y):
        return np.mean(np.argmax(self.predict_(X), axis=1) == np.argmax(y, axis=1))
    
    def cross_entropy(self, y, probs):
        return -1 * np.mean(y * np.log(probs))
    
    def confusion_matrix(self, actual, predicted,norm=False):
        """
        Compute the confusion matrix for the given actual and predicted outputs.

        Args:
        - actual (array-like): Actual outputs (ground truth).
        - predicted (array-like): Predicted outputs from the model.

        Returns:
        - matrix (np.ndarray): N x N confusion matrix, where N is the number of unique classes.
        """
      
        # Create an empty matrix
        matrix = np.zeros((len(self.classes), len(self.classes)), dtype=float)

        # Fill the matrix
        for i, true_class in enumerate(self.classes):
            for j, pred_class in enumerate(self.classes):
                matrix[i, j] = np.sum((actual == true_class) & (predicted == pred_class))

        if norm:
            for i in range(len(self.classes)):
                total = np.sum(matrix[i])
                for j in range(len(self.classes)):
                    matrix[i,j] = round(matrix[i,j] / total,2)
        
        matrix_df = pd.DataFrame(matrix, index=self.classes, columns=self.classes)
        return matrix_df
        