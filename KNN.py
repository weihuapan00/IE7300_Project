import numpy as np
from collections import Counter
class KNN:
    
    def __init__(self,k=3):
        self.k = k
    
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
        
    def predict(self,X):
        pred_y = [self._predict(x) for x in X]
        return np.array(pred_y) 
        
    
    def _predict(self,x):
        
        distances = [np.linalg.norm(x-x_train) for x_train in self.X_train]
        
        k_index = np.argsort(distances)[:self.k]
        
        k_nearest_label = [self.y_train[idx] for idx in k_index]
        
        return Counter(k_nearest_label)