'''
K Nearest Neighbors algorithm from scratch
'''

import numpy as np
from collections import Counter

def euclidean_distance(x1,x2):
  return np.sqrt(np.sum((x1 - x2)**2))
           
class KNN:
  def __init__(self, k=3):
    self.k = k
  
  def fit(self, X, y):
    self.X_train = X
    self.y_train = y
  
  def predict(self, X):
    y_pred = [self._predict(x) for x in X]
    return np.array(y_pred)
    
  def _predict(self, x):
    #calculate distances between x and all the training samples
    distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

    #sort by distances and return indices of first k neighbors
    k_idx = np.argsort(distances)[: self.k]

    #get the labels of first k nearest neighbors
    k_neighbor_labels = [self.y_train[i] for i in k_idx]

    #get the labels of first k nearest neighbors
    most_common = Counter(k_neighbor_labels).most_common(1)
    return most_common[0][0]


if __name__ == "__main__":
    # Importing libraries
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
      accuracy = np.sum(y_true == y_pred) / len(y_true)
      return accuracy

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    model = KNN(k = 3)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("KNN classification accuracy", accuracy(y_test, predictions))
