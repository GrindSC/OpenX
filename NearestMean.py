import numpy as np

class NearestMean:
    def __init__(self):
        self.means = None

    def fit(self, X, y):
        y_classes = np.unique(y)
        self.means = [np.mean(X[y == target_class], axis=0) for target_class in y_classes]

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for idx,sample in enumerate(X):
            euclidean_distances = [np.sqrt(np.sum(np.square(sample - mean))) for mean in self.means]
            y_pred[idx]=np.argmin(euclidean_distances)
        return (np.array(y_pred) + 1).astype(int)