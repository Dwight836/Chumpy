import numpy as np

# Some code taken from Geeks for Geeks. Thank you, geeks <3

class LinearRegression:
    def __init__(self, iterations=1000, learning_rate=0.001):

        self.iterations = iterations
        self.learning_rate = learning_rate

    def fit(self, x, y):
        # Fits the model to a given dataset
        # initializes weights

        # Don't really get what I'm doing here...
        self.n_rows = len(x)
        if len(x.shape) > 1:
            self.n_features = x.shape[1]
        elif len(x.shape) == 1:
            self.n_features = 1

        self.weights = np.zeros(self.n_features)

        # init bias, matrices...
        self.bias = 0
        self.x = x
        self.y = y

        # Updates weights in helper function
        for i in range(self.iterations):
            self.update_weights()

        return self

    def update_weights(self):
        # Updates weights
        y_pred = self.predict(self.x)

        # This applies gradient descent, the dot prod of predictions w data
        # Essentially gets ROC for derivatives
        dw = - (2 * (self.x.T).dot(self.y - y_pred)) / self.n_rows
        db = - 2 * np.sum(self.y - y_pred) / self.n_rows

        # Applies learning rate on weights, biases
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

        return self

    def predict(self, x):
        # Checks dimensions
        if x.ndim == 1:
            x = x.reshape(1, -1)
            # Broadcasts bias...
        return x.dot(self.weights) + self.bias

