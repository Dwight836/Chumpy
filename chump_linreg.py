from chumpy import Chumpy as chump

class ChumpLinearRegression:

    def __init__(self, iterations=1000, learning_rate=0.001):

        self.iterations = iterations
        self.learning_rate = learning_rate

        self.num_features = 0
        self.num_rows = 0

        self.x = None
        self.y = None
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        self.x = x
        self.y = y

        # This assumes x is a 2D array
        self.num_features = len(x[0])
        print(f'Model num features initialized at {len(x[0])}')

        self.num_rows = len(x)
        print(f'Model num training examples: {len(x)}')

        self.weights = chump.create_zeros(self.num_features)
        print(f'Weights initialized')

        for i in range(self.iterations):
            self.update_weights()

        return self

    def predict(self, x):
        # Gets prediction
        weight_product = chump.dot_product(mat_a=x, mat_b=self.weights)
        scaled_weight_product = chump.sum_scalar(matrix=weight_product, scalar=self.bias)
        return scaled_weight_product

    def update_weights(self):


        y_pred = self.predict(self.x)

        x_transpose = chump.transpose_matrix(self.x)

        residuals = chump.sum_matrices(a=self.y, b=y_pred)





