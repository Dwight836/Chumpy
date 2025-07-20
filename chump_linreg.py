from chumpy import Chumpy as chump


class ChumpLinearRegression:

    def __init__(self, iterations=10000, learning_rate=0.00001):

        self.iterations = iterations
        self.learning_rate = learning_rate

        self.num_features = 0
        self.num_rows = 0

        self.x = None
        self.y = None
        self.weights = None
        self.bias = 0

    def fit(self, x, y):
        """Runs the algorithm to fit the model :)  """
        self.x = x
        self.y = y

        # This assumes x is a 2D array
        self.num_features = len(x[0])
        self.num_rows = len(x)
        self.weights = chump.create_zeros(n=self.num_features)

        if False:
            print(f'Model num features initialized at {len(x[0])}')
            print(f'Model num training examples: {len(x)}')
            print(f'Weights initialized')

        for i in range(self.iterations):
            self.update_weights()
            if i % 100 == 0:
                print(f'Completed Loop #{i}')

        return self

    def predict(self, x):
        # Gets prediction
        weight_product = chump.dot_product(mat_a=x, mat_b=self.weights)
        scaled_weight_product = chump.sum_scalar(matrix=weight_product, scalar=self.bias)
        return scaled_weight_product

    def update_weights(self):
        """This performs one weight update step"""

        # Gets residuals and x transpose
        y_pred = self.predict(self.x)
        x_transpose = chump.transpose_matrix(x=self.x)
        residuals = chump.sum_matrices(a=self.y, b=y_pred, operation='subtract')

        # Gets derivatives
        weights_dot = chump.dot_product(mat_a=x_transpose, mat_b=residuals)
        dw = [[-((2 * val) / self.num_rows) for val in row] for row in weights_dot]
        flat_residuals = [val for sublist in residuals for val in sublist]
        db = -2 * (sum(flat_residuals) / self.num_rows)

        # Applies learning rate on weights, biases
        scaled_dw = chump.scale_matrix(matrix=dw, scalar=self.learning_rate)
        self.bias -= (self.learning_rate * db)
        self.weights = chump.sum_matrices(a=self.weights, b=scaled_dw, operation='subtract')

        return self
