# This is an attempt to replicate numpy in Python
# This will likely end badly
# But I want to do it in Python to understand in Java
# Performance will likely be horrendous
# Dwight Mayer, May 25th, 2025
import numpy as np


class Chumpy:
    def __init__(self):
        pass

    @staticmethod
    def transpose_matrix(x):
        # 2D lists only for now...
        # I should probably create a bad matrix class here.
        n_rows = len(x)
        n_cols = len(x[0])

        # Create a new matrix with swapped dimensions, initialized with zeros
        transposed_matrix = [[0 for _ in range(n_rows)] for _ in range(n_cols)]

        for i in range(n_rows):
            for j in range(n_cols):
                transposed_matrix[j][i] = x[i][j]

        return transposed_matrix

    @staticmethod
    def dot_product(mat_a, mat_b):


        # This should cast them into 2D vectors properly
        if isinstance(mat_a[0], (int, float)):
            mat_a = [[elem] for elem in mat_a]  # Convert 1D row vector to column matrix

        if isinstance(mat_b[0], (int, float)):
            mat_b = [[elem] for elem in mat_b]  # Convert 1D vector to column matrix
        # This assumes that mat_a and mat_b are both 2D lists
        # assert that n_cols A = n_rows B or vice versa...
        a_rows = len(mat_a)
        a_cols = len(mat_a[0])

        print(f'Mat B: {mat_b}')

        b_rows = len(mat_b)


        # how to get the number of columns...
        #b_cols = len(mat_b[0])
        if type(mat_b[0]) == int:
            b_cols = 1
        else:
            b_cols = len(mat_b[0])

        assert a_cols == b_rows, 'Bad structure'

        # This creates a 2D list of 0s
        result = [[0 for _ in range(b_cols)] for _ in range(a_rows)]

        # This fills the matrix with dot product results!!!
        for i in range(a_rows):
            for j in range(b_cols):
                for k in range(a_cols):  # or b_rows
                    result[i][j] += mat_a[i][k] * mat_b[k][j]

        # Returns dot product of two matrices!
        return result

    @staticmethod
    def sum_matrices(a, b, operation='subtract'):

        if isinstance(a[0], (int, float)):
            a = [[elem] for elem in a]

        # a and b are matrices of equal size
        rows = len(a)
        cols = len(a[0])

        # THis creates a matrix the size of A.
        result = [[0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                if operation == 'subtract':
                    result[i][j] = (a[i][j] - b[i][j])
                elif operation == 'add':
                    result[i][j] = (a[i][j] + b[i][j])
                else:
                    raise ValueError("Unsupported operation: use 'add' or 'subtract'.")

        return result

    @staticmethod
    def sum_scalar(matrix, scalar):
        """A bad way of adding up scalars, to be honest"""
        # print(f'Scalar datatype: {type(scalar)}')
        # for each row
        for i in range(len(matrix)):
            # for each column cell
            for j in range((len(matrix[0]))):
                matrix[i][j] += scalar

        return matrix

    @staticmethod
    def create_zeros(n):
        return [0 for _ in range(n)]

























