# This is testing my Python replication of numpy, chumpy
from chumpy import Chumpy
from linreg import LinearRegression
from chump_linreg import ChumpLinearRegression
import numpy as np

import pandas as pd



# also from Geeks for Geeks... im a bot

def main():

    df_ad = pd.read_csv('Advertising.csv')
    print(df_ad.head())

    x = df_ad.TV
    y = df_ad.sales

    x = np.array(x)
    x = x.reshape(-1, 1)

    print(x.shape, y.shape)






    # This is the shape format that it needs to be in.
    #x = np.array([1, 2, 3, 4, 5])
    #x = x.reshape(-1, 1)
    #y = np.array([3, 5, 7, 9, 11])
    #print(len((x.shape)))

    print(f'Shapes: {x.shape, y.shape}')

    # Doing something wrong. Unsure what.
    model = LinearRegression()
    model.fit(x, y)
    print(f'Slope : {model.weights[0]}')
    print(f'Intercept : {model.bias}')
    # Demonstration of Chumpy Functionality
    chump = Chumpy()
    chump_model = ChumpLinearRegression()

    chump_model.fit(x, y)
    print(f'Slope : {chump_model.weights[0]}')
    print(f'Intercept : {chump_model.bias}')



main()
