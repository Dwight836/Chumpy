# This is testing my Python replication of numpy, chumpy
from chumpy import Chumpy
from linreg import LinearRegression
from chump_linreg import ChumpLinearRegression
import numpy as np

import pandas as pd


def main():

    # Reads in data
    df_ad = pd.read_csv('Advertising.csv')
    x = df_ad.TV
    y = df_ad.sales

    # Temp numpy reshaping from Pandas series (low IQ fix)
    x = np.array(x)
    x = x.reshape(-1, 1)
    print(f'Shapes: {x.shape, y.shape}')

    chump_model = ChumpLinearRegression()
    chump_model.fit(x, y)
    print(f'Slope : {chump_model.weights[0]}')
    print(f'Intercept : {chump_model.bias}')


main()
