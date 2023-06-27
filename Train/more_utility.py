import os
import math
import time as tm
import tensorflow as tf
from tensorflow import keras  # tf.keras
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt


def standardize_2(Candles, logarithmic=False):
    Standard = []

    for market in range(Candles.shape[1]):
        subStandard = []
        for field in range(Candles.shape[2]):
            if logarithmic:
                nzPs = np.where( Candles[:, market, field] != 0.0 ) [0]
            else:
                nzPs = np.array(range(Candles.shape[0]))
            mu = np.average(Candles[nzPs, market, field])
            sigma = np.std(Candles[nzPs, market, field])
            standardized = (Candles[nzPs, market, field] - mu) / (sigma + 1e-15)
            subStandard.append( (market, field, mu, sigma) )
            assert standardized.dtype == Candles.dtype
            Candles[nzPs, market, field] = standardized
        Standard.append(subStandard)
    Standard = np.array(Standard)
    return Candles, Standard


def inverse_standardize_2(Candles, Standard, target_markets, chosen_field_y, logarithmic=False):
    epsilon = 0.005
    for market in target_markets:
        for field in chosen_field_y:
            _market, _field, mu, sigma = Standard[market, field]
            assert _market == market
            assert _field == field
            if logarithmic:
                # This is a math/logic error. Lucky, logarithmic is False for us.
                nzPs = np.where( np.abs(Candles[:, market, field]) >= epsilon ) [0]
            else:
                nzPs = np.array(range(Candles.shape[0]))
            Candles[nzPs, market, field] = Candles[nzPs, market, field] * sigma + mu           

    return Candles


from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial


def predict_eventFree(x, y, x_new, title, plot=False):
    poly = lagrange(x, y)
    y_new = Polynomial(poly.coef[::-1])(x_new)
    if plot:
        linewidth = 1.0
        plt.plot(x, y, label='known', linewidth=linewidth)
        plt.plot(x_new, y_new, label='extra', linewidth=linewidth)
        plt.legend(loc='center')
        title = "Predicted Event-Free" + ": {}".format("" if title is None else title)
        plt.title(title)
        plt.show()

def get_formed_data( Candles, CandleMarks, all_market_names, all_field_names, 
        min_true_candle_percent_x, chosen_fields_names_x, min_true_candle_percent_y, chosen_fields_names_y,
        target_market_names, tarket_market_top_percent
    ):
    pass
    return \
        Candles, CandleMarks, all_market_names, x_indices, y_indices, \
        chosen_market_names_x, chosen_field_names_x, chosen_market_names_y, chosen_field_names_y, \
        chosen_market_names, chosen_field_names, \
        target_market_names, target_markets

def infer(model, Candles,  Standard,  target_ts):
    
