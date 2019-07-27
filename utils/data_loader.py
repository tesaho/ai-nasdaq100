"""
pre-process and load csv stock data to pytorch tensor
"""

import torch
import pandas as pd
import numpy as np

index_file = "./nasdaq100/small/nasdaq100_padding.csv"
stock_file = "./data/AAPL_ohlc.csv"
featureNames = ["Close", "High", "Open", "Low", "Volume"]

def getXyFromStock(startDate=None, endDate=None, stockFile=stock_file, featureNames=featureNames):

    df = pd.read_csv(stockFile)
    df.set_index("Date", inplace=True)
    # forward 1 days log return
    df["y"] = np.log(df.Close/df.Close.shift(1)).shift(-1)

    # create X matrix
    if startDate != None:
        if endDate != None:
            dX = df[(df.index >= startDate) & (df.index <=endDate)]
        else:
            dX = df[df.index >= startDate]
    else:
        dX = df.copy()

    if featureNames != None:
        X = dX[featureNames].as_matrix()
    else:
        X = dX.as_matrix()
    y = dX.y.as_matrix().reshape((-1, 1))

    return X, y


def getXyFromConstituents(indexFile=index_file, debug=False):
    """
    used for LSTM, DA-RNN
    indexFile: nasdaq100 file
    debug: if True, import first 100 lines of data
    """

    data = pd.read_csv(indexFile, nrows=100 if debug else None)
    X_cols = [x for x in data.columns if x != "NDX"]
    # X (numDates, numStocks)
    # X = data[X_cols].as_matrix().reshape((len(data), 1, -1))
    X = data[X_cols].as_matrix().reshape((len(data), -1))
    # y (numDates, 1)
    y = data["NDX"].values.reshape((-1, 1))

    return X, y



def splitTrainTestTest(X, y, test_split_pct, val_split_pct, normMean=True):
    """
    X: original X data
    y: original y data
    test_split_pct: % of data used for test set
    val_split_pct: % of train data used for val set
    normMean: if True, normalize data using train_mean (subtract train_mean from each set)
    """
    test_idx = int(len(X) * test_split_pct)
    val_idx = int(len(X) * (1 - test_split_pct) * val_split_pct)

    train_X = X[:-test_idx - val_idx, ]
    train_y = y[:-test_idx - val_idx, ]
    print("train_X shape ", train_X.shape)
    print("train_y shape ", train_y.shape)

    val_X = X[-test_idx - val_idx:-test_idx, ]
    val_y = y[-test_idx - val_idx:-test_idx, ]
    print("val_X shape ", val_X.shape)
    print("val_y shape ", val_y.shape)

    test_X = X[-test_idx:, ]
    test_y = y[-test_idx:, ]
    print("test_X shape ", test_X.shape)
    print("test_y shape ", test_y.shape)

    ### normalize data
    if normMean:
        train_X_mean = np.mean(train_X, axis=0)
        train_y_mean = np.mean(train_y, axis=0)
        train_X -= train_X_mean
        train_y -= train_y_mean
        val_X -= train_X_mean
        val_y -= train_y_mean
        test_X -= train_X_mean
        test_y -= train_X_mean

    if val_split_pct > 0:
        return (train_X, train_y, val_X, val_y, test_X, test_y)
    else:
        return (train_X, train_y, test_X, test_y)