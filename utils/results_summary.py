"""
results summary
"""

import pandas as pd
import pylab
from sklearn.metrics import mean_squared_error

models = ["darnn_best", "cnn_lstm_best", "cnn2d_best", "lstm_best"]

resultsPath = "./results/"
dataPath = "./nasdaq100/small/nasdaq100_padding.csv"

data = pd.read_csv(dataPath)
train_idx = len(data)*0.70
announce_idx = 2909

def plotNasdaqPrices(data, savePath="./nasdaq100/small/"):

    data.NDX.loc[:train_idx].plot(label="train")
    data.NDX.loc[train_idx:].plot(label="test", color="green")
    data.NDX.loc[len(data) - announce_idx:].plot(color="orange", label="post-announcement")
    pylab.ylabel("Nasdaq 100 price")
    pylab.legend()
    pylab.title("Nasdaq 100 prices 2016-07-26 to 2016-12-22")
    pylab.savefig(savePath + "nasdaq100_prices.png")

def getTestMSE(models, resultsPath):

    total = []
    pre = []
    post = []

    for model in models:
        ydf = pd.read_csv("%s/%s/price_predictions_%s.csv" \
                          %(resultsPath, model, model))
        total.append(mean_squared_error(ydf.true, ydf.pred))
        pre.append(mean_squared_error(ydf.true.loc[:announce_idx], ydf.pred.loc[:announce_idx]))
        post.append(mean_squared_error(ydf.true.loc[announce_idx:], ydf.pred.loc[announce_idx:]))

    mse = pd.DataFrame({"MSE":total, "preMSE":pre, "postMSE":post}, index=models)

    return mse


