"""
import data using morningstar since yahoo and google api deprecated

"""
import os
import numpy as np
import pandas as pd
from pandas_datareader import data

savePath = "./data/"
stockFile = "./SPX_050818.csv"
startDate = "2010-01-01"
endDate = "2018-05-01"

def getOhlc(ticker, startDate, endDate, source="morningstar"):
    """
    :param ticker: "AAPL"
    :param startDate: "2010-01-01"
    :param endDate: "2018-05-01"
    :param source: "morningstar", "yahoo", "google"
    :param savePath: 
    :return: 
    """

    df =  data.DataReader(ticker, source, startDate, endDate)

    return df

# get constituents per day
def getConstituentList(stockFile=stockFile):

    stocks = pd.read_csv(stockFile)
    stocks.dropna(0, how="all", inplace=True)
    stocks["ticker"] = [x.split(" ")[0] for x in stocks.BbergTicker]

    return stocks

# create
skipTickers = ["ADK", "ANDV", "BF/B", "BRK/B", "BF.B", "BRK.B", "BHF", "BKNG",
               "CBRE", "DWDP"]
def getPrices(startDate=startDate, endDate=endDate, stockFile=stockFile,
              savePath=savePath, skipTickers=skipTickers):

    stocks = getConstituentList(stockFile=stockFile)

    for ticker in stocks.ticker[stocks.ExitDate != stocks.ExitDate]:
        if ticker not in skipTickers:
            print(ticker)
            if not checkForTicker(ticker):
                try:
                    pxs = getOhlc(ticker, "2010-01-01", "2018-05-01", source="morningstar")
                    pxs.to_csv("%s/%s_ohlc.csv" %(savePath, ticker))
                    print("%s saved" %ticker)
                except:
                    print("canʻt download ticker %s" %(ticker))
            else:
                print("%s_ohlc.csv exists" %(ticker))



def checkForTicker(ticker):

    files = os.listdir("./data/")
    if "%s_ohlc.csv" %ticker in files:
        return True
    else:
        return False