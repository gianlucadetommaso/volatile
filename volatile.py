#!/usr/bin/env python

import yfinance as yf
from clay import *
import numpy as np
import matplotlib.pyplot as plt


def load_data(tickers, start, end):
    num_stocks = len(tickers)
    stocks = yf.Tickers(tickers)

    sector_stocks = [stocks.tickers[i].info["sector"] for i in range(num_stocks)]
    sectors = np.unique(sector_stocks)
    sector_ids = [np.where(sectors == sec)[0][0] for sec in sector_stocks]

    df = stocks.download(start=start, end=end)
    log_prices = np.log(df["Close"].to_numpy()).T

    return dict(dates=df.index, sector_ids=sector_ids, log_prices=log_prices)


class StocksModel:
    def __init__(self, data: dict):
        self.log_prices = data["log_prices"]
        self.sector_ids = data["sector_ids"]
        self.num_stocks, self.t = self.log_prices.shape
        self.num_sectors = len(np.unique(self.sector_ids))

        self.order = 5

    def model(self):
        phi_s = Normal(scale=np.linspace(5 / (self.order + 1), 5, self.order + 1)[::-1],
                       shape=(self.num_sectors, self.order + 1), name="phi_s")
        phi = Normal(loc=embedding(self.sector_ids, phi_s), name="phi")
        psi_s = Normal(scale=5, shape=(self.num_sectors, 1), name="psi_s")
        psi = Normal(loc=embedding(self.sector_ids, psi_s), name="psi")

        tt = np.linspace(1 / self.t, 1, self.t) ** np.arange(self.order + 1).reshape(-1, 1)

        return Normal(loc=dot(phi, tt), scale=softplus(psi + 1 - tt[1]), shape=(self.num_stocks, self.t), name="y")

    def posterior(self):
        return self.model().distribution.observe(y=self.log_prices)

    def estimate_sales(self, MAP):
        llkd = self.model().distribution.observe(**MAP)
        return dict(y=llkd.mode()["y"], std=np.sqrt(llkd.variance()["y"]))

    def predict_sales(self, MAP):
        llkd = Normal(loc=dot(MAP["phi"], np.array([1 + 1 / self.t]) ** np.arange(self.order + 1).reshape(-1, 1)),
                      scale=softplus(MAP["psi"] + 1 / self.t),
                      shape=(self.num_stocks, 1), name="y").distribution
        return dict(y=llkd.mode()["y"], std=np.sqrt(llkd.variance()["y"]))


def main():
    tickers = ['TSLA', 'NCLH', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'DAL', 'GILD', 'IBM', 'RCL', 'UNM']
    train_data = load_data(tickers, start="2020-01-02", end="2020-09-09")
    test_data = load_data(tickers, start="2020-09-10", end="2020-09-10")
    #
    stocks_model = StocksModel(train_data)
    MAP = stocks_model.posterior().mode()
    est = stocks_model.estimate_sales(MAP)
    pred = stocks_model.predict_sales(MAP)

    q97_5 = 1.96
    plt.figure(figsize=(20, 12))
    num_columns = 3
    for i in range(stocks_model.num_stocks):
        plt.subplot(stocks_model.num_stocks // 3 + 1, num_columns, i + 1)
        plt.title(tickers[i], fontsize=15)
        plt.plot(train_data["dates"], train_data["log_prices"][i])
        plt.plot(train_data["dates"], est["y"][i])
        plt.fill_between(train_data["dates"], est["y"][i] - q97_5 * est["std"][i], est["y"][i] + q97_5 * est["std"][i],
                         alpha=0.2)
        plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 10))
    y_test = test_data["log_prices"]
    num_columns = 5
    for i in range(stocks_model.num_stocks):
        plt.subplot(stocks_model.num_stocks // num_columns + 1, num_columns, i + 1)
        plt.title(tickers[i], fontsize=15)

        plt.scatter(0, pred["y"][i], color='b', marker='o')
        plt.scatter(0, pred["y"][i] - q97_5 * pred["std"][i], color='b', marker='_')
        plt.scatter(0, pred["y"][i] + q97_5 * pred["std"][i], color='b', marker='_')
        plt.vlines(0, pred["y"][i] - q97_5 * pred["std"][i], pred["y"][i] + q97_5 * pred["std"][i], color='b')
        plt.scatter(0, y_test[i], color='r')
        plt.xlim([-1, 1])
        plt.xticks([], [])
        plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()





if __name__ == '__main__':
    main()

