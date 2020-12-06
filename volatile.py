#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import csv
import os.path
import pandas as pd

import yfinance as yf
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


def load_data(tickers: list):
    """
    Load relevant information from provided tickers.

    Parameters
    ----------
    tickers: list 
        Stock symbols

    Returns
    -------
    Dictionary including:
        - tickers: list of symbols with available information;
        - dates: dates corresponding to available prices
        - sectors: sectors name corresponding to `tickers`
        - sector_id: integers identifying sectors of corresponding `tickers`
        - logp: log-prices corresponding to `tickers`
    """
    # make tickers unique
    tickers = list(set(tickers))
    # download all last year available closing prices
    df = yf.download(tickers, period="1y")['Close']
    # fix inconsistency if only one stock is loaded
    if df.ndim == 1:
        df = pd.DataFrame(df).rename(columns={"Close": tickers[0]})
    # drop stocks that have NaN in at least half of the period
    df.drop(columns=df.columns[np.where((df.isnull().sum(0) > df.shape[0] // 2 + 1) == True)[0]], inplace=True)
    # raise exception if no stock is left
    if df.size == 0:
        raise Exception("No symbol with full information is available.")
    # propagate data backwards to fill NaNs, then forward, then drop possible duplicated dates
    df = df.fillna(method='bfill').fillna(method='ffill').drop_duplicates()
    # print out unavailable symbols
    missing_tickers = [tick for tick in tickers if tick not in df.columns]
    if len(missing_tickers) > 0:
        print('\nRemoving {} from list of symbols because yfinance could not provide full information.'.format(
            missing_tickers))
    # reset list of tickers and stocks
    tickers = list(df.columns)
    stocks = yf.Tickers(tickers)
    # store log-prices
    logp = np.log(df.to_numpy().T)

    filename = "stock_info.csv"
    print('\nAccessing stock information. For all symbols that you download for the first time, this can take a '
          'while. Otherwise, stock information is cached into ' + filename + ' and accessing it will be fast.')

    if not os.path.exists(filename):
        # create a .csv to store stock information
        with open(filename, 'w') as file:
            wr = csv.writer(file)
            for row in zip(["SYMBOL"], ["SECTOR"]):
                wr.writerow(row)
    # load stock information file
    stock_info = pd.read_csv(filename)

    # load sector information. If already available in stock information file, load it from there. Otherwise, try out if
    # it is available. If not, give a unique name to the sector of the corresponding stock.
    sector_name = []
    missing_sector_info = {}
    for i in range(len(tickers)):
        idx = np.where(stock_info["SYMBOL"].values == tickers[i])[0]
        if len(idx) > 0:
            sector_name.append(stock_info["SECTOR"][idx[0]])
        else:
            try:
                sector_name.append(stocks.tickers[i].info["sector"])
                missing_sector_info[tickers[i]] = sector_name[-1]
            except:
                sector_name.append("NA" + str(i))

    # cache sector information that was not present before, except for names that were given artificially.
    sectors = np.unique(sector_name)
    sector_id = [np.where(sectors == sector)[0][0] for sector in sector_name]
    stock_info = zip(list(missing_sector_info.keys()), list(missing_sector_info.values()))
    with open(filename, 'a+', newline='') as file:
        wr = csv.writer(file)
        for row in stock_info:
            wr.writerow(row)

    return dict(tickers=tickers, dates=pd.to_datetime(df.index).date, sectors=sectors, sector_id=sector_id, logp=logp)

def define_model(tt: np.array, order_scale: np.array):
    """
    Define and return graphical model.

    Parameters
    ----------
    tt: np.array
        Time sequence corresponding to dates in the data. It is sed to construct polynomial model and reweigh likelihood 
        scale.
    order_scale: np.array
        It reweighs prior scales of parameters at different orders of the polynomial.
    """
    return tfd.JointDistributionSequential([
           # phi_s
           tfd.Independent(tfd.Normal(loc=tf.zeros([num_sectors, order + 1]), scale=order_scale), 2),
           # phi
           lambda phi_s: tfd.Independent(tfd.Normal(loc=tf.gather(phi_s, data["sector_id"], axis=0),
                                                    scale=0.5 * order_scale), 2),
           # psi_s
           tfd.Independent(tfd.Normal(loc=0, scale=tf.ones([num_sectors, 1])), 2),
           # psi
           lambda psi_s: tfd.Independent(tfd.Normal(loc=tf.gather(psi_s, data["sector_id"], axis=0), scale=0.5), 2),
           # y
           lambda psi, psi_s, phi: tfd.Independent(tfd.Normal(loc=tf.tensordot(phi, tt, axes=1),
                                                              scale=tf.math.softplus(psi + 1 - tt[1])), 2)])

def training(phi_s, phi, psi_s, psi, model, logp, learning_rate = 0.01, num_steps=10000):
    """
    It performs optimization over the model parameters via Adam optimizer.

    Parameters
    ----------
    phi_s: tf.Tensor
        Initial values of sector-level polynomial parameters.
    phi: tf.Tensor
        Initial values of stock-level polynomial parameters.
    psi_s: tf.Tensor
        Initial values of sector-level likelihood scale parameters.
    psi: tf.Tensor
        Initial values of stock-level likelihood scale parameters.
    model: tfd.JointDistributionSequential
        Graphical model to train.
    logp: np.array
        Log-price stock information.
    learning_rate: float
        Adam's fixed learning rate.
    num_steps: int
        Adam's fixed number of iterations.

    Returns
    -------
    It returns trained parameters.
    """
    log_posterior = lambda phi_s, phi, psi_s, psi: model.log_prob([phi_s, phi, psi_s, psi, logp])
    loss = tfp.math.minimize(lambda: -log_posterior(phi_s, phi, psi_s, psi),
                             optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                             num_steps=num_steps)
    return phi_s, phi, psi_s, psi


if __name__ == '__main__':
    cli = ArgumentParser('Volatile: your day-to-day trading companion.',
                         formatter_class=ArgumentDefaultsHelpFormatter)
    cli.add_argument('-s', '--symbols', type=str, nargs='+', help='List of symbols.')
    cli.add_argument('-pe', '--plot-estimation', type=bool, default=True,
                     help='Plot estimates and uncertainty between start and current date.')
    cli.add_argument('-spt', '--save-prediction-table', type=bool, default=False,
                     help='Save prediction table in csv format.')
    args = cli.parse_args()

    today = dt.date.today().strftime("%Y-%m-%d")

    print('\nDownloading all available closing prices in the last year...')
    data = load_data(args.symbols)
    tickers = data["tickers"]
    num_sectors = len(set(data["sector_id"]))
    num_stocks = data["logp"].shape[0]

    # how many days to look ahead when comparing the current price against a prediction
    horizon = 5
    # order of the polynomial
    order = 7

    print("\nTraining the model...")

    # number of trading dates in the data
    t = data["logp"].shape[1]
    # times corresponding to trading dates in the data
    tt = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
    # reweighing factors for parameters corresponding to different orders of the polynomial
    order_scale = np.linspace(1 / (order + 1), 1, order + 1)[::-1].astype('float32')[None, :]
    # prediction times up to horizon
    tt_pred = np.arange(1 + horizon) / t

    # training is attempted up to twice
    attempt = 0
    while attempt < 2:
        # train
        model = define_model(tt, order_scale)
        phi_s, phi, psi_s, psi = (tf.Variable(model.sample()[:-1][i]) for i in range(4))
        phi_s, phi, psi_s, psi = training(phi_s, phi, psi_s, psi, model, data["logp"])
        # calculate estimators of log-prices
        logp_est = np.dot(phi.numpy(), tt)
        std_logp_est = np.log(1 + np.exp(psi.numpy() + 1 - tt[1]))
        # calculate estimators of prices
        p_est = np.exp(logp_est + std_logp_est ** 2 / 2)
        std_p_est = np.sqrt(np.exp(2 * logp_est + std_logp_est ** 2) * (np.exp(std_logp_est ** 2) - 1))
        # calculate predictions of log-prices
        logp_pred = np.dot(phi.numpy(), np.array([1 + tt_pred]) ** np.arange(order + 1)[:, None])
        std_logp_pred = np.log(1 + np.exp(psi.numpy() + tt_pred))
        # calculate prediction of prices
        p_pred = np.exp(logp_pred + std_logp_pred ** 2 / 2)
        std_p_pred = np.sqrt(np.exp(2 * logp_pred + std_logp_pred ** 2) * (np.exp(std_logp_pred ** 2) - 1))
        # determine score
        scores = ((logp_pred[:, horizon] - data["logp"][:, -1]) / std_logp_pred[:, horizon])
        # retrain if at least one prediction is 5 standard deviations away the current price
        if np.max(np.abs(scores) > 5):
            attempt += 1
        else:
            break

    print("Training completed.")

    # rank according to score
    rank = np.argsort(scores)[::-1]
    ranked_tickers = np.array(tickers)[rank]
    ranked_scores = scores[rank]
    ranked_p = np.exp(data["logp"])[rank]
    ranked_p_est = p_est[rank]
    ranked_std_p_est = std_p_est[rank]
    ranked_p_pred = p_pred[rank]
    ranked_std_p_pred = std_p_pred[rank]
    # calculate sector estimators of log-prices
    logp_sec_est = np.dot(phi_s.numpy(), tt)
    std_logp_sec_est = np.log(1 + np.exp(psi_s.numpy() + 1 - tt[1]))
    # calculate sector estimators of prices
    p_sec_est = np.exp(logp_sec_est + std_logp_sec_est ** 2 / 2)
    std_p_sec_est = np.sqrt(np.exp(2 * logp_sec_est + std_logp_sec_est ** 2) * (np.exp(std_logp_sec_est ** 2) - 1))

    # stock thresholds
    st = {"HIGHLY BELOW TREND": 3, "BELOW TREND": 2, "ALONG TREND": 0, "ABOVE TREND": -2, "HIGHLY ABOVE TREND": -3}
    # stock information
    si = {"HIGHLY BELOW TREND": np.where(ranked_scores > st["HIGHLY BELOW TREND"])[0],
              "BELOW TREND": np.where((ranked_scores <= st["HIGHLY BELOW TREND"]) & (ranked_scores > st["BELOW TREND"]))[0],
              "ALONG TREND": np.where((ranked_scores <= st["BELOW TREND"]) & (ranked_scores > st["ABOVE TREND"]))[0],
              "ABOVE TREND": np.where((ranked_scores <= st["ABOVE TREND"]) & (ranked_scores > st["HIGHLY ABOVE TREND"]))[0],
              "HIGHLY ABOVE TREND": np.where(ranked_scores <= st["HIGHLY ABOVE TREND"])[0]}
    si = {k: v[0] for k, v in si.items() if len(v) > 0}
    # rate all stocks
    ranked_rating = np.array(list(si.keys())).repeat(list(np.diff(list(si.values()))) + [num_stocks - list(si.values())[-1]]).tolist()

    if args.plot_estimation:
        num_columns = 3
        print('\nPlotting sector estimation...')
        # determine which sectors were not available to avoid plotting
        NA_sectors = np.where(np.array([sec[:2] for sec in data["sectors"]]) == "NA")[0]
        num_NA_sectors = len(NA_sectors)

        # plot sectors
        left_sec_est = np.maximum(0, p_sec_est - std_p_sec_est)
        right_sec_est = p_sec_est + std_p_sec_est
        fig = plt.figure(figsize=(20, max(num_sectors - num_NA_sectors, 5)))
        j = 0
        for i in range(num_sectors):
            if i not in NA_sectors:
                j += 1
                plt.subplot(int(np.ceil((num_sectors - num_NA_sectors) / num_columns)), num_columns, j)
                plt.title(data["sectors"][i], fontsize=15)
                plt.plot(data["dates"], p_sec_est[i], label="sector estimation", color="C1")
                plt.fill_between(data["dates"], left_sec_est[i], right_sec_est[i], alpha=0.2, label="+/- 1 st. dev.", color="C0")
                plt.yticks(fontsize=12)
                plt.xticks(rotation=45)
                plt.legend(loc="upper left")
        plt.tight_layout()
        fig_name = 'sector_estimation.png'
        fig.savefig(fig_name, dpi=fig.dpi)
        print('Sector estimation plot has been saved in this directory as {}.'.format(fig_name))

        print('\nPlotting stock estimation...')
        # determine which stocks are along trend to avoid plotting
        along_trend = np.where(np.array(ranked_rating) == "ALONG TREND")[0]
        num_out_trend = num_stocks - len(along_trend)

        # plot stocks
        ranked_left_est = np.maximum(0, ranked_p_est - 2 * ranked_std_p_est)
        ranked_right_est = ranked_p_est + 2 * ranked_std_p_est
        if num_out_trend > 0:
            j = 0
            fig = plt.figure(figsize=(20, max(num_out_trend, 5)))
            for i in range(num_stocks):
                if i not in along_trend:
                    j += 1
                    plt.subplot(int(np.ceil(num_out_trend / num_columns)), num_columns, j)
                    plt.title(ranked_tickers[i], fontsize=15)
                    plt.plot(data["dates"], ranked_p[i], label="data")
                    plt.plot(data["dates"], ranked_p_est[i], label="stock estimation")
                    plt.fill_between(data["dates"], ranked_left_est[i], ranked_right_est[i], alpha=0.2, label="+/- 2 st. dev.")
                    plt.yticks(fontsize=12)
                    plt.xticks(rotation=45)
                    plt.legend(loc="upper left")
            plt.tight_layout()
            fig_name = 'stock_estimation.png'
            fig.savefig(fig_name, dpi=fig.dpi)
            print('Stock estimation plot has been saved in this directory as {}.'.format(fig_name))

    print("\nPREDICTION TABLE")
    print(60 * "--")
    print("{:<11} {:<25} {:<25} {:<37} {:<15}".format("SYMBOL", "PRICE ON " + str(data["dates"][-1]), "PREDICTED PRICE NEXT DAY",
                                              "STANDARD DEVIATION OF PREDICTION", "RATING"))
    print(60 * "--")
    for i in range(num_stocks):
        print("{:<11} {:<25} {:<25} {:<37} {:<15}".format(ranked_tickers[i], ranked_p[i, -1],
                                                  ranked_p_pred[i, 1], ranked_std_p_pred[i, 1], ranked_rating[i]))
        print(60 * "--")

    if args.save_prediction_table:
        tab_name = 'prediction_table.csv'
        table = zip(["SYMBOL"] + ranked_tickers.tolist(),
                    ["PRICE ON " + str(data["dates"][-1])] + ranked_p[:, -1].tolist(),
                    ["PREDICTED PRICE NEXT DAY"] + ranked_p_pred[:, 1].tolist(),
                    ["STANDARD DEVIATION OF PREDICTION"] + ranked_std_p_pred[:, 1].tolist(),
                    ["RATING"] + ranked_rating)
        with open(tab_name, 'w') as file:
            wr = csv.writer(file)
            for row in table:
                wr.writerow(row)
        print('\nThe prediction table printed above has been saved in this directory as {}.'.format(tab_name))
