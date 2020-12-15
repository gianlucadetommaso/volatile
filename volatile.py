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
        - dates: dates corresponding to available prices;
        - sectors: list of sectors at stock-level;
        - industries: list of industries at stock-level;
        - logp: log-prices at stock-level.
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
            for row in zip(["SYMBOL"], ["SECTOR"], ["INDUSTRY"]):
                wr.writerow(row)
    # load stock information file
    stock_info = pd.read_csv(filename)

    # load sector and industry information. If any is already available in the stock information file, load it from
    # there. Otherwise, try out if it is available in the data. If not, give it a unique name.
    sectors = []
    industries = []
    missing_sector = {}
    missing_industry = {}
    for i in range(len(tickers)):
        idx = np.where(stock_info["SYMBOL"].values == tickers[i])[0]
        if len(idx) > 0:
            sectors.append(stock_info["SECTOR"][idx[0]])
            industries.append(stock_info["INDUSTRY"][idx[0]])
        else:
            try:
                info = stocks.tickers[i].info
                sectors.append(info["sector"])
                missing_sector[tickers[i]] = sectors[-1]
                industries.append(info["industry"])
                missing_industry[tickers[i]] = industries[-1]
            except:
                sectors.append("NA_sector" + str(i))
                industries.append("NA_industry" + str(i))

    # cache information that was not present before, except for names that were given artificially.
    stock_info = zip(list(missing_sector.keys()), list(missing_sector.values()), list(missing_industry.values()))
    with open(filename, 'a+', newline='') as file:
        wr = csv.writer(file)
        for row in stock_info:
            wr.writerow(row)

    return dict(tickers=tickers, dates=pd.to_datetime(df.index).date, sectors=sectors, industries=industries, logp=logp)

def define_model(tt: np.array, order_scale: np.array, sectors: list, industries: list):
    """
    Define and return graphical model.

    Parameters
    ----------
    tt: np.array
        Time sequence corresponding to dates in the data. It is sed to construct polynomial model and reweigh likelihood 
        scale.
    order_scale: np.array
        It reweighs prior scales of parameters at different orders of the polynomial.
    sectors: list
        Sectors at stock-level.
    industries: list
        Industries at stock-level.
    """
    # find unique names of sectors
    usectors = np.unique(sectors)
    num_sectors = len(usectors)
    # provide sector IDs at stock-level
    sectors_id = [np.where(usectors == sector)[0][0] for sector in sectors]
    # find unique names of industries and store indices
    uindustries, industries_idx = np.unique(industries, return_index=True)
    # provide industry IDs at stock-level
    industries_id = [np.where(uindustries == industry)[0][0] for industry in industries]
    # provide sector IDs at industry-level
    sectors_industry_id = np.array(sectors_id)[industries_idx].tolist()
    # order of the polynomial model
    order = len(order_scale) - 1

    return tfd.JointDistributionSequential([
           # phi_m
           tfd.Independent(tfd.Normal(loc=tf.zeros([1, order + 1]), scale=4 * order_scale), 2),
           # phi_s
           lambda phi_m: tfd.Independent(tfd.Normal(loc=tf.repeat(phi_m, num_sectors, axis=0), scale=2 * order_scale), 2),
           # phi_i
           lambda phi_s: tfd.Independent(tfd.Normal(loc=tf.gather(phi_s, sectors_industry_id, axis=0),
                                                    scale=order_scale), 2),
           # phi
           lambda phi_i: tfd.Independent(tfd.Normal(loc=tf.gather(phi_i, industries_id, axis=0),
                                                    scale=0.5 * order_scale), 2),
           # psi_m
           tfd.Normal(loc=0, scale=4),
           # psi_s
           lambda psi_m: tfd.Independent(tfd.Normal(loc=psi_m, scale=2 * tf.ones([num_sectors, 1])), 2),
           # psi_i
           lambda psi_s: tfd.Independent(tfd.Normal(loc=tf.gather(psi_s, sectors_industry_id, axis=0), scale=1), 2),
           # psi
           lambda psi_i: tfd.Independent(tfd.Normal(loc=tf.gather(psi_i, industries_id, axis=0), scale=0.5), 2),
           # y
           lambda psi, psi_i, psi_s, psi_m, phi: tfd.Independent(tfd.Normal(loc=tf.tensordot(phi, tt, axes=1),
                                                                            scale=tf.math.softplus(psi + 1 - tt[1])), 2)])

def training(phi_m: tf.Tensor, phi_s: tf.Tensor, phi_i: tf.Tensor, phi: tf.Tensor, psi_m: tf.Tensor, psi_s: tf.Tensor,
             psi_i: tf.Tensor, psi: tf.Tensor, model: tfd.JointDistributionSequential, logp: np.array,
             learning_rate: float = 0.01, num_steps: int =10000, plot_loss: bool = False):
    """
    It performs optimization over the model parameters via Adam optimizer.

    Parameters
    ----------
    phi_m: tf.Tensor
        Initial value of market-level polynomial parameter.
    phi_s: tf.Tensor
        Initial values of sector-level polynomial parameters.
    phi_i: tf.Tensor
        Initial values of industry-level polynomial parameters.
    phi: tf.Tensor
        Initial values of stock-level polynomial parameters.
    psi_m: tf.Tensor
        Initial values of market-level likelihood scale parameter.
    psi_s: tf.Tensor
        Initial values of sector-level likelihood scale parameters.
    psi_i: tf.Tensor
        Initial values of industry-level likelihood scale parameters.
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
    plot_loss: bool
        If True, a loss function decay plot is saved in the current directory.

    Returns
    -------
    It returns trained parameters.
    """
    def log_posterior(phi_m, phi_s, phi_i, phi, psi_m, psi_s, psi_i, psi, logp):
        return model.log_prob([phi_m, phi_s, phi_i, phi, psi_m, psi_s, psi_i, psi, logp])
    loss = tfp.math.minimize(lambda: -log_posterior(phi_m, phi_s, phi_i, phi, psi_m, psi_s, psi_i, psi, logp),
                             optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                             num_steps=num_steps)
    if plot_loss:
        fig_name = 'loss_decay.png'
        fig = plt.figure(figsize=(10, 3))
        plt.plot(loss)
        plt.legend(["loss decay"])
        plt.xlabel("iteration")
        fig.savefig(fig_name, dpi=fig.dpi)
        print('Loss function decay plot has been saved in this directory as {}.'.format(fig_name))
    return phi_m, phi_s, phi_i, phi, psi_m, psi_s, psi_i, psi

def order_selection(logp: np.array, orders: np.array = np.arange(2, 14), horizon: int = 5):
    t = logp[:, :-horizon].shape[1]
    losses = []
    suborders = [orders[2 * i:2 * (i + 1)] for i in range(int(np.ceil(0.5 * len(orders))))]
    i = 0
    print("\nModel selection in progress. This can take a few minutes...")
    while i < len(suborders):
        sub_losses = []
        for order in suborders[i]:
            # times corresponding to trading dates in the data
            tt = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
            # reweighing factors for parameters corresponding to different orders of the polynomial
            order_scale = np.linspace(1 / (order + 1), 1, order + 1)[::-1].astype('float32')[None, :]
            # prediction times up to horizon
            tt_pred = np.arange(1, 1 + horizon) / t

            # training the model
            model = define_model(tt, order_scale, data['sectors'], data['industries'])
            phi_m, phi_s, phi_i, phi, psi_m, psi_s, psi_i, psi = (tf.Variable(tf.zeros_like(model.sample()[:-1][i])) for i in
                                                                  range(8))
            phi_m, phi_s, phi_i, phi, psi_m, psi_s, psi_i, psi = training(phi_m, phi_s, phi_i, phi, psi_m, psi_s, psi_i, psi,
                                                                          model, logp[:, :-horizon])
            # calculate stock-level predictions of log-prices
            logp_pred = np.dot(phi.numpy(), np.array([1 + tt_pred]) ** np.arange(order + 1)[:, None])
            std_logp_pred = np.log(1 + np.exp(psi.numpy() + tt_pred))
            scores = (logp_pred - logp[:, -horizon:]) / std_logp_pred
            sub_losses.append(np.mean(scores ** 2) - 1)
        if len(losses) > 0 and np.min(sub_losses) > np.min(losses):
            break
        losses += sub_losses
        i += 1
    order = orders[np.argmin(losses)]
    print("Model selection completed. Volatile will use a polynomial model of degree {}.".format(order))
    return order

if __name__ == '__main__':
    cli = ArgumentParser('Volatile: your day-to-day trading companion.',
                         formatter_class=ArgumentDefaultsHelpFormatter)
    cli.add_argument('-s', '--symbols', type=str, nargs='+', help='List of symbols.')
    cli.add_argument('--save-table', action='store_true',
                     help='Save prediction table in csv format.')
    cli.add_argument('--no-plots', action='store_true',
                     help='Plot estimates with their uncertainty over time.')
    cli.add_argument('--plot-loss', action='store_true',
                     help='Plot loss function decay over training iterations.')
    args = cli.parse_args()

    today = dt.date.today().strftime("%Y-%m-%d")

    print('\nDownloading all available closing prices in the last year...')
    if args.symbols is None:
        with open("symbols_list.txt", "r") as my_file:
            args.symbols = my_file.readlines()[0].split(" ")
    data = load_data(args.symbols)
    tickers = data["tickers"]
    num_stocks = data['logp'].shape[0]

    # how many days to look ahead when comparing the current price against a prediction
    horizon = 5
    # order of the polynomial
    order = order_selection(data['logp']) if num_stocks >= 30 else 5

    print("\nTraining the model...")

    # number of trading dates in the data
    t = data["logp"].shape[1]
    # times corresponding to trading dates in the data
    tt = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
    # reweighing factors for parameters corresponding to different orders of the polynomial
    order_scale = np.linspace(1 / (order + 1), 1, order + 1)[::-1].astype('float32')[None, :]
    # prediction times up to horizon
    tt_pred = np.arange(1 + horizon) / t

    # training the model
    model = define_model(tt, order_scale, data['sectors'], data['industries'])
    phi_m, phi_s, phi_i, phi, psi_m, psi_s, psi_i, psi = (tf.Variable(tf.zeros_like(model.sample()[:-1][i])) for i in range(8))
    phi_m, phi_s, phi_i, phi, psi_m, psi_s, psi_i, psi = training(phi_m, phi_s, phi_i, phi, psi_m, psi_s, psi_i, psi,
                                                                  model, data["logp"], plot_loss=args.plot_loss)
    # calculate stock-level estimators of log-prices
    logp_est = np.dot(phi.numpy(), tt)
    std_logp_est = np.log(1 + np.exp(psi.numpy() + 1 - tt[1]))
    # calculate stock-level estimators of prices
    p_est = np.exp(logp_est + std_logp_est ** 2 / 2)
    # calculate stock-level predictions of log-prices
    logp_pred = np.dot(phi.numpy(), np.array([1 + tt_pred]) ** np.arange(order + 1)[:, None])
    std_logp_pred = np.log(1 + np.exp(psi.numpy() + tt_pred))
    std_p_est = np.sqrt(np.exp(2 * logp_est + std_logp_est ** 2) * (np.exp(std_logp_est ** 2) - 1))
    # calculate stock-level prediction of prices
    p_pred = np.exp(logp_pred + std_logp_pred ** 2 / 2)
    std_p_pred = np.sqrt(np.exp(2 * logp_pred + std_logp_pred ** 2) * (np.exp(std_logp_pred ** 2) - 1))
    # calculate industry-level estimators of log-prices
    logp_ind_est = np.dot(phi_i.numpy(), tt)
    std_logp_ind_est = np.log(1 + np.exp(psi_i.numpy() + 1 - tt[1]))
    # calculate industry-level estimators of prices
    p_ind_est = np.exp(logp_ind_est + std_logp_ind_est ** 2 / 2)
    std_p_ind_est = np.sqrt(np.exp(2 * logp_ind_est + std_logp_ind_est ** 2) * (np.exp(std_logp_ind_est ** 2) - 1))
    # calculate sector-level estimators of log-prices
    logp_sec_est = np.dot(phi_s.numpy(), tt)
    std_logp_sec_est = np.log(1 + np.exp(psi_s.numpy() + 1 - tt[1]))
    # calculate sector-level estimators of prices
    p_sec_est = np.exp(logp_sec_est + std_logp_sec_est ** 2 / 2)
    std_p_sec_est = np.sqrt(np.exp(2 * logp_sec_est + std_logp_sec_est ** 2) * (np.exp(std_logp_sec_est ** 2) - 1))
    # calculate market-level estimators of log-prices
    logp_mkt_est = np.dot(phi_m.numpy(), tt)
    std_logp_mkt_est = np.log(1 + np.exp(psi_m.numpy() + 1 - tt[1]))
    # calculate market-level estimators of prices
    p_mkt_est = np.exp(logp_mkt_est + std_logp_mkt_est ** 2 / 2)
    std_p_mkt_est = np.sqrt(np.exp(2 * logp_mkt_est + std_logp_mkt_est ** 2) * (np.exp(std_logp_mkt_est ** 2) - 1))

    print("Training completed.")

    # calculate score
    scores = ((logp_pred[:, horizon] - data["logp"][:, -1]) / std_logp_pred[:, horizon])
    # rank according to score
    rank = np.argsort(scores)[::-1]
    ranked_tickers = np.array(tickers)[rank]
    ranked_scores = scores[rank]
    ranked_p = np.exp(data["logp"])[rank]
    ranked_p_est = p_est[rank]
    ranked_std_p_est = std_p_est[rank]
    ranked_p_pred = p_pred[rank]
    ranked_std_p_pred = std_p_pred[rank]
    
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

    if not args.no_plots:
        print('\nPlotting market estimation...')
        fig = plt.figure(figsize=(10,3))
        left_mkt_est = np.maximum(0, p_mkt_est - 2 * std_p_mkt_est)
        right_mkt_est = p_mkt_est + 2 * std_p_mkt_est

        plt.plot(data["dates"], p_mkt_est[0], label="market estimation", color="C1")
        plt.fill_between(data["dates"], left_mkt_est[0], right_mkt_est[0], alpha=0.2, label="+/- 2 st. dev.", color="C0")
        plt.legend(loc="upper left")
        fig_name = 'market_estimation.png'
        fig.savefig(fig_name, dpi=fig.dpi)
        print('Market estimation plot has been saved in this directory as {}.'.format(fig_name))

        num_columns = 3
        print('\nPlotting sector estimation...')
        # determine which sectors were not available to avoid plotting
        usectors = np.unique(data["sectors"])
        num_sectors = len(usectors)
        NA_sectors = np.where(np.array([sec[:2] for sec in usectors]) == "NA")[0]
        num_NA_sectors = len(NA_sectors)

        left_sec_est = np.maximum(0, p_sec_est - 2 * std_p_sec_est)
        right_sec_est = p_sec_est + 2 * std_p_sec_est
        fig = plt.figure(figsize=(20, max(num_sectors - num_NA_sectors, 5)))
        j = 0
        for i in range(num_sectors):
            if i not in NA_sectors:
                j += 1
                plt.subplot(int(np.ceil((num_sectors - num_NA_sectors) / num_columns)), num_columns, j)
                plt.title(usectors[i], fontsize=15)
                plt.plot(data["dates"], p_sec_est[i], label="sector estimation", color="C1")
                plt.fill_between(data["dates"], left_sec_est[i], right_sec_est[i], alpha=0.2, label="+/- 2 st. dev.", color="C0")
                plt.yticks(fontsize=12)
                plt.xticks(rotation=45)
                plt.legend(loc="upper left")
        plt.tight_layout()
        fig_name = 'sector_estimation.png'
        fig.savefig(fig_name, dpi=fig.dpi)
        print('Sector estimation plot has been saved in this directory as {}.'.format(fig_name))

        print('\nPlotting industry estimation...')
        uindustries = np.unique(data["industries"])
        num_industries = len(uindustries)
        NA_industries = np.where(np.array([ind[:2] for ind in uindustries]) == "NA")[0]
        num_NA_industries = len(NA_industries)

        left_ind_est = np.maximum(0, p_ind_est - 2 * std_p_ind_est)
        right_ind_est = p_ind_est + 2 * std_p_ind_est
        fig = plt.figure(figsize=(20, max(num_industries - num_NA_industries, 5)))
        j = 0
        for i in range(num_industries):
            if i not in NA_industries:
                j += 1
                plt.subplot(int(np.ceil((num_industries - num_NA_industries) / num_columns)), num_columns, j)
                plt.title(uindustries[i], fontsize=15)
                plt.plot(data["dates"], p_ind_est[i], label="industry estimation", color="C1")
                plt.fill_between(data["dates"], left_ind_est[i], right_ind_est[i], alpha=0.2, label="+/- 2 st. dev.", color="C0")
                plt.yticks(fontsize=12)
                plt.xticks(rotation=45)
                plt.legend(loc="upper left")
        plt.tight_layout()
        fig_name = 'industry_estimation.png'
        fig.savefig(fig_name, dpi=fig.dpi)
        print('Industry estimation plot has been saved in this directory as {}.'.format(fig_name))

        # determine which stocks are along trend to avoid plotting them
        along_trend = np.where(np.array(ranked_rating) == "ALONG TREND")[0]
        num_out_trend = num_stocks - len(along_trend)

        if num_out_trend > 0:
            print('\nPlotting stock estimation...')
            ranked_left_est = np.maximum(0, ranked_p_est - 2 * ranked_std_p_est)
            ranked_right_est = ranked_p_est + 2 * ranked_std_p_est

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
        elif os.path.exists('stock_estimation.png'):
            os.remove('stock_estimation.png')

    print("\nPREDICTION TABLE")
    ranked_sectors = [name if name[:2] != "NA" else "Not Available" for name in np.array(data["sectors"])[rank]]
    ranked_industries = [name if name[:2] != "NA" else "Not Available" for name in np.array(data["industries"])[rank]]
    num_dashes = 193
    print(num_dashes * "-")
    print("{:<11} {:<26} {:<42} {:<25} {:<28} {:<37} {:<15}".format("SYMBOL", "SECTOR", "INDUSTRY",
                                                             "PRICE ON " + str(data["dates"][-1]),
                                                             "PREDICTED PRICE NEXT DAY",
                                                             "STANDARD DEVIATION OF PREDICTION", "SCORE", "RATING"))
    print(num_dashes * "-")
    for i in range(num_stocks):
        print("{:<11} {:<26} {:<42} {:<25} {:<28} {:<37} {:<15}".format(ranked_tickers[i], ranked_sectors[i],
                                                                        ranked_industries[i], ranked_p[i, -1],
                                                                        ranked_p_pred[i, 1], ranked_std_p_pred[i, 1],
                                                                        ranked_rating[i]))
        print(num_dashes * "-")
        if i + 1 in si.values():
            print(num_dashes * "-")

    if args.save_table:
        tab_name = 'prediction_table.csv'
        table = zip(["SYMBOL"] + ranked_tickers.tolist(),
                    ['SECTOR'] + ranked_sectors,
                    ['INDUSTRY'] + ranked_industries,
                    ["PRICE ON " + str(data["dates"][-1])] + ranked_p[:, -1].tolist(),
                    ["PREDICTED PRICE NEXT DAY"] + ranked_p_pred[:, 1].tolist(),
                    ["STANDARD DEVIATION OF PREDICTION"] + ranked_std_p_pred[:, 1].tolist(),
                    ["RATING"] + ranked_rating)
        with open(tab_name, 'w') as file:
            wr = csv.writer(file)
            for row in table:
                wr.writerow(row)
        print('\nThe prediction table printed above has been saved in this directory as {}.'.format(tab_name))
