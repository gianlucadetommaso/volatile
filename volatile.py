#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
import csv
import os.path

from download import download
from tools import convert_currency, extract_hierarchical_info
from plotting import *

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import multitasking

def define_model(info: dict, level: str = "stock") -> tfd.JointDistributionSequentialAutoBatched:
    """
    Define and return graphical model.

    Parameters
    ----------
    info: dict
        Data information.
    level: str
        Level of the model; possible candidates are "stock", "industry", "sector" and "market".
    """
    tt = info['tt']
    order_scale = info['order_scale']
    order =  len(order_scale) - 1
    num_sectors = info['num_sectors']
    sec2ind_id = info['sector_industries_id']
    ind_id = info['industries_id']

    available_levels = ["market", "sector", "industry", "stock"]
    if level not in available_levels:
        raise Exception("Selected level is unknown. Please provide one of the following levels: {}.".format(available_levels))

    m = [tfd.Normal(loc=tf.zeros([1, order + 1]), scale=4 * order_scale), # phi_m
         tfd.Normal(loc=0, scale=4)] # psi_m

    if level != "market":
        m += [lambda psi_m, phi_m: tfd.Normal(loc=tf.repeat(phi_m, num_sectors, axis=0), scale=2 * order_scale), # phi_s
              lambda phi_s, psi_m: tfd.Normal(loc=psi_m, scale=2 * tf.ones([num_sectors, 1]))] # psi_s

        if level != "sector":
            sec2ind_id = info['sector_industries_id']
            m += [lambda psi_s, phi_s: tfd.Normal(loc=tf.gather(phi_s, sec2ind_id, axis=0), scale=order_scale), # phi_i
                  lambda phi_i, psi_s: tfd.Normal(loc=tf.gather(psi_s, sec2ind_id, axis=0), scale=1)] # psi_ii

            if level != "industry":
                ind_id = info['industries_id']
                m += [lambda psi_i, phi_i: tfd.Normal(loc=tf.gather(phi_i, ind_id, axis=0), scale=0.5 * order_scale), # phi
                      lambda phi, psi_i: tfd.Normal(loc=tf.gather(psi_i, ind_id, axis=0), scale=0.5)]  # psi

    if level == "market":
        m += [lambda psi_m, phi_m: tfd.Normal(loc=tf.tensordot(phi_m, tt, axes=1), scale=tf.math.softplus(psi_m))] # y
    if level == "sector":
        m += [lambda psi_s, phi_s: tfd.Normal(loc=tf.tensordot(phi_s, tt, axes=1), scale=tf.math.softplus(psi_s))] # y
    if level == "industry":
        m += [lambda psi_i, phi_i: tfd.Normal(loc=tf.tensordot(phi_i, tt, axes=1), scale=tf.math.softplus(psi_i))] # y
    if level == "stock":
        m += [lambda psi, phi: tfd.Normal(loc=tf.tensordot(phi, tt, axes=1), scale=tf.math.softplus(psi))] # y

    return tfd.JointDistributionSequentialAutoBatched(m)

def train(logp: np.array, info: dict, learning_rate: float = 0.01, num_steps: int = 10000, plot_losses: bool = False) -> tuple:
    """
    It performs sequential optimization over the model parameters via Adam optimizer, training at different levels to
    provide sensible initial solutions at finer levels.

    Parameters
    ----------
    logp: np.array
        Log-price at stock-level.
    info: dict
        Data information.
    learning_rate: float
        Adam's fixed learning rate.
    num_steps: int
        Adam's fixed number of iterations.
    plot_losses: bool
        If True, a losses decay plot is saved in the current directory.

    Returns
    -------
    It returns a tuple of trained parameters.
    """
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    num_steps_l = int(np.ceil(num_steps // 4))

    # market
    model = define_model(info, "market")
    phi_m, psi_m = (tf.Variable(tf.zeros_like(model.sample()[:2][i])) for i in range(2))
    loss_m = tfp.math.minimize(lambda: -model.log_prob([phi_m, psi_m, logp.mean(0, keepdims=1)]),
                             optimizer=optimizer, num_steps=num_steps_l)
    # sector
    model = define_model(info, "sector")
    phi_m, psi_m = tf.constant(phi_m), tf.constant(psi_m)
    phi_s, psi_s = (tf.Variable(tf.zeros_like(model.sample()[2:4][i])) for i in range(2))
    logp_s = np.array([logp[np.where(np.array(info['sectors_id']) == k)[0]].mean(0) for k in range(info['num_sectors'])])
    loss_s = tfp.math.minimize(lambda: -model.log_prob([phi_m, psi_m, phi_s, psi_s, logp_s]),
                             optimizer=optimizer, num_steps=num_steps_l)

    # industry
    model = define_model(info, "industry")
    phi_s, psi_s = tf.constant(phi_s), tf.constant(psi_s)
    phi_i, psi_i = (tf.Variable(tf.zeros_like(model.sample()[4:6][i])) for i in range(2))
    logp_i = np.array([logp[np.where(np.array(info['industries_id']) == k)[0]].mean(0) for k in range(info['num_industries'])])
    loss_i = tfp.math.minimize(lambda: -model.log_prob([phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, logp_i]),
                             optimizer=optimizer, num_steps=num_steps_l)
    # stock
    model = define_model(info, "stock")
    phi_i, psi_i = tf.constant(phi_i), tf.constant(psi_i)
    phi, psi = (tf.Variable(tf.zeros_like(model.sample()[6:8][i])) for i in range(2))
    loss = tfp.math.minimize(lambda: -model.log_prob([phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, phi, psi, logp]),
                             optimizer=optimizer, num_steps=num_steps_l)

    if plot_losses:
        fig_name = 'losses_decay.png'
        fig = plt.figure(figsize=(20, 3))
        plt.subplot(141)
        plt.title("market-level", fontsize=12)
        plt.plot(loss_m)
        plt.subplot(142)
        plt.title("sector-level", fontsize=12)
        plt.plot(loss_s)
        plt.subplot(143)
        plt.title("industry-level", fontsize=12)
        plt.plot(loss_i)
        plt.subplot(144)
        plt.title("stock-level", fontsize=12)
        plt.plot(loss)
        plt.legend(["loss decay"], fontsize=12, loc="upper right")
        plt.xlabel("iteration", fontsize=12)
        fig.savefig(fig_name, dpi=fig.dpi)
        print('Losses decay plot has been saved in this directory as {}.'.format(fig_name))
    return phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, phi, psi

def softplus(x: np.array) -> np.array:
    """
    It is a function from real to positive numbers

    Parameters
    ----------
    x: np.array
        Real value.
    """
    return np.log(1 + np.exp(x))

def estimate_logprice_statistics(phi: np.array, psi: np.array, tt: np.array) -> tuple:
    """
    It estimates mean and standard deviations of log-prices.

    Parameters
    ----------
    phi: np.array
        Parameters of regression polynomial.
    psi: np.array
        Parameters of standard deviation.
    tt: np.array
        Sequence of times to evaluate statistics at.

    Returns
    -------
    It returns a tuple of mean and standard deviation log-price estimators.
    """
    return np.dot(phi, tt), softplus(psi)

def estimate_price_statistics(mu: np.array, sigma: np.array):
    """
    It estimates mean and standard deviations of prices.

    Parameters
    ----------
    mu: np.array
        Mean estimates of log-prices.
    sigma: np.array
        Standard deviation estimates of log-prices.

    Returns
    -------
    It returns a tuple of mean and standard deviation price estimators.
    """
    return np.exp(mu + sigma ** 2 / 2), np.sqrt(np.exp(2 * mu + sigma ** 2) * (np.exp(sigma ** 2) - 1))

def rate(scores: np.array, lower_bounds: dict = None) -> list:
    """
    Rate scores according to `lower_bounds`. Possible rates are `HIGHLY BELOW TREND`, `BELOW TREND`, `ALONG TREND`,
    `ABOVE TREND` and `HIGHLY ABOVE TREND`.

    Parameters
    ----------
    scores: np.array
        An array of scores for each stock.
    lower_bounds: dict
        It has for keys possible rates and for values corresponding lower-bound lower_bounds, meaning that for a
        stock to be given a certain rate it needs to have score above its lower bound and below higher lower bounds of
        other rates.

    Returns
    -------
    rates: list
        List of rates for each stock.
    """
    if lower_bounds is None:
        lower_bounds = {"HIGHLY BELOW TREND": 3, "BELOW TREND": 2, "ALONG TREND": -2, "ABOVE TREND": -3}
    rates = []
    for i in range(len(scores)):
        if scores[i] > lower_bounds["HIGHLY BELOW TREND"]:
            rates.append("HIGHLY BELOW TREND")
        elif scores[i] > lower_bounds["BELOW TREND"]:
            rates.append("BELOW TREND")
        elif scores[i] > lower_bounds["ALONG TREND"]:
            rates.append("ALONG TREND")
        elif scores[i] > lower_bounds["ABOVE TREND"]:
            rates.append("ABOVE TREND")
        else:
            rates.append("HIGHLY ABOVE TREND")
    return rates

def estimate_matches(tickers: list, phi: np.array, tt: np.array) -> dict:
    """
    It estimates matches of correlated stocks.

    Parameters
    ----------
    tickers: list
        List of tickers
    phi: np.array
        Parameters of regression polynomial.
    tt: np.array
        Array of times corresponding to days of trading.

    Returns
    -------
    matches: dict
        For each symbol, this dictionary contains a corresponding `match` symbol, the `index` of the match symbol in the
        list of symbols and the computed `distance` between the two.
    """
    dtt = np.arange(1, tt.shape[0])[:, None] * tt[1:] / tt[1, None]
    dlogp_est = np.dot(phi[:, 1:],  dtt)
    num_stocks = len(tickers)
    try:
        assert num_stocks <= 2000
        match_dist = np.sum((dlogp_est[:, None] - dlogp_est[None]) ** 2, 2)
        match_minidx = np.argsort(match_dist, 1)[:, 1]
        match_mindist = np.sort(match_dist, 1)[:, 1]
        matches = {tickers[i]: {"match": tickers[match_minidx[i]],
                              "index": match_minidx[i],
                              "distance": match_mindist[i]} for i in range(num_stocks)}
    except:
        num_threads = min([len(tickers), multitasking.cpu_count() * 2])
        multitasking.set_max_threads(num_threads)

        matches = {}

        @multitasking.task
        def _estimate_one(i, tickers, dlogp_est):
            match_dist = np.sum((dlogp_est[i] - dlogp_est) ** 2, 1)
            match_minidx = np.argsort(match_dist)[1]
            match_mindist = np.sort(match_dist)[1]
            matches[tickers[i]] = {"match": tickers[match_minidx], "index": match_minidx, "distance": match_mindist}

        for i in range(num_stocks):
            _estimate_one(i, tickers, dlogp_est)

    return matches

if __name__ == '__main__':
    cli = ArgumentParser('Volatile: your day-to-day trading companion.',
                         formatter_class=ArgumentDefaultsHelpFormatter)
    cli.add_argument('-s', '--symbols', type=str, nargs='+', help=SUPPRESS)
    cli.add_argument('--rank', type=str, default="rate",
                     help="If `rate`, stocks are ranked in the prediction table and in the stock estimation plot from "
                          "the highest below to the highest above trend; if `growth`, ranking is done from the largest"
                          " to the smallest trend growth at the last date.")
    cli.add_argument('--save-table', action='store_true',
                     help='Save prediction table in csv format.')
    cli.add_argument('--no-plots', action='store_true',
                     help='Plot estimates with their uncertainty over time.')
    cli.add_argument('--plot-losses', action='store_true',
                     help='Plot loss function decay over training iterations.')
    args = cli.parse_args()

    if args.rank.lower() not in ["rate", "growth"]:
        raise Exception("{} not recognized. Please provide one between `rate` and `growth`.".format(args.rank))

    today = dt.date.today().strftime("%Y-%m-%d")

    print('\nDownloading all available closing prices in the last year...')
    if args.symbols is None:
        with open("symbols_list.txt", "r") as my_file:
            args.symbols = my_file.readlines()[0].split(" ")
    data = download(args.symbols)
    tickers = data["tickers"]
    logp = np.log(data['price'])

    # convert currencies to most frequent one
    for i, curr in enumerate(data['currencies']):
        if curr != data['default_currency']:
            logp[i] = convert_currency(logp[i], np.array(data['exchange_rates'][curr]), type='forward')

    num_stocks, t = logp.shape

    info = extract_hierarchical_info(data['sectors'], data['industries'])

    print("\nTraining a model that discovers correlations...")
    # order of the polynomial
    order = 52

    # times corresponding to trading dates in the data
    info['tt'] = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
    # reweighing factors for parameters corresponding to different orders of the polynomial
    info['order_scale'] = np.ones((1, order + 1), dtype='float32')

    # train the model
    phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, phi, psi = train(logp, info, num_steps=50000)

    print("Training completed.")

    print("\nEstimate top matches...")
    matches = estimate_matches(tickers, phi.numpy(), info['tt'])
    print("Top matches estimation completed.")

    print("\nTraining a model that estimates and predicts trends...")
    # how many days to look ahead when comparing the current price against a prediction
    horizon = 5
    # order of the polynomial
    order = 2

    # times corresponding to trading dates in the data
    info['tt'] = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
    # reweighing factors for parameters corresponding to different orders of the polynomial
    info['order_scale'] = np.linspace(1 / (order + 1), 1, order + 1)[::-1].astype('float32')[None, :]

    # train the model
    phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, phi, psi = train(logp, info, plot_losses=args.plot_losses)

    print("Training completed.")

    ## log-price statistics (Normal distribution)
    # calculate stock-level estimators of log-prices
    logp_est, std_logp_est = estimate_logprice_statistics(phi.numpy(), psi.numpy(), info['tt'])
    # calculate stock-level predictions of log-prices
    tt_pred = ((1 + (np.arange(1 + horizon) / t)) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
    logp_pred, std_logp_pred = estimate_logprice_statistics(phi.numpy(), psi.numpy(), tt_pred)
    # calculate industry-level estimators of log-prices
    logp_ind_est, std_logp_ind_est = estimate_logprice_statistics(phi_i.numpy(), psi_i.numpy(), info['tt'])
    # calculate sector-level estimators of log-prices
    logp_sec_est, std_logp_sec_est = estimate_logprice_statistics(phi_s.numpy(), psi_s.numpy(), info['tt'])
    # calculate market-level estimators of log-prices
    logp_mkt_est, std_logp_mkt_est = estimate_logprice_statistics(phi_m.numpy(), psi_m.numpy(), info['tt'])

    # compute score
    scores = (logp_pred[:, horizon] - logp[:, -1]) / std_logp_pred.squeeze()
    # compute growth as percentage price variation
    growth = np.dot(phi.numpy()[:, 1:], np.arange(1, order + 1)) / t

    # convert log-price currencies back (standard deviations of log-prices stay the same)
    for i, curr in enumerate(data['currencies']):
        if curr != data['default_currency']:
            logp[i] = convert_currency(logp[i], np.array(data['exchange_rates'][curr]), type='backward')
            logp_est[i] = convert_currency(logp_est[i], np.array(data['exchange_rates'][curr]), type='backward')

    ## price statistics (log-Normal distribution)
    # calculate stock-level estimators of prices
    p_est, std_p_est = estimate_price_statistics(logp_est, std_logp_est)
    # calculate stock-level prediction of prices
    p_pred, std_p_pred = estimate_price_statistics(logp_pred, std_logp_pred)
    # calculate industry-level estimators of prices
    p_ind_est, std_p_ind_est = estimate_price_statistics(logp_ind_est, std_logp_ind_est)
    # calculate sector-level estimators of prices
    p_sec_est, std_p_sec_est = estimate_price_statistics(logp_sec_est, std_logp_sec_est)
    # calculate market-level estimators of prices
    p_mkt_est, std_p_mkt_est = estimate_price_statistics(logp_mkt_est, std_logp_mkt_est)

    # rank according to score
    rank = np.argsort(scores)[::-1] if args.rank == "rate" else np.argsort(growth)[::-1]
    ranked_tickers = np.array(tickers)[rank]
    ranked_scores = scores[rank]
    ranked_p = data['price'][rank]
    ranked_currencies = np.array(data['currencies'])[rank]
    ranked_growth = growth[rank]
    ranked_matches = np.array([matches[ticker]["match"] for ticker in ranked_tickers])

    # rate stocks
    ranked_rates = rate(ranked_scores)

    if not args.no_plots:
        plot_market_estimates(data, p_mkt_est, std_p_mkt_est)
        plot_sector_estimates(data, info, p_sec_est, std_p_sec_est)
        plot_industry_estimates(data, info, p_ind_est, std_p_ind_est)
        plot_stock_estimates(data, p_est, std_p_est, args.rank, rank, ranked_rates)
        plot_matches(data, matches)

    print("\nPREDICTION TABLE")
    ranked_sectors = [name if name[:2] != "NA" else "Not Available" for name in np.array(list(data["sectors"].values()))[rank]]
    ranked_industries = [name if name[:2] != "NA" else "Not Available" for name in np.array(list(data["industries"].values()))[rank]]

    strf = "{:<15} {:<26} {:<42} {:<16} {:<22} {:<11} {:<4}"
    num_dashes = 143
    separator = num_dashes * "-"
    print(num_dashes * "-")
    print(strf.format("SYMBOL", "SECTOR", "INDUSTRY", "PRICE", "RATE", "GROWTH", "MATCH"))
    print(separator)
    for i in range(num_stocks):
        print(strf.format(ranked_tickers[i], ranked_sectors[i], ranked_industries[i],
                          "{} {}".format(np.round(ranked_p[i, -1], 2), ranked_currencies[i]), ranked_rates[i],
                          "{}{}{}".format("+" if ranked_growth[i] >= 0 else "", np.round(100 * ranked_growth[i], 2), '%'),
                          ranked_matches[i]))
        print(separator)
        if i < num_stocks - 1 and ranked_rates[i] != ranked_rates[i + 1]:
            print(separator)

    if args.save_table:
        tab_name = 'prediction_table.csv'
        table = zip(["SYMBOL"] + ranked_tickers.tolist(),
                    ['SECTOR'] + ranked_sectors,
                    ['INDUSTRY'] + ranked_industries,
                    ["PRICE"] + ["{} {}".format(np.round(ranked_p[i, -1], 2), ranked_currencies[i]) for i in range(num_stocks)],
                    ["RATE"] + ranked_rates,
                    ["GROWTH"] + ranked_growth.tolist(),
                    ["MATCH"] + ranked_matches.tolist())

        with open(tab_name, 'w') as file:
            wr = csv.writer(file)
            for row in table:
                wr.writerow(row)
        print('\nThe prediction table printed above has been saved to {}/{}.'.format(os.getcwd(), tab_name))
