#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from argparse import ArgumentParser
import csv

import yfinance as yf
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


def load_data(tickers, start, end):
    tickers = list(set(tickers))
    stocks = yf.Tickers(tickers)
    df = stocks.download(start=start, end=end)['Close'].dropna(1)
    missing_tickers = [tick for tick in tickers if tick not in df.columns]
    if len(missing_tickers) > 0:
        print('\nRemoving {} from list of symbols because yahoo-finance could not provide full information.\n'.format(
            missing_tickers))
    tickers = list(df.columns)
    stocks = yf.Tickers(tickers)

    logp = np.log(df.to_numpy().T)
    listed = np.where(np.sum(np.isnan(logp), 1) == 0)[0].tolist()
    logp = logp[listed]

    sector_name = []
    for i in listed:
        try:
            sector_name.append(stocks.tickers[i].info["sector"])
        except:
            sector_name.append("NA" + str(i))
    sectors = np.unique(sector_name)
    sector_id = [np.where(sectors == sector)[0][0] for sector in sector_name]

    return dict(tickers=tickers, dates=df.index, sector_id=sector_id, logp=logp)


if __name__ == '__main__':
    today = dt.date.today()
    one_year_ago = dt.date.today() - dt.timedelta(365)
    cli = ArgumentParser('Volatile: your day-to-day companion for financial stock trading.')
    cli.add_argument('-s', '--symbols', type=str, nargs='+', help='List of symbols.')
    cli.add_argument('-sd', '--start_date', type=str, default=one_year_ago.strftime("%Y-%m-%d"),
                     help='Start collecting data from this date.')
    cli.add_argument('-cd', '--current_date', type=str, default=today.strftime("%Y-%m-%d"),
                     help='Collect data up to this date. Predictions will concern the next available trading day.')
    args = cli.parse_args()

    data = load_data(args.symbols, start=args.start_date, end=args.current_date)
    tickers = data["tickers"]
    num_sectors = len(set(data["sector_id"]))
    num_stocks = data["logp"].shape[0]

    num_days = (dt.datetime.strptime(args.current_date, "%Y-%m-%d").date()
                - dt.datetime.strptime(args.start_date, "%Y-%m-%d").date()).days
    order =  np.clip(num_days // 30, 1, 20)
    t = data["logp"].shape[1]
    tt = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')

    model = tfd.JointDistributionSequential([
        # phi_s
        tfd.Independent(tfd.Normal(loc=tf.zeros([num_sectors, 1]), scale=1), 2),
        # phi
        lambda phi_s: tfd.Independent(
            tfd.Normal(loc=tf.gather(phi_s, data["sector_id"], axis=0),
                       scale=np.linspace(1 / (order + 1), 1, order + 1)[::-1].astype('float32')[None, :]), 2),
        # psi_s
        tfd.Independent(tfd.Normal(loc=0, scale=tf.ones([num_sectors, 1])), 2),
        # psi
        lambda psi_s: tfd.Independent(tfd.Normal(loc=tf.gather(psi_s, data["sector_id"], axis=0), scale=1), 2),
        # y
        lambda psi, psi_s, phi: tfd.Independent(tfd.Normal(loc=tf.tensordot(phi, tt, axes=1),
                                                           scale=tf.math.softplus(psi + 1 - tt[1])), 2)])

    print("\nTraining the model...")
    phi_s, phi, psi_s, psi = (tf.Variable(model.sample()[:-1][i]) for i in range(4))
    logposterior = lambda phi_s, phi, psi_s, psi: model.log_prob([phi_s, phi, psi_s, psi, data["logp"]])
    loss = tfp.math.minimize(lambda: -logposterior(phi_s, phi, psi_s, psi),
                             optimizer=tf.optimizers.Adam(learning_rate=0.01),
                             num_steps=10000)

    logp_est = np.dot(phi.numpy(), tt)
    std_logp_est = np.log(1 + np.exp(psi.numpy() + 1 - tt[1]))
    p_est = np.exp(logp_est + std_logp_est ** 2 / 2)
    std_p_est = np.sqrt(np.exp(2 * logp_est + std_logp_est ** 2) * (np.exp(std_logp_est ** 2) - 1))

    logp_pred = np.dot(phi.numpy(), np.array(1 + 1 / t) ** np.arange(order + 1))
    std_logp_pred = np.log(1 + np.exp(psi.numpy().squeeze() + 1 / t))
    p_pred = np.exp(logp_pred + std_logp_pred ** 2 / 2)
    std_p_pred = np.sqrt(np.exp(2 * logp_pred + std_logp_pred ** 2) * (np.exp(std_logp_pred ** 2) - 1))

    scores = ((logp_pred - data["logp"][:, -1]) / std_logp_pred)
    rank = np.argsort(scores)[::-1]
    ranked_tickers = np.array(tickers)[rank]
    ranked_p = np.exp(data["logp"])[rank]
    ranked_p_est = p_est[rank]
    ranked_std_p_est = std_p_est[rank]
    ranked_p_pred = p_pred[rank]
    ranked_std_p_pred = std_p_pred[rank]

    ranked_left_est = np.maximum(0, ranked_p_est - 2 * ranked_std_p_est)
    ranked_right_est = ranked_p_est + 2 * ranked_std_p_est

    num_columns = 3
    fig = plt.figure(figsize=(20, num_stocks))
    plt.suptitle("Price estimation between " + str(data["dates"][0])[:10] + " and " + str(data["dates"][-1])[:10]
                 + " via a " + str(order) + "-order polynomial regression", y=1.03, fontsize=20)
    for i in range(num_stocks):
        plt.subplot(num_stocks // 3 + 1, num_columns, i + 1)
        plt.title(ranked_tickers[i], fontsize=15)
        plt.plot(data["dates"], ranked_p[i], label="data")
        plt.plot(data["dates"], ranked_p_est[i], label="estimation")
        plt.fill_between(data["dates"], ranked_left_est[i], ranked_right_est[i], alpha=0.2, label="+/- 2 st. dev.")
        plt.yticks(fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(loc="upper left")
    plt.tight_layout()
    fig_name = 'estimation' + args.current_date + '.png'
    fig.savefig(fig_name, dpi=fig.dpi)
    print('\nStock estimation plots have been saved in this directory as {}.'.format(fig_name))

    print("\nRANKING SYMBOLS FROM MOST TO LEAST POTENTIALLY PROFITABLE")
    print(40 * "--")
    print("{:<8} {:<25} {:<25} {:<25}".format("symbol", "current price", "next price", "standard deviation"))
    print(40 * "--")
    for i in range(num_stocks):
        print("{:<8} {:<25} {:<25} {:<25}".format(ranked_tickers[i], ranked_p[i, -1],
                                                  ranked_p_pred[i], ranked_std_p_pred[i]))
        print(40 * "--")


    tab_name = 'prediction' + args.current_date + '.csv'
    table = zip(["SYMBOLS"] + ranked_tickers.tolist(),
                ["OBSERVED CURRENT PRICE"] + ranked_p[:, -1].tolist(),
                ["PREDICTED NEXT PRICE"] + ranked_p_pred.tolist(),
                ["STANDARD DEVIATION OF PREDICTION"] + ranked_std_p_pred.tolist())
    with open(tab_name, 'w') as file:
        wr = csv.writer(file)
        for row in table:
            wr.writerow(row)
    print('\nThe prediction data printed above has been saved in this directory as {}.'.format(tab_name))

