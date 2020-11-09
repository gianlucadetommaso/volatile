#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import functools
import datetime as dt

import yfinance as yf
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


def load_data(tickers, start, end):
    tickers = list(set(tickers))
    stocks = yf.Tickers(tickers)
    df = stocks.download(start=dt.datetime.strftime(start, "%Y-%m-%d"),
                         end=dt.datetime.strftime(end, "%Y-%m-%d"))['Close'].dropna(1)
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


def main():
    tickers = ['TSLA', 'NCLH', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'DAL', 'GILD', 'IBM', 'RCL', 'UNM', 'AMRN', 'AAL',
               'AAPL', 'AMAT', 'BP', 'CCL', 'PLAY', 'EJT1', 'EQR', 'FSLY', 'INTC', 'MSF', 'NLOK', 'OPTT', 'CKH',
               'TSM', 'VRTX', 'VNO', 'WDI', 'WBA', 'XBIT', 'BNTX', 'PFE']
    daydiff = 365
    start = dt.date.today() - dt.timedelta(daydiff)
    end = dt.date.today()
    data = load_data(tickers, start=start, end=end)
    tickers = data["tickers"]
    num_sectors = len(set(data["sector_id"]))
    num_stocks = data["logp"].shape[0]

    order =  np.clip(daydiff // 30, 1, 20)
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
    figname = 'overview' + dt.datetime.strftime(end, "%Y-%m-%d") + '.png'
    fig.savefig(figname, dpi=fig.dpi)
    print('Overview of stocks estimation has been saved in this directory with name ' + figname)

    print("\nBest-to-worst ranked symbol\n")
    print("{:<8} {:<25} {:<25} {:<25}".format("symbol", "current price", "next price", "standard deviation"))
    print(45 * "--")
    for i in range(num_stocks):
        print("{:<8} {:<25} {:<25} {:<25}".format(ranked_tickers[i], ranked_p[i, -1],
                                                  ranked_p_pred[i], ranked_std_p_pred[i]))
        print(40 * "--")


if __name__ == '__main__':
    main()

