#!/usr/bin/env python
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime as dt

from tools import convert_currency, extract_hierarchical_info, compute_risk
from models import *
from download import download, get_exchange_rates
from volatile import rate, estimate_logprice_statistics, estimate_price_statistics
from bots import *

if __name__ == '__main__':
    today = dt.date.today().strftime("%Y-%m-%d")
    onemonthago = (dt.date.today() - dt.timedelta(30)).strftime("%Y-%m-%d")

    cli = ArgumentParser('Volatile Bot-Tournament.', formatter_class=ArgumentDefaultsHelpFormatter)
    cli.add_argument('-s', '--symbols', type=str, nargs='+', help=SUPPRESS)
    cli.add_argument('--capital', type=float, default=100000.0, help='Bots start with this available capital at '
                                                                           'the beginning of the tournament. ')
    cli.add_argument('--currency', type=str, default='USD', help='Currency of the capital in input.')
    cli.add_argument('--start', type=str, default=onemonthago, help='Approximate initial date of the bot-tournament. Format: '
                                                             'YY-MM-DD`.')
    cli.add_argument('--end', type=str, default=today, help='Approximate final date of the bot-tournament. Format: '
                                                             'YY-MM-DD`.')
    args = cli.parse_args()

    if args.capital < 0:
        raise Exception("Capital must be a non-negative number.")
    if args.start >= args.end:
        raise Exception("Start date must be before end date.")

    print('\nDownloading all available closing prices in the last year...')
    if args.symbols is None:
        with open("symbols_list.txt", "r") as my_file:
            args.symbols = my_file.readlines()[0].split(" ")

    # download data
    start_download = (dt.datetime.strptime(args.start, '%Y-%m-%d') - dt.timedelta(365)).strftime("%Y-%m-%d")
    data = download(args.symbols, start=start_download, end=args.end)
    tickers = data["tickers"]
    price = data['price']
    logp = np.log(price)

    # number of tournament days
    num_days = int(np.sum(data['dates'] >= args.start))

    # convert currencies to most frequent one
    for i, curr in enumerate(data['currencies']):
        if curr != data['default_currency']:
            logp[i] = convert_currency(logp[i], np.array(data['exchange_rates'][curr]), type='forward')
            price[i] = np.exp(logp[i])

    # convert initial capital
    if args.currency == data['default_currency']:
        xrate = 1.0
    elif args.currency in data['exchange_rates']:
        xrate = np.array(data['exchange_rates'][args.currency])[-num_days]
    else:
        xrate = get_exchange_rates([args.currency], data['default_currency'], data['dates'],
                                   start=args.start, end=args.end)[args.currency][-num_days]
    args.capital *= xrate

    # Volatile specifics
    num_stocks, num_records = logp.shape
    order = 2
    horizon = 5
    t = num_records - num_days

    # extract hierarchical info
    info = extract_hierarchical_info(data['sectors'], data['industries'])
    info['order_scale'] = np.linspace(1 / (order + 1), 1, order + 1)[::-1].astype('float32')[None, :]
    info['tt'] = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
    tt_pred = ((1 + (np.arange(1 + horizon) / t)) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')

    # tournament participants
    names = ["Adam", "Betty", "Chris", "Dany", "Eddy", "Flora"]
    tournament = {name: globals()[name](args.capital) for name in names}

    # initialize capitals
    uninvested = np.zeros((len(names), num_days))
    invested = np.zeros((len(names), num_days))
    capitals = np.zeros((len(names), num_days))
    risks = np.zeros((len(names), num_days))

    str_format = "{:<11} {:<20} {:<20} {:<20} {:<60} {:<20}"
    num_dashes = 143
    separator = num_dashes * "-"

    print("\n*** LET'S THE BOT-TOURNAMENT BEGINS! ***\n")
    for j in range(num_days, 0, -1):
        phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, phi, psi = train_msis_mcs(logp[:, -t - j:-j], info)

        logp_est, std_logp_est = estimate_logprice_statistics(phi.numpy(), psi.numpy(), info['tt'])
        logp_pred, std_logp_pred = estimate_logprice_statistics(phi.numpy(), psi.numpy(), tt_pred)
        logp_ind_est, std_logp_ind_est = estimate_logprice_statistics(phi_i.numpy(), psi_i.numpy(), info['tt'])
        logp_sec_est, std_logp_sec_est = estimate_logprice_statistics(phi_s.numpy(), psi_s.numpy(), info['tt'])
        logp_mkt_est, std_logp_mkt_est = estimate_logprice_statistics(phi_m.numpy(), psi_m.numpy(), info['tt'])

        p_est, std_p_est = estimate_price_statistics(logp_est, std_logp_est)
        p_ind_est, std_p_ind_est = estimate_price_statistics(logp_ind_est, std_logp_ind_est)
        p_sec_est, std_p_sec_est = estimate_price_statistics(logp_sec_est, std_logp_sec_est)
        p_mkt_est, std_p_mkt_est = estimate_price_statistics(logp_mkt_est, std_logp_mkt_est)

        variances = {}
        for i, ticker in enumerate(tickers):
            variances[ticker] = {"stock": std_p_est[i, -1] ** 2,
                                 "industry": std_p_ind_est[info['industries_id'][i], -1] ** 2,
                                 "sector": std_p_sec_est[info['sectors_id'][i], -1] ** 2,
                                 "market": std_p_mkt_est[0, -1] ** 2}

        scores = (logp_pred[:, horizon] - logp[:, -j - 1]) / std_logp_pred.squeeze()
        rates = rate(scores)
        growth = np.dot(phi.numpy()[:, 1:], np.arange(1, order + 1))

        bot_info = {tickers[i]: {"price": price[i, -j - 1], "rate": rates[i], "growth": growth[i], "scores": scores[i]} for i in range(num_stocks)}
        next_price = {tickers[i]: price[i, -j] for i in range(num_stocks)}

        print()
        print("DATE:", data['dates'].date[-j].strftime("%Y-%m-%d"))
        print(separator)
        print(str_format.format("BOT", "CAPITAL", "UNINVESTED", "INVESTED", "OWNED", "RISK"))
        print(separator)
        for i, (name, bot) in enumerate(tournament.items()):
            bot.trade(bot_info)
            bot.compute_capital(next_price)

            risks[i, num_days - j] = compute_risk(bot.portfolio, variances, data['sectors'], data['industries'])
            capitals[i, num_days - j] = bot.capital / xrate

            print(str_format.format(name,
                  "{} {}".format(np.round(bot.capital / xrate, 2), args.currency),
                  "{} {}".format(np.round(bot.uninvested / xrate, 2), args.currency),
                  "{} {}".format(np.round(bot.invested / xrate, 2), args.currency),
                  ' '.join(map(str, list(bot.portfolio.keys())[:10])) + ("..." if len(bot.portfolio.keys()) > 10 else ""),
                  np.round(risks[i, num_days - j], 2)))

        print(separator)
        print(separator)

    # plot capitals
    fig = plt.figure(figsize=(15, 8))
    plt.title("Capitals over time", fontsize=15)
    plt.plot(data['dates'][-num_days:], capitals.T)
    plt.legend(names, loc="upper left", fontsize=12)
    plt.xticks(rotation=45)
    plt.ylabel("capital in {}".format(args.currency))

    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig_name = 'plots/tournament_capitals.png'
    fig.savefig(fig_name, dpi=fig.dpi)
    print('Plot of capitals over time has been saved in {}/{}.'.format(os.getcwd(), fig_name))

    fig = plt.figure(figsize=(15, 8))
    plt.title("Portfolio risk over time", fontsize=15)
    plt.plot(data['dates'][-num_days:], risks.T)
    plt.legend(names, loc="upper left", fontsize=12)
    plt.xticks(rotation=45)
    plt.ylabel("risk")

    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig_name = 'plots/tournament_risks.png'
    fig.savefig(fig_name, dpi=fig.dpi)
    print('Plot of portfolio risk over time has been saved in {}/{}.'.format(os.getcwd(), fig_name))
