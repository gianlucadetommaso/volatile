#!/usr/bin/env python
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, SUPPRESS
import pandas as pd
import matplotlib.pyplot as plt
import os

from tools import convert_currency, extract_hierarchical_info

from download import download, get_exchange_rates
from volatile import softplus, train, rate
from bots import *

if __name__ == '__main__':
    cli = ArgumentParser('Volatile Bot-Tournament.', formatter_class=ArgumentDefaultsHelpFormatter)
    cli.add_argument('-s', '--symbols', type=str, nargs='+', help=SUPPRESS)
    cli.add_argument('--capital', type=float, default=100000.0, help='Bots start with this available capital at '
                                                                           'the beginning of the tournament. ')
    cli.add_argument('--currency', type=str, default='USD', help='Currency of the capital in input.')
    cli.add_argument('--days', type=int, default=30, help='Number of days the tournament runs for. It should be an')
    args = cli.parse_args()

    if args.capital < 0:
        raise Exception("Capital must be a non-negative number.")
    if args.days < 1:
        raise Exception("Number of tournament days must be an integer greater than 0.")

    print('\nDownloading all available closing prices in the last year...')
    if args.symbols is None:
        with open("symbols_list.txt", "r") as my_file:
            args.symbols = my_file.readlines()[0].split(" ")

    # download data
    data = download(args.symbols)
    tickers = data["tickers"]
    price = data['price']
    logp = np.log(price)

    # convert currencies to most frequent one
    for i, curr in enumerate(data['currencies']):
        if curr != data['default_currency']:
            logp[i] = convert_currency(logp[i], np.array(data['exchange_rates'][curr]), type='forward')
            price[i] = np.exp(logp[i])

    # convert initial capital
    if args.currency == data['default_currency']:
        xrate = 1.0
    elif args.currency in data['exchange_rates']:
        xrate = np.array(data['exchange_rates'][args.currency])[-args.days]
    else:
        xrate = get_exchange_rates([args.currency], data['default_currency'], data['dates'])[args.currency][-args.days]
    args.capital *= xrate

    # Volatile specifics
    num_stocks, t = logp.shape
    order = 2
    horizon = 5

    # extract hierarchical info
    info = extract_hierarchical_info(data['sectors'], data['industries'])
    info['order_scale'] = np.linspace(1 / (order + 1), 1, order + 1)[::-1].astype('float32')[None, :]

    # tournament participants
    names = ["Adam", "Betty", "Chris", "Dany", "Eddy"]
    tournament = {name: globals()[name](args.capital) for name in names}

    # initialize capitals
    uninvested = np.zeros((len(names), args.days))
    invested = np.zeros((len(names), args.days))
    capitals = np.zeros((len(names), args.days))

    str_format = "{:<20} {:<25} {:<25} {:<25} {:<80}"
    num_dashes = 175
    separator = num_dashes * "-"

    print("\n*** LET'S THE BOT-TOURNAMENT BEGINS! ***\n")
    for j in range(args.days, 0, -1):
        t = logp[:, :-j].shape[1]
        info['tt'] = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
        tt_pred = ((1 + (np.arange(1 + horizon) / t)) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')

        phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, phi, psi = train(logp[:, :-j], info)

        logp_pred = np.dot(phi.numpy(), tt_pred)
        std_logp_pred = softplus(psi.numpy())
        scores = (logp_pred[:, horizon] - logp[:, -j - 1]) / std_logp_pred.squeeze()
        rates = rate(scores)

        bot_info = {tickers[i]: {"price": price[i, -j - 1], "rate": rates[i]} for i in range(num_stocks)}
        next_price = {tickers[i]: price[i, -j] for i in range(num_stocks)}

        print()
        print("DATE:", data['dates'].date[-j].strftime("%Y-%m-%d"))
        print(separator)
        print(str_format.format("BOT", "CAPITAL", "UNINVESTED", "INVESTED", "OWNED"))
        print(separator)
        for i, (name, bot) in enumerate(tournament.items()):
            bot.trade(bot_info)
            bot.compute_capital(next_price)
            print(str_format.format(name,
                  "{} {}".format(np.round(bot.capital / xrate, 2), args.currency),
                  "{} {}".format(np.round(bot.uninvested / xrate, 2), args.currency),
                  "{} {}".format(np.round(bot.invested / xrate, 2), args.currency),
                  ' '.join(map(str, list(bot.portfolio.keys())[:10])) + ("..." if len(bot.portfolio.keys()) > 10 else "")))
            capitals[i, args.days - j] = bot.capital / xrate
        print(separator)
        print(separator)

    # plot capitals
    fig = plt.figure(figsize=(15, 8))
    plt.title("Capitals over time", fontsize=15)
    plt.plot(data['dates'][-args.days:], capitals.T)
    plt.legend(names, loc="upper left", fontsize=12)
    plt.xticks(rotation=45)
    plt.ylabel("capital in {}".format(args.currency))
    fig_name = 'tournament_capitals.png'
    fig.savefig(fig_name, dpi=fig.dpi)
    print('Plot of capitals over time has been saved in {}/{}.'.format(os.getcwd(), fig_name))
