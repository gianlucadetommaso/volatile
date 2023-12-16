#!/usr/bin/env python
import os.path
import pandas as pd
import requests
import multitasking
import json
import csv
import numpy as np
import datetime as dt
from typing import Union

from tools import ProgressBar

def download(tickers: list, start: Union[str, int] = None, end: Union[str, int] = None, interval: str = "1d") -> dict:
    """
    Download historical data for tickers in the list.

    Parameters
    ----------
    tickers: list
        Tickers for which to download historical information.
    start: str or int
        Start download data from this date.
    end: str or int
        End download data at this date.
    interval: str
        Frequency between data.

    Returns
    -------
    data: dict
        Dictionary including the following keys:
        - tickers: list of tickers
        - logp: array of log-adjusted closing prices, shape=(num stocks, length period);
        - volume: array of volumes, shape=(num stocks, length period);
        - sectors: dictionary of stock sector for each ticker;
        - industries: dictionary of stock industry for each ticker.
    """
    tickers = tickers if isinstance(tickers, (list, set, tuple)) else tickers.replace(',', ' ').split()
    tickers = list(set([ticker.upper() for ticker in tickers]))

    data = {}
    si_columns = ["SYMBOL", "CURRENCY", "SECTOR", "INDUSTRY"]
    si_filename = "stock_info.csv"
    if not os.path.exists(si_filename):
        # create a .csv to store stock information
        with open(si_filename, 'w') as file:
            wr = csv.writer(file)
            wr.writerow(si_columns)
    # load stock information file
    si = pd.read_csv(si_filename)
    missing_tickers = [ticker for ticker in tickers if ticker not in si['SYMBOL'].values]
    missing_si, na_si = {}, {}
    currencies = {}

    if end is None:
        end = int(dt.datetime.timestamp(dt.datetime.today()))
    elif type(end) is str:
        end = int(dt.datetime.timestamp(dt.datetime.strptime(end, '%Y-%m-%d')))
    if start is None:
        start = int(dt.datetime.timestamp(dt.datetime.today() - dt.timedelta(915)))
    elif type(start) is str:
        start = int(dt.datetime.timestamp(dt.datetime.strptime(start, '%Y-%m-%d')))

    @multitasking.task
    def _download_one_threaded(ticker: str, start: str, end: str, interval: str = "1d"):
        """
        Download historical data for a single ticker with multithreading. Plus, it scrapes missing stock information.

        Parameters
        ----------
        ticker: str
            Ticker for which to download historical information.
        interval: str
            Frequency between data.
        start: str
            Start download data from this date.
        end: str
            End download data at this date.
        """
        data_one = _download_one(ticker, start, end, interval)

        try:
            data_one = data_one["chart"]["result"][0]
            data[ticker] = _parse_quotes(data_one)

            if ticker in missing_tickers:
                currencies[ticker] = data_one['meta']['currency']
                try:
                    html = requests.get(url='https://finance.yahoo.com/quote/' + ticker).text
                    json_str = html.split('root.App.main =')[1].split('(this)')[0].split(';\n}')[0].strip()
                    info = json.loads(json_str)['context']['dispatcher']['stores']['QuoteSummaryStore']['summaryProfile']
                    assert (len(info['sector']) > 0) and (len(info['industry']) > 0)
                    missing_si[ticker] = dict(sector=info["sector"], industry=info["industry"])
                except:
                    pass
        except:
            pass
        progress.animate()

    num_threads = min([len(tickers), multitasking.cpu_count() * 2])
    multitasking.set_max_threads(num_threads)

    progress = ProgressBar(len(tickers), 'completed')

    for ticker in tickers:
        _download_one_threaded(ticker, start, end, interval)
    multitasking.wait_for_tasks()

    progress.completed()

    if len(data) == 0:
        raise Exception("No symbol with full information is available.")

    data = pd.concat(data.values(), keys=data.keys(), axis=1, sort=True)
    data.drop(columns=data.columns[data.isnull().sum(0) > 0.33 * data.shape[0]], inplace=True)
    data = data.fillna(method='bfill').fillna(method='ffill').drop_duplicates()

    info = zip(list(missing_si.keys()), [currencies[ticker] for ticker in missing_si.keys()],
                                        [v['sector'] for v in missing_si.values()],
                                        [v['industry'] for v in missing_si.values()])
    with open(si_filename, 'a+', newline='') as file:
        wr = csv.writer(file)
        for row in info:
            wr.writerow(row)
    si = pd.read_csv('stock_info.csv').set_index("SYMBOL").to_dict(orient='index')

    missing_tickers = [ticker for ticker in tickers if ticker not in data.columns.get_level_values(0)[::2].tolist()]
    tickers = data.columns.get_level_values(0)[::2].tolist()
    if len(missing_tickers) > 0:
        print('\nRemoving {} from list of symbols because we could not collect full information.'.format(missing_tickers))

    # download exchange rates and convert to most common currency
    currencies = [si[ticker]['CURRENCY'] if ticker in si else currencies[ticker] for ticker in tickers]
    ucurrencies, counts = np.unique(currencies, return_counts=True)
    default_currency = ucurrencies[np.argmax(counts)]
    xrates = get_exchange_rates(currencies, default_currency, data.index, start, end, interval)

    return dict(tickers=tickers,
                dates=pd.to_datetime(data.index),
                price=data.iloc[:, data.columns.get_level_values(1) == 'Adj Close'].to_numpy().T,
                volume=data.iloc[:, data.columns.get_level_values(1) == 'Volume'].to_numpy().T,
                currencies=currencies,
                exchange_rates=xrates,
                default_currency=default_currency,
                sectors={ticker: si[ticker]['SECTOR'] if ticker in si else "NA_" + ticker for ticker in tickers},
                industries={ticker: si[ticker]['INDUSTRY'] if ticker in si else "NA_" + ticker for ticker in tickers})

def _download_one(ticker: str, start: int, end: int, interval: str = "1d") -> dict:
    """
    Download historical data for a single ticker.

    Parameters
    ----------
    ticker: str
        Ticker for which to download historical information.
    start: int
        Start download data from this timestamp date.
    end: int
        End download data at this timestamp date.
    interval: str
        Frequency between data.

    Returns
    -------
    data: dict
        Scraped dictionary of information.
    """
    base_url = 'https://query1.finance.yahoo.com'
    params = dict(period1=start, period2=end, interval=interval.lower(), includePrePost=False)
    url = "{}/v8/finance/chart/{}".format(base_url, ticker)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    data = requests.get(url=url, params=params, headers=headers)
    if "Will be right back" in data.text:
        raise RuntimeError("*** YAHOO! FINANCE is currently down! ***\n")
    data = data.json()
    return data


def _parse_quotes(data: dict, parse_volume: bool = True) -> pd.DataFrame:
    """
    It creates a data frame of adjusted closing prices, and, if `parse_volume=True`, volumes. If no adjusted closing
    price is available, it sets it equal to closing price.

    Parameters
    ----------
    data: dict
        Data containing historical information of corresponding stock.
    parse_volume: bool
        Include or not volume information in the data frame.
    """
    timestamps = data["timestamp"]
    ohlc = data["indicators"]["quote"][0]
    closes = ohlc["close"]
    if parse_volume:
        volumes = ohlc["volume"]
    try:
        adjclose = data["indicators"]["adjclose"][0]["adjclose"]
    except:
        adjclose = closes

    # fix NaNs in the second-last entry of adjusted closing prices
    if adjclose[-2] is None:
        adjclose[-2] = adjclose[-1]

    assert (np.array(adjclose) > 0).all()

    quotes = {"Adj Close": adjclose}
    if parse_volume:
        quotes["Volume"] = volumes
    quotes = pd.DataFrame(quotes)
    quotes.index = pd.to_datetime(timestamps, unit="s").date
    quotes.sort_index(inplace=True)
    quotes = quotes.loc[~quotes.index.duplicated(keep='first')]

    return quotes

def get_exchange_rates(from_currencies: list, to_currency: str, dates: pd.Index, start: Union[str, int] = None,
                       end: Union[str, int] = None, interval: str = "1d") -> dict:
    """
    It finds the most common currency and set it as default one. For any other currency, it downloads exchange rate
    closing prices to the default currency and return them as data frame.

    Parameters
    ----------
    from_currencies: list
        A list of currencies to convert.
    to_currency: str
        Currency to convert to.
    dates: date
        Dates for which exchange rates should be available.
    start: str or int
        Start download data from this timestamp date.
    end: str or int
        End download data at this timestamp date.
    interval: str
        Frequency between data.

    Returns
    -------
    xrates: dict
        A dictionary with currencies as keys and list of exchange rates at desired dates as values.
    """
    if end is None:
        end = int(dt.datetime.timestamp(dt.datetime.today()))
    elif type(end) is str:
        end = int(dt.datetime.timestamp(dt.datetime.strptime(end, '%Y-%m-%d')))
    if start is None:
        start = int(dt.datetime.timestamp(dt.datetime.today() - dt.timedelta(915)))
    elif type(start) is str:
        start = int(dt.datetime.timestamp(dt.datetime.strptime(start, '%Y-%m-%d')))

    ucurrencies, counts = np.unique(from_currencies, return_counts=True)
    tmp = {}
    if to_currency not in ucurrencies or len(ucurrencies) > 1:
        for curr in ucurrencies:
            if curr != to_currency:
                tmp[curr] = _download_one(curr + to_currency + "=x", start, end, interval)
                tmp[curr] = _parse_quotes(tmp[curr]["chart"]["result"][0], parse_volume=False)["Adj Close"]
        tmp = pd.concat(tmp.values(), keys=tmp.keys(), axis=1, sort=True)
        xrates = pd.DataFrame(index=dates, columns=tmp.columns)
        xrates.loc[xrates.index.isin(tmp.index)] = tmp
        xrates = xrates.fillna(method='bfill').fillna(method='ffill')
        xrates.to_dict(orient='list')
    else:
        xrates = tmp
    return xrates
