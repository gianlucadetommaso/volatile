import os.path
import pandas as pd
import requests
import multitasking
import json
import csv
import numpy as np

from tools import ProgressBar

def parse_quotes(data):
    timestamps = data["timestamp"]
    ohlc = data["indicators"]["quote"][0]
    closes, volumes = ohlc["close"], ohlc["volume"]
    try:
        adjclose = data["indicators"]["adjclose"][0]["adjclose"]
    except:
        adjclose = closes

    quotes = pd.DataFrame({"Adj Close": adjclose, "Volume": volumes})
    quotes.index = pd.to_datetime(timestamps, unit="s").date
    quotes.sort_index(inplace=True)

    return quotes


def _download_one(ticker: str, interval: str = "1d", period: str = "1y"):
    """
    Download historical data for a single ticker.

    Parameters
    ----------
    ticker: str
        Ticker for which to download historical information.
    interval: str
        Frequency between data.
    period: str
        Data period to download.

    Returns
    -------
    data: dict
        Scraped dictionary of information.
    """
    base_url = 'https://query1.finance.yahoo.com'

    params = dict(range=period, interval=interval.lower(), includePrePost=False)

    url = "{}/v8/finance/chart/{}".format(base_url, ticker)
    data = requests.get(url=url, params=params)

    if "Will be right back" in data.text:
        raise RuntimeError("*** YAHOO! FINANCE is currently down! ***\n")

    data = data.json()
    return data

def download(tickers: list, interval: str = "1d", period: str = "1y"):
    """
    Download historical data for tickers in the list.

    Parameters
    ----------
    tickers: list
        Tickers for which to download historical information.
    interval: str
        Frequency between data.
    period: str
        Data period to download.

    Returns
    -------
    data: dict
        Dictionary including the following keys:
        - tickers: list of tickers
        - logp: array of log-adjusted closing prices, shape=(num stocks, length period);
        - volume: array of volumes, shape=(num stocks, length period);
        - sectors: list of stock sectors;
        - industries: list stock industries.
    """
    tickers = tickers if isinstance(tickers, (list, set, tuple)) else tickers.replace(',', ' ').split()
    tickers = list(set([ticker.upper() for ticker in tickers]))

    data = {}
    si_columns = ["SYMBOL", "SECTOR", "INDUSTRY"]
    si_filename = "stock_info.csv"
    if not os.path.exists(si_filename):
        # create a .csv to store stock information
        with open(si_filename, 'w') as file:
            wr = csv.writer(file)
            for row in zip([[c] for c in si_columns]):
                wr.writerow(row)
    # load stock information file
    si = pd.read_csv(si_filename)
    missing_tickers = [ticker for ticker in tickers if ticker not in si['SYMBOL'].values]
    missing_si, na_si = {}, {}

    @multitasking.task
    def _download_one_threaded(ticker: str, interval: str = "1d", period: str = "1y"):
        """
        Download historical data for a single ticker with multithreading. Plus, it scrapes missing stock information.

        Parameters
        ----------
        ticker: str
            Ticker for which to download historical information.
        interval: str
            Frequency between data.
        period: str
            Data period to download.
        """
        data_one = _download_one(ticker, interval, period)

        try:
            data[ticker] = parse_quotes(data_one["chart"]["result"][0])

            if ticker in missing_tickers:
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
        _download_one_threaded(ticker, interval, period)
    multitasking.wait_for_tasks()

    progress.completed()

    if len(data) == 0:
        raise Exception("No symbol with full information is available.")

    data = pd.concat(data.values(), keys=data.keys(), axis=1)
    data.drop(columns=data.columns[data.isnull().sum(0) > 0.33 * data.shape[0]], inplace=True)
    data = data.fillna(method='bfill').fillna(method='ffill').drop_duplicates()

    info = zip(list(missing_si.keys()), [v['sector'] for v in missing_si.values()],
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

    return dict(tickers=tickers,
                dates=pd.to_datetime(data.index),
                logp=np.log(data.iloc[:, data.columns.get_level_values(1) == 'Adj Close'].to_numpy().T),
                volume=data.iloc[:, data.columns.get_level_values(1) == 'Volume'].to_numpy().T,
                sectors=[si[ticker]['SECTOR'] if ticker in si else "NA_" + ticker for ticker in tickers],
                industries=[si[ticker]['INDUSTRY'] if ticker in si else "NA_" + ticker for ticker in tickers])