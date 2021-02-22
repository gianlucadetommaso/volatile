#!/usr/bin/env python
import sys
import numpy as np

def convert_currency(logp: np.array, xrate: np.array, type: str = "forward") -> np.array:
    """
    It converts currency of price in log-form. If `type=forward`, the conversion is from the original log-price currency
    to the one determined by the exchange rate. Vice versa if `type=backward`.

    Parameters
    ----------
    logp: np.array
        Log-price of a stock.
    xrate: np.array
        Exchange rate from stock currency to another one.
    type: str
        Conversion type. It can be either `forward` or `backward`.

    Returns
    -------
    It returns converted log-price.
    """
    if type == "forward":
        return logp + np.log(xrate)
    if type == "backward":
        return logp - np.log(xrate)
    raise Exception("Conversion type {} not recognised.".format(type))

class ProgressBar:
    """
    Bar displaying percentage progression.
    """

    def __init__(self, iterations, text='completed'):
        self.text = text
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '-'
        self.width = 50
        self.__update_amount(0)
        self.elapsed = 1

    def completed(self):
        if self.elapsed > self.iterations:
            self.elapsed = self.iterations
        self.update_iteration(1)
        print('\r' + str(self), end='')
        sys.stdout.flush()
        print()

    def animate(self, iteration=None):
        if iteration is None:
            self.elapsed += 1
        else:
            self.elapsed += iteration

        print('\r' + str(self), end='')
        sys.stdout.flush()
        self.update_iteration()

    def update_iteration(self, val=None):
        val = val if val is not None else self.elapsed / float(self.iterations)
        self.__update_amount(val * 100.0)
        self.prog_bar += '  %s of %s %s' % (
            self.elapsed, self.iterations, self.text)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * \
                        num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
                        (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)

def extract_hierarchical_info(sectors: dict, industries: dict) -> dict:
    """
    Extract information about sectors and industries useful to construct the probabilistic model.

    Parameters
    ----------
    sectors: dict
        Dict of sectors at stock-level.
    industries: dict
        Dict of industries at stock-level.

    Returns
    -------
    It returns a dictionary including the following keys:
    - num_sectors: number of unique sectors;
    - num_industries: number of unique industries;
    - sectors_id: a list of indices at stock-level, corresponding to the sector they belong to;
    - industries_id: a list of indices at stock-level, corresponding to the industries they belong to;
    - sector_industries_id; a list of indices at industry-level, corresponding to the sectors they belong to;
    - unique_sectors: array of unique sector names;
    - unique_industries: array of unique industry names.
    """
    # find unique names of sectors
    usectors = np.unique(list(sectors.values()))
    num_sectors = len(usectors)
    # provide sector IDs at stock-level
    sectors_id = [np.where(usectors == sector)[0][0] for sector in sectors.values()]
    # find unique names of industries and store indices
    uindustries, industries_idx = np.unique(list(industries.values()), return_index=True)
    num_industries = len(uindustries)
    # provide industry IDs at stock-level
    industries_id = [np.where(uindustries == industry)[0][0] for industry in industries.values()]
    # provide sector IDs at industry-level
    sector_industries_id = np.array(sectors_id)[industries_idx].tolist()

    # place relevant information in dictionary
    return dict(num_stocks=len(sectors), num_sectors=num_sectors, num_industries=num_industries, industries_id=industries_id,
                sectors_id=sectors_id, sector_industries_id=sector_industries_id, unique_sectors = usectors,
                unique_industries=uindustries)

def compute_risk(portfolio: dict, variances: dict, sectors: dict, industries: dict):
    """
    It computes a portfolio risk measure.

    Parameters
    ----------
    portfolio: dict
        A dictionary with tickers as keys and another dictionary as values. The latter must include the following pairs:
        - "units": number of owned units of the corresponding stock.
    variances: dict
        A dictionary with tickers as keys and another dictionary as values. The latter must include the following pairs:
        - "stock": variance at stock level;
        - "industry": variance at industry level;
        - "sector": variance at sector level;
        - "market: variance at market level.
    sectors:
        A dictionary of tickers as keys and corresponding sectors as values.
    industries:
        A dictionary of tickers as keys and corresponding industries as values.

    Returns
    -------
    risk: float
        A positive number indicating the computed risk of the portfolio.
    """
    risk = 0
    for t1 in portfolio:
        for t2 in portfolio:
            tmp = variances[t1]['market']
            if t1 == t2:
                tmp += variances[t1]['stock']
            if industries[t1] == industries[t2]:
                tmp += variances[t1]['industry']
                if sectors[t1] == sectors[t2]:
                    tmp += variances[t1]['sector']
            risk += portfolio[t1]['units'] * portfolio[t2]['units'] * tmp
    return np.sqrt(risk) / max(len(portfolio), 1)

