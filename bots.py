import numpy as np

class Bot:
    """
    This is a general bot structure.
    """
    def __init__(self, capital: float, portfolio: dict = None):
        self.capital = capital
        self.uninvested = capital
        self.invested = 0
        self.portfolio = {} if portfolio is None else portfolio

    def transact_capital(self, ticker, units: int, price: float, type: str):
        if type == "sell":
            transaction = units * price
            self.uninvested += transaction
            del self.portfolio[ticker]

        elif type == "buy":
            transaction = units * price
            self.uninvested -= transaction
            self.portfolio[ticker] = {"units": units, "purchase_price": price}

        else:
            raise Exception("Transaction type {} not recognised. Choose between `sell` and `buy`.".format(type))

    def compute_capital(self, price: dict):
        self.invested = 0.0
        for ticker in self.portfolio:
            self.invested += self.portfolio[ticker]['units'] * price[ticker]
        self.capital = self.uninvested + self.invested

class Adam(Bot):
    def __init__(self, capital: float, portfolio: dict = None):
        super().__init__(capital, portfolio)
        self.min_rel_profit = 0.03
        self.max_rel_loss = 0.1

    def trade(self, info: dict):
        ## sell strategy
        owned_stocks = list(self.portfolio.keys())
        for ticker in owned_stocks:
            purchase_price = self.portfolio[ticker]["purchase_price"]
            rel_margin = (info[ticker]['price'] - purchase_price) / purchase_price
            if rel_margin > self.min_rel_profit or rel_margin < -self.max_rel_loss:
                self.transact_capital(ticker, self.portfolio[ticker]['units'], info[ticker]['price'], type="sell")

        ## buy strategy
        for ticker in info:
            if ticker not in self.portfolio.keys() and info[ticker]['rate'] == "HIGHLY BELOW TREND":
                units = int(np.minimum(self.uninvested, self.capital / 30) // info[ticker]['price'])
                if units >= 1:
                    self.transact_capital(ticker, units, info[ticker]['price'], type="buy")

class Betty(Bot):
    def __init__(self, capital: float, portfolio: dict = None):
        super().__init__(capital, portfolio)
        self.min_rel_profit = 0.1
        self.max_rel_loss = 0.03

    def trade(self, info: dict):
        ## sell strategy
        owned_stocks = list(self.portfolio.keys())
        for ticker in owned_stocks:
            purchase_price = self.portfolio[ticker]["purchase_price"]
            rel_margin = (info[ticker]['price'] - purchase_price) / purchase_price
            if rel_margin > self.min_rel_profit or rel_margin < -self.max_rel_loss:
                self.transact_capital(ticker, self.portfolio[ticker]['units'], info[ticker]['price'], type="sell")

        ## buy strategy
        for ticker in info:
            if ticker not in self.portfolio.keys() and info[ticker]['rate'] == "HIGHLY ABOVE TREND":
                units = int(np.minimum(self.uninvested, self.capital / 30) // info[ticker]['price'])
                if units >= 1:
                    self.transact_capital(ticker, units, info[ticker]['price'], type="buy")

class Chris(Bot):
    def __init__(self, capital: float, portfolio: dict = None):
        super().__init__(capital, portfolio)
        self.buy_only = ["GOOGL", "AMZN", "AAPL", "MSFT", "FB"]

    def trade(self, info: dict):
        ## buy strategy
        buy_only = [ticker for ticker in self.buy_only if ticker in info]
        for ticker in buy_only:
            if ticker in info:
                count = np.maximum(1, len(buy_only) - len(self.portfolio))
                units = int(self.uninvested / count // info[ticker]['price'])
                if units >= 1:
                    self.transact_capital(ticker, units, info[ticker]['price'], type="buy")

class Dany(Bot):
    def __init__(self, capital: float, portfolio: dict = None):
        super().__init__(capital, portfolio)
        self.min_rel_profit = 0.1
        self.max_rel_loss = 0.2

    def trade(self, info: dict):
        ## sell strategy
        owned_stocks = list(self.portfolio.keys())
        for ticker in owned_stocks:
            purchase_price = self.portfolio[ticker]["purchase_price"]
            rel_margin = (info[ticker]['price'] - purchase_price) / purchase_price
            if rel_margin > self.min_rel_profit or rel_margin < -self.max_rel_loss:
                self.transact_capital(ticker, self.portfolio[ticker]['units'], info[ticker]['price'], type="sell")

        ## buy strategy
        for ticker in info:
            if ticker not in self.portfolio.keys() and info[ticker]['rate'] in ["HIGHLY BELOW TREND", "BELOW TREND"]:
                units = int(np.minimum(self.uninvested, self.capital / 30) // info[ticker]['price'])
                if units >= 1:
                    self.transact_capital(ticker, units, info[ticker]['price'], type="buy")

class Eddy(Bot):
    def __init__(self, capital: float, portfolio: dict = None):
        super().__init__(capital, portfolio)
        self.min_rel_profit = 0.2
        self.max_rel_loss = 0.1

    def trade(self, info: dict):
        ## sell strategy
        owned_stocks = list(self.portfolio.keys())
        for ticker in owned_stocks:
            purchase_price = self.portfolio[ticker]["purchase_price"]
            rel_margin = (info[ticker]['price'] - purchase_price) / purchase_price
            if rel_margin > self.min_rel_profit or rel_margin < -self.max_rel_loss:
                self.transact_capital(ticker, self.portfolio[ticker]['units'], info[ticker]['price'], type="sell")

        ## buy strategy
        for ticker in info:
            if ticker not in self.portfolio.keys() and info[ticker]['rate'] in ["HIGHLY ABOVE TREND", "ABOVE TREND"]:
                units = int(np.minimum(self.uninvested, self.capital / 30) // info[ticker]['price'])
                if units >= 1:
                    self.transact_capital(ticker, units, info[ticker]['price'], type="buy")

class Flora(Bot):
    def __init__(self, capital: float, portfolio: dict = None):
        super().__init__(capital, portfolio)
        self.min_rel_profit = 0.1
        self.max_rel_loss = 0.2

    def trade(self, info: dict):
        ## sell strategy
        owned_stocks = list(self.portfolio.keys())
        for ticker in owned_stocks:
            purchase_price = self.portfolio[ticker]["purchase_price"]
            rel_margin = (info[ticker]['price'] - purchase_price) / purchase_price
            if rel_margin > self.min_rel_profit or rel_margin < -self.max_rel_loss:
                self.transact_capital(ticker, self.portfolio[ticker]['units'], info[ticker]['price'], type="sell")

        ## buy strategy
        growths = np.array([info[ticker]['growth'] for ticker in info])
        idx = np.argsort(growths)[::-1]
        sorted_tickers = np.array(list(info.keys()))[idx]
        for ticker in sorted_tickers:
            if ticker not in self.portfolio.keys() and info[ticker]['rate'] == "ALONG TREND" and info[ticker]['growth'] >= 1:
                units = int(np.minimum(self.uninvested, self.capital / 30) // info[ticker]['price'])
                if units >= 1:
                    self.transact_capital(ticker, units, info[ticker]['price'], type="buy")