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

    def compute_capital(self, price: dict):
        self.invested = 0.0
        for ticker in self.portfolio:
            self.invested += self.portfolio[ticker]['number'] * price[ticker]
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
                transaction = self.portfolio[ticker]['number'] * info[ticker]['price']
                self.uninvested += transaction
                del self.portfolio[ticker]

        ## buy strategy
        for ticker in info:
            if ticker not in self.portfolio.keys() and info[ticker]['rate'] == "HIGHLY BELOW TREND":
                num = int(np.minimum(self.uninvested, self.capital / 30) // info[ticker]['price'])
                if num >= 1:
                    transaction = num * info[ticker]['price']
                    self.uninvested -= transaction
                    self.portfolio[ticker] = {"number": num, "purchase_price": info[ticker]['price']}

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
                transaction = self.portfolio[ticker]['number'] * info[ticker]['price']
                self.uninvested += transaction
                del self.portfolio[ticker]

        ## buy strategy
        for ticker in info:
            if ticker not in self.portfolio.keys() and info[ticker]['rate'] == "HIGHLY ABOVE TREND":
                num = int(np.minimum(self.uninvested, self.capital / 30) // info[ticker]['price'])
                if num >= 1:
                    transaction = num * info[ticker]['price']
                    self.uninvested -= transaction
                    self.portfolio[ticker] = {"number": num, "purchase_price": info[ticker]['price']}

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
                num = int(self.uninvested / count // info[ticker]['price'])
                if num >= 1:
                    transaction = num * info[ticker]['price']
                    self.uninvested -= transaction
                    self.portfolio[ticker] = {"number": num, "purchase_price": info[ticker]['price']}

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
                transaction = self.portfolio[ticker]['number'] * info[ticker]['price']
                self.uninvested += transaction
                del self.portfolio[ticker]

        ## buy strategy
        for ticker in info:
            if ticker not in self.portfolio.keys() and info[ticker]['rate'] in ["HIGHLY BELOW TREND", "BELOW TREND"]:
                num = int(np.minimum(self.uninvested, self.capital / 30) // info[ticker]['price'])
                if num >= 1:
                    transaction = num * info[ticker]['price']
                    self.uninvested -= transaction
                    self.portfolio[ticker] = {"number": num, "purchase_price": info[ticker]['price']}

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
                transaction = self.portfolio[ticker]['number'] * info[ticker]['price']
                self.uninvested += transaction
                del self.portfolio[ticker]

        ## buy strategy
        for ticker in info:
            if ticker not in self.portfolio.keys() and info[ticker]['rate'] in ["HIGHLY ABOVE TREND", "ABOVE TREND"]:
                num = int(np.minimum(self.uninvested, self.capital / 30) // info[ticker]['price'])
                if num >= 1:
                    transaction = num * info[ticker]['price']
                    self.uninvested -= transaction
                    self.portfolio[ticker] = {"number": num, "purchase_price": info[ticker]['price']}
