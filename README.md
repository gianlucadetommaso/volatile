<table>
<tr>
<td width=170>
  
![b1d3115b-97e1-4703-b382-49fc786b9e19_200x200](https://user-images.githubusercontent.com/32386694/100524005-e02f4280-31b4-11eb-9765-a53c138929d9.png)

</td>
<td>

# Volatile 
### Your day-to-day trading companion.
The word "volatile" comes from the Latin *volatilis*, meaning "having wings" or "able to fly". With time, the financial market adopted it to describe asset price variability over time. Here, Volatile becomes a "trading companion", designed to help you every day to make unemotional, algorithmic-based, trading decisions.

</td>
</tr>
</table>

If you expect Volatile to predict the unpredictable, you are in the wrong place. Be reasonable: this is a swing trading software, runnable on your laptop, aimed to quickly discover out-of-trend opportunities by comparing current stock prices to their projections in a few days. If the current price is much lower than its future projection, perhaps it is a good opportunity to buy; vice versa, if it is much higher, perhaps it is a good moment to sell. This does neither mean the projection will be necessarily met, nor that you will make a short-term profit for every single transaction you make. Anything could happen. However, running Volatile on a daily basis will put you in condition to very quickly survey the market, find good opportunities and base your trading decisions on models, algorithms and data. 

### What to expect and how to use
Volatiles estimates stock trends, predict short-term future prices, then ranks and rates. All you need to do to run Volatile is to open your terminal and type
```ruby
python volatile.py
```
Volatile will automatically analyse the list of stock symbols saved in `symbols_list.txt`. This should neither be considered to be a privileged nor a complete list of stocks; feel free to update it as you please (do not worry if by chance you enter a symbol twice). Mind that it can take a while to access information of stock symbols that are either not in the list or that you pass for the first time. For this reason, relevant stock information is stored in `stock_info.csv` and will be fast to access from the second time onwards.

When the run is complete, a prediction table like the following will appear printed on your shell:

<img width="929" alt="Screenshot 2021-01-19 at 16 28 27" src="https://user-images.githubusercontent.com/32386694/105055893-7c661f00-5a6b-11eb-8d21-932b7b93665d.png">

For each symbol, the table tells you its sector and industry, then the last available price and finally a rating. Possible ratings are HIGHLY ABOVE TREND, ABOVE TREND, ALONG TREND, BELOW TREND and HIGHLY BELOW TREND. Symbols appear in the table ranked from the furthest below to the furthest above their respective trends. Ranking and rating are derived from a score metric that compares the predicted price in 5 trading days (usually this corresponds to the price in one week) to the last available observed price, scaling by the standard deviation of the prediction; see the technical part below for more details. The prediction table can be saved in the current directory as `prediction_table.csv` by adding the following flag to the command above: `--save-table`.

In the current directory, several estimation plots will appear. `stock_estimation.png` is a visualisation of stock prices and their estimations over the last year, together with a notion of uncertainty and daily trading volume. Only stocks rated either above or below their trends will be plotted, ranked as in the prediction table. Notice how the estimation crucially attempts to reproduce the trend of a stock but not to learn its noise. The uncertainty, on the other hand, depends on the stock volatility; the smaller the volatility, the more confident we are about our estimates, the more a sudden shift from the trend will be regarded as significant. You can use this plot as a sanity check that the estimation procedure agrees with your intuition. Make sure to glance at it before any transaction.

<img width="994" alt="Screenshot 2021-01-19 at 16 29 26" src="https://user-images.githubusercontent.com/32386694/105055924-825c0000-5a6b-11eb-9936-de49abc7a151.png">

 `sector_estimation.png` and `industry_estimation.png` are plots that help you to quickly visualise estimated sector and industry performances. A sector estimate can be thought as the average behaviour of its belonging industries, which in turn should be regarded as the average behaviour of its belonging stocks. Both sectors and industries are ranked in alphabetical order. 
 
<img width="1329" alt="Screenshot 2021-01-19 at 16 30 31" src="https://user-images.githubusercontent.com/32386694/105055907-7ec87900-5a6b-11eb-88c5-8e152397a1d6.png">

<img width="994" alt="Screenshot 2021-01-19 at 16 29 59" src="https://user-images.githubusercontent.com/32386694/105055915-80923c80-5a6b-11eb-8d18-35782a9a527e.png">

Finally,  `market_estimation.png` shows the overall estimated market trend. This can be considered as the average of the sector estimates. Use this plot to immediately know in what phase the stock market currently is.

<img width="923" alt="Screenshot 2021-01-19 at 16 30 16" src="https://user-images.githubusercontent.com/32386694/105055901-7d974c00-5a6b-11eb-8ece-16d6a99ac5ff.png">

If you do not want plots to be saved in the current directory, you can disable them by adding the flag `--no-plots`.

### Bot-tournament
In order to argue whether the information provided by Volatile were any useful in the past, we offer the possibility to run a "tournament", where a set of bots trades daily according to some pre-fixed strategies, which are described further below. Please mind that the tournament is a *simplified* market scenario where bots are allowed to buy and sell once a day, only and exactly at adjusted closing prices. Price variations intra-trading session, transaction fees, slippage and dividends are currently not simulated. You can run the tournament as follows:
```ruby
python tournament.py
```
By default, the tournament runs for 30 days of trading. You can change it, for example, to 10 days by adding the flag `--days 10` to the command above. Furthermore, bots start with an initial capital of 100000 USD; if you instead wanted, let us say, 5000 EUR, you can add the flags `--capital 5000` and `--currency EUR`. 

While the tournament is running, you will be able to see its current state parsed in your shell. For example:

<img width="1113" alt="Screenshot 2021-01-24 at 19 41 25" src="https://user-images.githubusercontent.com/32386694/105640211-b7ee5800-5e74-11eb-9418-fae506124fde.png">

For every day of the tournament and each bot, you can see its total capital, how much of it is invested and uninvested, and a list of stock the bot owns. When the tournament is over, a plot of capitals over time like the following is saved in the current directory as `tournament_capitals.png`.

<img height="350" alt="Screenshot 2021-01-24 at 19 46 41" src="https://user-images.githubusercontent.com/32386694/105640213-ba50b200-5e74-11eb-8214-913d96257301.png">

In this case, Betty notably leads the way, making over 35% of her initial capital in only 30 days of trading.

### How to install
The easiest way to use Volatile is to:
- open [this](https://raw.githubusercontent.com/gianlucadetommaso/volatile/master/Volatile.ipynb) notebook;
- depending on your OS, press `ctrl+s` or `cmd+s` to save it as a `.ipynb` file (make sure not to save it as a `.txt` file, which is the default option);
- upload the notebook on [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) and run it. 

Alternatively, you can download Volatile locally. First, open a terminal and go to the directory where you intend to install Volatile. On a Mac or Linux, you can do so by typing
```ruby
cd path/to/your/directory
```
If you are fine installing Volatile in your home directory, instead of the command before you can just type ``cd``. Then, download Volatile from Github and get in its main directory by typing
```ruby
git clone https://github.com/gianlucadetommaso/volatile.git
cd volatile
```
We recommend to activate a virtual environment. Type
```ruby
pip install virtualenv
virtualenv venv 
source venv/bin/activate
```
Now that you are in your virtual environment, install the dependencies:
```ruby
pip install -r requirements.txt
```

**Important**: Volatile depends on Tensorflow, which is currently supported only up to Python 3.8, not yet Python 3.9 (see [here](https://www.tensorflow.org/install/pip)); make sure to activate the virtual environment with the right Python version.

Done! You're all set to use Volatile. 

### Behind the scenes (technical)
**Model description.** Volatile adopts a Bayesian hierarchical model based on adjusted closing prices, sector and industry information, estimating log-price via polynomials in time. 

Denote <a href="https://www.codecogs.com/eqnedit.php?latex=\tau_t=t/T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau_t=t/T" title="\tau_t=t/T" /></a> to represent times at which observations arrive. <a href="https://www.codecogs.com/eqnedit.php?latex=T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T" title="T" /></a> corresponds to the number of days in the training dataset, which is taken to be the last one year of data.

Furthermore, denote <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma_j=1&space;-&space;\tfrac{j}{D&plus;1},\text{&space;for&space;}j=0,\dots,D" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma_j=1&space;-&space;\tfrac{j}{D&plus;1},\text{&space;for&space;}j=0,\dots,D" title="\gamma_j=1 - \tfrac{j}{D+1},\text{ for }j=0,\dots,D" /></a> to be prior scale parameters associated to the j-th order of a polynomial with degree <a href="https://www.codecogs.com/eqnedit.php?latex=D" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D" title="D" /></a>. Decreasing the scales as <a href="https://www.codecogs.com/eqnedit.php?latex=j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?j" title="j" /></a> increases penalises deviation from zero of higher-order parameters, thereby encouraging simpler models. Currently, we set <a href="https://www.codecogs.com/eqnedit.php?latex=D=2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D=2" title="D=2" /></a>.

We write:
- <a href="https://www.codecogs.com/eqnedit.php?latex=\text{sec}(\ell)=k,\text{&space;for&space;}\ell=1,\dots,&space;L\text{&space;and&space;}k=1\dots,K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{sec}(\ell)=k,\text{&space;for&space;}\ell=1,\dots,&space;L\text{&space;and&space;}k=1\dots,K" title="\text{sec}(\ell)=k,\text{ for }\ell=1,\dots, L\text{ and }k=1\dots,K" /></a> to indicate that an industry <a href="https://www.codecogs.com/eqnedit.php?latex=\ell" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\ell" title="\ell" /></a> belongs to a sector <a href="https://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k" title="k" /></a>, where <a href="https://www.codecogs.com/eqnedit.php?latex=L" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L" title="L" /></a> is the number of industries and <a href="https://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?K" title="K" /></a> the number of sectors;
- <a href="https://www.codecogs.com/eqnedit.php?latex=\text{ind}(i)=\ell,\text{&space;for&space;}i=1,\dots,&space;N\text{&space;and&space;}\ell=1\dots,L" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{ind}(i)=\ell,\text{&space;for&space;}i=1,\dots,&space;N\text{&space;and&space;}\ell=1\dots,L" title="\text{ind}(i)=\ell,\text{ for }i=1,\dots, N\text{ and }\ell=1\dots,L" /></a> to indicate that a stock <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a> belongs to an industry <a href="https://www.codecogs.com/eqnedit.php?latex=\ell" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\ell" title="\ell" /></a>, where <a href="https://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N" title="N" /></a> the number of stocks.

Then, we construct the hierarchical model

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{align*}\phi^m_j&space;&\sim&space;\mathcal{N}(0,\&space;16\gamma_j^2)\\&space;\phi^s_{k,j}&space;&\sim&space;\mathcal{N}(\phi^m_j,\&space;4\gamma_j^2)\\&space;\phi^\iota_{\ell,j}&space;&\sim&space;\mathcal{N}(\phi^s_{\text{sec}(\ell),j},\&space;\gamma_j^2)\\&space;\phi_{i,j}&space;&\sim&space;\mathcal{N}(\phi^\iota_{\text{ind}(i),j},\&space;\tfrac{1}{4}\gamma_j^2)\\&space;\psi^m&space;&\sim&space;\mathcal{N}(0,\&space;16)\\&space;\psi_k^s&space;&\sim&space;\mathcal{N}(\psi^m,\&space;4)\\&space;\psi^\iota_{\ell}&space;&\sim&space;\mathcal{N}(\psi_{\text{sec}(\ell)}^s,\&space;1)\\&space;\psi_{i}&space;&\sim&space;\mathcal{N}(\psi^\iota_{\text{ind}(i)},\&space;\tfrac{1}{4})\\&space;y_{t,i}&space;&\sim&space;\mathcal{N}\left(\sum_{j=0}^{D}\phi_{i,j}\,\tau_t^j,&space;\text{softplus}^2(\psi_i)\right)\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{align*}\phi^m_j&space;&\sim&space;\mathcal{N}(0,\&space;16\gamma_j^2)\\&space;\phi^s_{k,j}&space;&\sim&space;\mathcal{N}(\phi^m_j,\&space;4\gamma_j^2)\\&space;\phi^\iota_{\ell,j}&space;&\sim&space;\mathcal{N}(\phi^s_{\text{sec}(\ell),j},\&space;\gamma_j^2)\\&space;\phi_{i,j}&space;&\sim&space;\mathcal{N}(\phi^\iota_{\text{ind}(i),j},\&space;\tfrac{1}{4}\gamma_j^2)\\&space;\psi^m&space;&\sim&space;\mathcal{N}(0,\&space;16)\\&space;\psi_k^s&space;&\sim&space;\mathcal{N}(\psi^m,\&space;4)\\&space;\psi^\iota_{\ell}&space;&\sim&space;\mathcal{N}(\psi_{\text{sec}(\ell)}^s,\&space;1)\\&space;\psi_{i}&space;&\sim&space;\mathcal{N}(\psi^\iota_{\text{ind}(i)},\&space;\tfrac{1}{4})\\&space;y_{t,i}&space;&\sim&space;\mathcal{N}\left(\sum_{j=0}^{D}\phi_{i,j}\,\tau_t^j,&space;\text{softplus}^2(\psi_i)\right)\end{align*}" title="\begin{align*}\phi^m_j &\sim \mathcal{N}(0,\ 16\gamma_j^2)\\ \phi^s_{k,j} &\sim \mathcal{N}(\phi^m_j,\ 4\gamma_j^2)\\ \phi^\iota_{\ell,j} &\sim \mathcal{N}(\phi^s_{\text{sec}(\ell),j},\ \gamma_j^2)\\ \phi_{i,j} &\sim \mathcal{N}(\phi^\iota_{\text{ind}(i),j},\ \tfrac{1}{4}\gamma_j^2)\\ \psi^m &\sim \mathcal{N}(0,\ 16)\\ \psi_k^s &\sim \mathcal{N}(\psi^m,\ 4)\\ \psi^\iota_{\ell} &\sim \mathcal{N}(\psi_{\text{sec}(\ell)}^s,\ 1)\\ \psi_{i} &\sim \mathcal{N}(\psi^\iota_{\text{ind}(i)},\ \tfrac{1}{4})\\ y_{t,i} &\sim \mathcal{N}\left(\sum_{j=0}^{D}\phi_{i,j}\,\tau_t^j, \text{softplus}^2(\psi_i)\right)\end{align*}" /></a>

Parameters at market-level <a href="https://www.codecogs.com/eqnedit.php?latex=\phi^m\text{&space;and&space;}\psi^m" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi^m\text{&space;and&space;}\psi^m" title="\phi^m\text{ and }\psi^m" /></a> are prior means for sector-level parameters <a href="https://www.codecogs.com/eqnedit.php?latex=\phi^s\text{&space;and&space;}\psi^s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi^s\text{&space;and&space;}\psi^s" title="\phi^s\text{ and }\psi^s" /></a> , which in turn are prior means for industry-level parameters <a href="https://www.codecogs.com/eqnedit.php?latex=\phi^\iota\text{&space;and&space;}\psi^\iota" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi^\iota\text{&space;and&space;}\psi^\iota" title="\phi^\iota\text{ and }\psi^\iota" /></a> ; finally, the latter are prior means for stock-level parameters <a href="https://www.codecogs.com/eqnedit.php?latex=\phi\text{&space;and&space;}\psi." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi\text{&space;and&space;}\psi." title="\phi\text{ and }\psi." /></a> Components of the parameters at each level are supposed to be conditionally independent given the parameters at the level above in the hierarchy. Whereas <a href="https://www.codecogs.com/eqnedit.php?latex=\phi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /></a> are used to determine the coefficients of the polynomial model,  <a href="https://www.codecogs.com/eqnedit.php?latex=\psi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\psi" title="\phi" /></a> are used to determine the scales of the likelihood function.

**Inference.** In order to estimate parameters, we condition on adjusted closing log-prices <a href="https://www.codecogs.com/eqnedit.php?latex=y_{t,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{t,i}" title="y_{t,i}" /></a>, for all <a href="https://www.codecogs.com/eqnedit.php?latex=t=1,\dots&space;T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t=1,\dots&space;T" title="t=1,\dots T" /></a>, then we estimate the mode of the posterior distribution, also known as Maximum-A-Posteriori (MAP). From a frequentist statistics perspective, this corresponds to a polynomial regression task where we minimise a regularised mean-squared error loss. In practice, we train the model sequentially at different levels, that is first we train a market-level model to find market-level parameters; then we fix the market-level parameters and train a sector-level model to find sector-level parameters; and so on. A plot showing the losses decay during training can be saved in the current directory as `losses_decay.png` by adding the flag `--plot-losses` in the command line.

**Stock-level estimates.** Obtained our estimates <a href="https://www.codecogs.com/eqnedit.php?latex=\hat\phi^m,\hat\phi^s,\hat\phi^\iota,\hat\phi,\hat\psi^m,\hat\psi^s,\hat\psi^\iota\text{&space;and&space;}&space;\hat\psi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat\phi^m,\hat\phi^s,\hat\phi^\iota,\hat\phi,\hat\psi^m,\hat\psi^s,\hat\psi^\iota\text{&space;and&space;}&space;\hat\psi" title="\hat\phi^m,\hat\phi^s,\hat\phi^\iota,\hat\phi,\hat\psi^m,\hat\psi^s,\hat\psi^\iota\text{ and } \hat\psi" /></a>, we can use the likelihood mean <a href="https://www.codecogs.com/eqnedit.php?latex=\hat&space;y_{t,i}=\sum_{j=0}^{D}\hat\phi_{i,j}\,\tau_t^j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat&space;y_{t,i}=\sum_{j=0}^{D}\hat\phi_{i,j}\,\tau_t^j" title="\hat y_{t,i}=\sum_{j=0}^{D}\hat\phi_{i,j}\,\tau_t^j" /></a> as an estimator of the log-prices for any time in the past, as well as a predictor for times in the short future. As a measure of uncertainty, we take the learned scale of the likelihood, that is <a href="https://www.codecogs.com/eqnedit.php?latex=\hat\sigma_i=\text{softplus}(\psi_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat\sigma_i=\text{softplus}(\psi_i)" title="\hat\sigma_i=\text{softplus}(\psi_i)" /></a>.

**Ranking and rating.** Given the selected model complexity, Volatile trains the model and provides a rating for each stock by introducing the following score:

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{score}_i=\frac{\hat&space;y_{T&plus;5,i}-y_{T,i}}{\hat\sigma_i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{score}_i=\frac{\hat&space;y_{T&plus;5,i}-y_{T,i}}{\hat\sigma_i}" title="\text{score}_i=\frac{\hat y_{T+5,i}-y_{T,i}}{\hat\sigma_i}" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;y_{T,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;y_{T,i}" title="y_{T,i}" /></a> is the last available log-price and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\hat&space;y_{T&plus;5,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat&space;y_{T&plus;5,i}" title="\hat y_{T+5,i}" /></a> is its prediction in 5 trading days (usually, that corresponds to the log-price in one week). If the future prediction is larger than the current price, the score will be positive; the larger the difference and the more confident we are about the prediction (or equivalently, the smaller the standard deviation is), the more positive will be the score. We can reason similarly if the score is negative. In other words, a large positive score indicates that the current price is undervalued with respect to its stock trend, therefore an opportunity to buy; a large negative score indicates, vice versa, that the current price is overvalued with respect to its stock trend, therefore a moment to sell. 

Then, stocks are rated according to the following criteria:
- HIGHLY BELOW TREND if <a href="https://www.codecogs.com/eqnedit.php?latex=\text{score}_i>3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{score}_i>3" title="\text{score}_i>3" /></a>; 
- BELOW TREND if <a href="https://www.codecogs.com/eqnedit.php?latex=2<\text{score}_i<=3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2<\text{score}_i<=3" title="2<\text{score}_i<=3" /></a>;
- ALONG TREND if <a href="https://www.codecogs.com/eqnedit.php?latex=-2<\text{score}_i<=2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?-2<\text{score}_i<=2" title="-2<\text{score}_i<=2" /></a>;
- ABOVE TREND if <a href="https://www.codecogs.com/eqnedit.php?latex=-3<\text{score}_i<=-2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?-3<\text{score}_i<=-2" title="-3<\text{score}_i<=-2" /></a>;
- HIGHLY ABOVE TREND if <a href="https://www.codecogs.com/eqnedit.php?latex=\text{score}_i<=-3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{score}_i<=-3" title="\text{score}_i<=-3" /></a>.

**Estimators for all hierarchy levels.** Because we model log-prices as a Gaussian, the distribution of prices is a log-Normal distribution, whose mean and standard deviation can be derived in closed form from the estimators <a href="https://www.codecogs.com/eqnedit.php?latex=\hat&space;y_{t,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat&space;y_{t,i}" title="\hat y_{t,i}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\hat\sigma_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat\sigma_i" title="\hat\sigma_i" /></a>. We use log-Normal distribution statistics at times <a href="https://www.codecogs.com/eqnedit.php?latex=t=1\dots,T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t=1\dots,T" title="t=1\dots,T" /></a> to produce the stock estimation plot and at time <a href="https://www.codecogs.com/eqnedit.php?latex=T&plus;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T&plus;1" title="T+1" /></a> to fill the prediction table. In order to produce the market, sector and industry estimation plots, we proceed analogously but with estimators at respective levels, that is <a href="https://www.codecogs.com/eqnedit.php?latex=\hat&space;y^m_{t}=\sum_{j=0}^{D}\hat\phi^m_{j}\,\tau_t^j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat&space;y^m_{t}=\sum_{j=0}^{D}\hat\phi^m_{j}\,\tau_t^j" title="\hat y^m_{t}=\sum_{j=0}^{D}\hat\phi^m_{j}\,\tau_t^j" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\hat\sigma^m=\text{softplus}(\psi^m)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat\sigma^m=\text{softplus}(\psi^m)" title="\hat\sigma^m=\text{softplus}(\psi^m)" /></a> for market, <a href="https://www.codecogs.com/eqnedit.php?latex=y^s_{t,k}=\sum_{j=0}^{D}\hat\phi^s_{k,j}\,\tau_t^j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y^s_{t,k}=\sum_{j=0}^{D}\hat\phi^s_{k,j}\,\tau_t^j" title="y^s_{t,k}=\sum_{j=0}^{D}\hat\phi^s_{k,j}\,\tau_t^j" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\hat\sigma^s_k=\text{softplus}(\psi^s_k)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat\sigma^s_k=\text{softplus}(\psi^s_k)" title="\hat\sigma^s_k=\text{softplus}(\psi^s_k)" /></a> for sector, <a href="https://www.codecogs.com/eqnedit.php?latex=\hat&space;y^\iota_{t,\ell}=\sum_{j=0}^{D}\hat\phi^\iota_{\ell,j}\,\tau_t^j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat&space;y^\iota_{t,\ell}=\sum_{j=0}^{D}\hat\phi^\iota_{\ell,j}\,\tau_t^j" title="\hat y^\iota_{t,\ell}=\sum_{j=0}^{D}\hat\phi^\iota_{\ell,j}\,\tau_t^j" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\hat\sigma^\iota_\ell=\text{softplus}(\psi^\iota_\ell)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat\sigma^\iota_\ell=\text{softplus}(\psi^\iota_\ell)" title="\hat\sigma^\iota_\ell=\text{softplus}(\psi^\iota_\ell)" /></a> for industry.

**Currency conversion.** If the symbols passed to Volatile have different price currencies, we first find the most common currency and set it as default, then we download the last year of exchange rate information and convert all currencies to the default one. Training and score metric computation are executed using converted prices. Mathematically, if <a href="https://www.codecogs.com/eqnedit.php?latex=p_{t,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{t,i}" title="p_{t,i}" /></a> is the price of a certain stock in its currency, we define <a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{p}_{t,i}=r_{t,\text{curr}(i)}p_{t,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{p}_{t,i}=r_{t,\text{curr}(i)}p_{t,i}" title="\tilde{p}_{t,i}=r_{t,\text{curr}(i)}p_{t,i}" /></a> to be the converted price, where <a href="https://www.codecogs.com/eqnedit.php?latex=r_{t,\text{curr}(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_{t,\text{curr}(i)}" title="r_{t,\text{curr}(i)}" /></a> is the exchange rate from the original currency <a href="https://www.codecogs.com/eqnedit.php?latex=\text{curr}(i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{curr}(i)" title="\text{curr}(i)" /></a> to the default one. Then, the corresponding log-prices follow the relation <a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{y}_{t,i}=\log&space;r_{t,\text{curr}(i)}&plus;y_{t,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{y}_{t,i}=\log&space;r_{t,\text{curr}(i)}&plus;y_{t,i}" title="\tilde{y}_{t,i}=\log r_{t,\text{curr}(i)}+y_{t,i}" /></a>. Because we model <a href="https://www.codecogs.com/eqnedit.php?latex=y_{t,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{t,i}" title="y_{t,i}" /></a> as a Gaussian, <a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{y}_{t,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{y}_{t,i}" title="\tilde{y}_{t,i}" /></a> is also a Gaussian with the additional log-exchange rate in the mean and same standard deviation. Therefore, after mean and standard deviation estimates of <a href="https://www.codecogs.com/eqnedit.php?latex=\tilde{y}_{t,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tilde{y}_{t,i}" title="\tilde{y}_{t,i}" /></a> are computed, estimators for <a href="https://www.codecogs.com/eqnedit.php?latex=y_{t,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{t,i}" title="y_{t,i}" /></a> can be promptly obtained, from which log-Normal mean and standard deviation estimators of <a href="https://www.codecogs.com/eqnedit.php?latex=p_{t,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_{t,i}" title="p_{t,i}" /></a> can in turn be produced.

### Bots description
Meet the participants of the bot-tournament! Remember: you can run the tournament via `python tournament.py`.

<table>
<tr>
<td width=170>

<img src='https://avataaars.io/?avatarStyle=Circle&topType=ShortHairDreads02&accessoriesType=Blank&hairColor=Platinum&facialHairType=BeardMedium&facialHairColor=BlondeGolden&clotheType=BlazerSweater&eyeType=EyeRoll&eyebrowType=UpDownNatural&mouthType=Disbelief&skinColor=Brown' width="150"
/>

</td>
<td>
  
### Adam
Adam is a fairly cautious trader: he picks up very cheap stocks and tries to make a small profit out of them. He buys a stock only if it is rated as HIGHLY BELOW TREND, with a maximum transaction of 3.33% of his current capital. He sells a stock as soon as he makes a 3% profit or a 10% loss from it.
</td>
</tr>
</table>

<table>
<tr>
<td width=170>

<img src='https://avataaars.io/?avatarStyle=Circle&topType=Turban&accessoriesType=Kurt&hatColor=PastelRed&facialHairType=Blank&clotheType=ShirtScoopNeck&clotheColor=Blue02&eyeType=Close&eyebrowType=DefaultNatural&mouthType=Default&skinColor=DarkBrown' width="150" alt="avatar"
/>

</td>
<td>
  
### Betty
Betty is a risk-lover: she believes that what is going up will keep going up and jumps on it. She buys a stock only if it is rated as HIGHLY ABOVE TREND, with a maximum transaction of 3.33% of her current capital. She sells a stock as soon as she makes a 10% profit or a 3% loss from it.
</td>
</tr>
</table>

<table>
<tr>
<td width=170>

<img src='https://avataaars.io/?avatarStyle=Circle&topType=WinterHat2&accessoriesType=Wayfarers&hatColor=Blue03&facialHairType=Blank&clotheType=ShirtVNeck&clotheColor=Red&eyeType=Happy&eyebrowType=RaisedExcitedNatural&mouthType=ScreamOpen&skinColor=Yellow' width="150" alt="avatar"
/>

</td>
<td>
  
### Chris
Chris is a tech-lover: he will buy his favourite tech stocks as soon as possible, as much as possible, and hold them until retirement. They are AMZN, GOOGL, FB, AAPL, MSFT.
</td>
</tr>
</table>

<table>
<tr>
<td width=170>

<img src='https://avataaars.io/?avatarStyle=Circle&topType=LongHairFrida&accessoriesType=Blank&facialHairType=Blank&clotheType=Overall&clotheColor=Heather&eyeType=Squint&eyebrowType=RaisedExcited&mouthType=Smile&skinColor=Tanned'
/>

</td>
<td>
  
### Dany
Dany believes that if she waits long enough, cheap stocks will make a profit. She buys a stock only if it is rated as HIGHLY BELOW TREND or BELOW TREND, with a maximum transaction of 3.33% of her current capital. She sells a stock as soon as she makes a 10% profit or a 20% loss from it.
</td>
</tr>
</table>

<table>
<tr>
<td width=170>

<img src='https://avataaars.io/?avatarStyle=Circle&topType=LongHairCurly&accessoriesType=Blank&hairColor=Brown&facialHairType=BeardLight&facialHairColor=BrownDark&clotheType=ShirtScoopNeck&clotheColor=White&eyeType=Wink&eyebrowType=UnibrowNatural&mouthType=Smile&skinColor=Light'
/>

</td>
<td>
  
### Eddy
Eddy prefers stocks that are going fairly strong. He buys a stock only if it is rated as HIGHLY ABOVE TREND or ABOVE TREND, with a maximum transaction of 3.33% of his current capital. He sells a stock as soon as he makes a 20% profit or a 10% loss from it.
</td>
</tr>
</table>