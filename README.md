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
Volatiles estimates stock trends, predict short-term future prices, then ranks and rates. For example, suppose that you want Volatile to analyze the following stock symbols: AMZN, GOOGL, JPM, DAL, DUK, KO, LULU, AMD, and UNH. If you use either Mac or Linux, you can just type in your terminal
```ruby
./volatile.py -s AMZN GOOGL JPM DAL DUK KO LULU AMD UNH
```
When the run is complete, a prediction table like the following will appear printed on your shell:

<img width="792" alt="Screenshot 2020-11-28 at 11 38 20" src="https://user-images.githubusercontent.com/32386694/100509519-4cd31e80-316e-11eb-9316-eba4bdbec43b.png">

For each symbol, the table tells you the last available observed price, the predicted closing price at the next trading day, its standard deviation and finally a rating. Possible ratings are STRONG BUY, BUY, NEUTRAL, SELL and STRONG SELL. Symbols appears in the table ranked from most to least potentially profitable. Ranking and rating are derived from a score metric that compares the predicted closing price in 5 trading days (usually this corresponds to the price in one week) with the last available observed price, normalizing by the standard deviation of the prediction; see the technical part below for more details. A .csv file with the data contained in the prediction table can be saved in the current directory by adding the following flag to the command above: `-spt=True`.

In the current directory, two estimation plots will appear. One is a visualization of stock prices and their estimations over the last year, together with a notion of uncertainty. Notice how the estimation crucially attempts to reproduce the trend of a stock but not to learn its noise. The uncertainty, on the other hand, depends on the stock volatily; the smaller the volatility, the more confident we are about our estimates, the more a suddend shift from the trend will be regarded as an opportunity to buy or sell. You can use this plot as a sanity check that the estimation procedure is doing a sensible job. Like in the prediction table, the first and the last stocks respectively represent the highest estimated opportunities to buy and sell. Make sure to glance at them before any transaction.
![stock_estimation_plots](https://user-images.githubusercontent.com/32386694/100509895-68d6c000-316e-11eb-8be4-75b2f117d723.png)

A second plot estimates the sector of the stocks the you have provided. A sector estimate is the average of the belonging stocks. If, like in this case, you only provide a few symbols to Volatile, you cannot expect sector information to be accurate. However, when a larger amount of symbols is provided, this plot is a good way to quickly check how the sector of the stock that you intend to buy or sell is doing.
![sector_estimation_plots](https://user-images.githubusercontent.com/32386694/100510060-74c28200-316e-11eb-9fbb-ed39d107df24.png)
If you do not want plots to be saved in the current directory, you can disable them by adding the flag `-pe=False`.

Please find a larger list of stock symbols in `symbols_list.rtf`, that you can easily access, copy and then paste in the command above. Under no circumstance this should be considered as a privileged list of stocks. Feel free to update this list as you please (do not worry if by chance you enter a symbol twice). Mind that it can take a while to access sector information of symbols that were not in the list and that you pass to Volatile for the first time. Because of that, their sector name gets stored in `stock_info.csv` and will be fast to access from the second time onwards.

### How to install
Installing Volatile follows a standard procedure. First, open a terminal and go to the directory where you intend to install Volatile. On a Mac or Linux, you can do so by typing
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
pip install --requirement requirements.txt
```
Done! You're all set to use Volatile. 

### Behind the scenes (technical)
Volatile adopts a Bayesian hierarchical model based on price and sector information, estimating log-price time series via polynomials in time. 

Denote <a href="https://www.codecogs.com/eqnedit.php?latex=\tau_t=t/T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau_t=t/T" title="\tau_t=t/T" /></a> to represent times at which observations arrive. <a href="https://www.codecogs.com/eqnedit.php?latex=T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T" title="T" /></a> corresponds to the number of days in the training dataset, which is taken to be the last one year of data.

Furthermore, denote <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma_j=(D&plus;1-j)/(D&plus;1),\text{&space;for&space;}j=0,\dots,D" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma_j=(D&plus;1-j)/(D&plus;1),\text{&space;for&space;}j=0,\dots,D" title="\sigma_j=(D+1-j)/(D+1),\text{ for }j=0,\dots,D" /></a> to be prior scale parameters associated to the j-th order of a polynomial with degree <a href="https://www.codecogs.com/eqnedit.php?latex=D" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D" title="D" /></a>. Decreasing the scales as <a href="https://www.codecogs.com/eqnedit.php?latex=j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?j" title="j" /></a> increases penalizes deviation from zero of higher-order parameters, thereby encouraging simpler models. We set <a href="https://www.codecogs.com/eqnedit.php?latex=D=7" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D=7" title="D=7" /></a>, that, given the model described below and the amount of data taken for training, we found to be a good complexity to capture trends without overfitting data.

In addition, we write <a href="https://www.codecogs.com/eqnedit.php?latex=\text{sec(i)}=k,\text{&space;for&space;}i=1,\dots,&space;N\text{&space;and&space;}k=1\dots,K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{sec(i)}=k,\text{&space;for&space;}i=1,\dots,&space;N\text{&space;and&space;}k=1\dots,K" title="\text{sec(i)}=k,\text{ for }i=1,\dots, N\text{ and }k=1\dots,K" /></a> to indicate that a stock <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a> belongs to a sector <a href="https://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k" title="k" /></a>, where <a href="https://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N" title="N" /></a> is the number of stocks and <a href="https://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="https://latex.codecogs.com/gif.latex?K" title="K" /></a> the number of sectors. Then, we construct the hierarchical model

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\begin{align*}&space;\phi^s_{k,j}&space;&\sim&space;\mathcal{N}(0,\&space;\sigma_j^2)\\&space;\phi_{i,j}&space;&\sim&space;\mathcal{N}(\phi^s_{\text{sec}(i),j},\&space;\tfrac{1}{4}\sigma_j^2)\\&space;\psi^s_{k}&space;&\sim&space;\mathcal{N}(0,\&space;1)\\&space;\psi_{i}&space;&\sim&space;\mathcal{N}(\psi^s_{\text{sec}(i),j},\&space;1)\\&space;y_{t,i}&space;&\sim&space;\mathcal{N}\left(\sum_{j=0}^{D}\phi_{i,j}\,\tau_t^j,&space;\&space;\log^2(1&space;&plus;&space;e^{\psi_i&space;&plus;&space;|1&space;-&space;\tau_t|})\right)&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\begin{align*}&space;\phi^s_{k,j}&space;&\sim&space;\mathcal{N}(0,\&space;\sigma_j^2)\\&space;\phi_{i,j}&space;&\sim&space;\mathcal{N}(\phi^s_{\text{sec}(i)},\&space;\tfrac{1}{4}\sigma_j^2)\\&space;\psi^s_{k}&space;&\sim&space;\mathcal{N}(0,\&space;1)\\&space;\psi_{i}&space;&\sim&space;\mathcal{N}(\psi^s_{\text{sec}(i)},\&space;1)\\&space;y_{t,i}&space;&\sim&space;\mathcal{N}\left(\sum_{j=0}^{D}\phi_{i,j}\,\tau_t^j,&space;\&space;\log^2(1&space;&plus;&space;e^{\psi_i&space;&plus;&space;|1&space;-&space;\tau_t|})\right)&space;\end{align*}" title="\begin{align*} \phi^s_{k,j} &\sim \mathcal{N}(0,\ \sigma_j^2)\\ \phi_{i,j} &\sim \mathcal{N}(\phi^s_{\text{sec}(i)},\ \tfrac{1}{4}\sigma_j^2)\\ \psi^s_{k} &\sim \mathcal{N}(0,\ 1)\\ \psi_{i} &\sim \mathcal{N}(\psi^s_{\text{sec}(i),j},\ 1)\\ y_{t,i} &\sim \mathcal{N}\left(\sum_{j=0}^{D}\phi_{i,j}\,\tau_t^j, \ \log^2(1 + e^{\psi_i + |1 - \tau_t|})\right) \end{align*}" /></a>

Parameters at sector-level <a href="https://www.codecogs.com/eqnedit.php?latex=\phi^s\text{&space;and&space;}\psi^s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi^s\text{&space;and&space;}\psi^s" title="\phi^s\text{ and }\psi^s" /></a> are respectively the prior means of the corresponding stock-level parameters <a href="https://www.codecogs.com/eqnedit.php?latex=\phi\text{&space;and&space;}\psi." target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi\text{&space;and&space;}\psi." title="\phi\text{ and }\psi." /></a>

The sector-level parameters are supposed to be independent over their components; the stock-level parameters are supposed to be conditionally independent over their components given the sector-level parameters. Whereas <a href="https://www.codecogs.com/eqnedit.php?latex=\phi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\phi" title="\phi" /></a> are used to determine the coefficients of the polynomial model,  <a href="https://www.codecogs.com/eqnedit.php?latex=\psi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\psi" title="\phi" /></a> are used to determine the scales of the likelihood function. The likelihood, defined in the last line of the hierarchical model, is a Gaussian centered at the polynomial model, with scales that become larger and larger the further the time index <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t" title="t" /></a> gets from the current time <a href="https://www.codecogs.com/eqnedit.php?latex=T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T" title="T" /></a>. In other words, recent data are weighted more the older ones, which get less and less importance the older they get.

In order to estimate the parameters, we condition on the log-prices <a href="https://www.codecogs.com/eqnedit.php?latex=y_{t,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{t,i}" title="y_{t,i}" /></a>, for all <a href="https://www.codecogs.com/eqnedit.php?latex=t=1,\dots&space;T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t=1,\dots&space;T" title="t=1,\dots T" /></a>, then we estimate the mode of the posterior distribution, also known as Maximum-A-Posteriori (MAP). From a frequentist statistics perspective, this corresponds to a polynomial regression task where we minimize a regularized and weighted mean-squared error loss.

Obtained our estimates <a href="https://www.codecogs.com/eqnedit.php?latex=\hat\phi^s,\&space;\hat\phi,\&space;\hat\psi^s\text{&space;and&space;}&space;\hat\psi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat\phi^s,\&space;\hat\phi,\&space;\hat\psi^s\text{&space;and&space;}&space;\hat\psi" title="\hat\phi^s,\ \hat\phi,\ \hat\psi^s\text{ and } \hat\psi" /></a>, we can use the likelihood mean <a href="https://www.codecogs.com/eqnedit.php?latex=\hat&space;y_{t,i}=\sum_{j=0}^{D}\hat\phi_{i,j}\,\tau_t^j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat&space;y_{t,i}=\sum_{j=0}^{D}\hat\phi_{i,j}\,\tau_t^j" title="\hat y_{t,i}=\sum_{j=0}^{D}\hat\phi_{i,j}\,\tau_t^j" /></a> as an estimate of the data for any time in the past, as well as a predictor for times in the short future. As a measure of uncertainty, we take the learned scale of the likelihood, that is <a href="https://www.codecogs.com/eqnedit.php?latex=\hat\sigma_{t,i}=\log(1&plus;e^{\hat\psi_i&plus;|1-\tau_t|})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat\sigma_{t,i}=\log(1&plus;e^{\hat\psi_i&plus;|1-\tau_t|})" title="\hat\sigma_{t,i}=\log(1+e^{\hat\psi_i+|1-\tau_t|})" /></a>.

Given these estimates, Volatile provides a rating for each stock by introducing the following metric:

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{score}_i=\frac{\hat&space;y_{T&plus;5,i}-y_{T,i}}{\hat\sigma_{T&plus;5,i}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{score}_i=\frac{\hat&space;y_{T&plus;5,i}-y_{T,i}}{\hat\sigma_{T&plus;5,i}}" title="\text{score}_i=\frac{\hat y_{T+5,i}-y_{T,i}}{\hat\sigma_{T+5,i}}" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;y_{T,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;y_{T,i}" title="y_{T,i}" /></a> is the last available log-price, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\hat&space;y_{T&plus;5,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat&space;y_{T&plus;5,i}" title="\hat y_{T+5,i}" /></a> is its prediction in 5 trading days (usually, that corresponds to the log-price in one week) and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\hat\sigma_{T&plus;5,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\hat\sigma_{T&plus;5,i}" title="\hat\sigma_{T+5,i}" /></a> is the estimated standard deviation of the prediction. If the future prediction is larger than the current price, the score will be positive; the larger the difference and the more confident we are about the prediction (or equivalently, the smaller the standard deviation is), the more positive will be the score. We can reason similarly if the score is negative. In other words, a large positive score indicates that the current price is undervalued with respect to its stock trend, therefore an opportunity to buy; a large negative score indicates, vice versa, that the current price is overvalued with respect to its stock trend, therefore a moment to sell. 

Then, stocks are rated according to the following criteria:
- STRONG BUY if <a href="https://www.codecogs.com/eqnedit.php?latex=\text{score}_i>3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{score}_i>3" title="\text{score}_i>3" /></a>; 
- BUY if <a href="https://www.codecogs.com/eqnedit.php?latex=2<\text{score}_i<=3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2<\text{score}_i<=3" title="2<\text{score}_i<=3" /></a>;
- NEUTRAL if <a href="https://www.codecogs.com/eqnedit.php?latex=-2<\text{score}_i<=2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?-2<\text{score}_i<=2" title="-2<\text{score}_i<=2" /></a>;
- SELL if <a href="https://www.codecogs.com/eqnedit.php?latex=-3<\text{score}_i<=-2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?-3<\text{score}_i<=-2" title="-3<\text{score}_i<=-2" /></a>;
- STRONG SELL if <a href="https://www.codecogs.com/eqnedit.php?latex=\text{score}_i<=-3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{score}_i<=-3" title="\text{score}_i<=-3" /></a>.

Because we model log-prices as a Gaussian, the distribution of prices is a log-Normal distribution, whose mean and standard deviation can be derived in closed form from the estimators <a href="https://www.codecogs.com/eqnedit.php?latex=\hat&space;y_{t,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat&space;y_{t,i}" title="\hat y_{t,i}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\hat\sigma_{t,i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat\sigma_{t,i}" title="\hat\sigma_{t,i}" /></a>. We use log-Normal distribution statistics at times <a href="https://www.codecogs.com/eqnedit.php?latex=t=1\dots,T" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t=1\dots,T" title="t=1\dots,T" /></a> to produce the stock estimation plots and at time <a href="https://www.codecogs.com/eqnedit.php?latex=T&plus;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?T&plus;1" title="T+1" /></a> to fill the prediction table. In order to produce sector estimation plots, we procede analogously but with sector-level estimators <a href="https://www.codecogs.com/eqnedit.php?latex=\hat&space;y^s_{t,k}=\sum_{j=0}^{D}\hat\phi^s_{k,j}\,\tau_t^j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat&space;y^s_{t,k}=\sum_{j=0}^{D}\hat\phi^s_{k,j}\,\tau_t^j" title="\hat y^s_{t,k}=\sum_{j=0}^{D}\hat\phi^s_{k,j}\,\tau_t^j" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\hat\sigma^s_{t,k}=\log(1&plus;e^{\hat\psi^s_k&plus;|1-\tau_t|})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat\sigma^s_{t,k}=\log(1&plus;e^{\hat\psi^s_k&plus;|1-\tau_t|})" title="\hat\sigma^s_{t,k}=\log(1+e^{\hat\psi^s_k+|1-\tau_t|})" />.
