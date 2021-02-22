#!/usr/bin/env python
from volatile import *
from models import *

from copy import deepcopy

def second_moment_loss(tt: np.array, mu: np.array, sigma: np.array, logp: np.array) -> float:
    """
    Because log-price likelihood is Gaussian, standardizing the test data should give a second moment (i.e. squared
    expectation plus variance) of 1.

    Parameters
    ----------
    tt: np.array
        Array of times.
    mu: np.array
        Mean parameters.
    sigma: np.array
        Standard deviation parameters.
    logp: np.array
        Log-prices at stock-level.
    """
    est, std = estimate_logprice_statistics(mu, sigma, tt)
    scores = (est - logp) / std
    return np.abs(np.mean(scores ** 2) - 1)

def negative_loglikelihood(model: tfd.JointDistributionSequentialAutoBatched, params: tuple, logp: np.array) -> float:
    """
    It returns the negative log-likelihood of the model.

    Parameters
    ----------
    model: tfd.JointDistributionSequentialAutoBatched
        Probabilistic model.
    params: tuple
        Parameters of the model.
    logp: np.array
        Log-prices at stock-level.
    """
    return -model.log_prob_parts(list(params) + [logp])[-1].numpy()

if __name__ == '__main__':
    cli = ArgumentParser('Volatile: your day-to-day trading companion.',
                         formatter_class=ArgumentDefaultsHelpFormatter)
    cli.add_argument('-s', '--symbols', type=str, nargs='+', help=SUPPRESS)
    cli.add_argument('--days', type=int, default=5,
                     help="This refers to the number of last days to hold out for validation.")
    args = cli.parse_args()

    print('\nDownloading all available closing prices in the last year...')
    if args.symbols is None:
        with open("symbols_list.txt", "r") as my_file:
            args.symbols = my_file.readlines()[0].split(" ")
    data = download(args.symbols)
    tickers = data["tickers"]
    logp = np.log(data['price'])

    # convert currencies to most frequent one
    for i, curr in enumerate(data['currencies']):
        if curr != data['default_currency']:
            logp[i] = convert_currency(logp[i], np.array(data['exchange_rates'][curr]), type='forward')

    num_stocks = logp.shape[0]
    t = logp[:, :-args.days].shape[1]

    ## independent stock model
    print("\ns_2 model evaluation...")
    order = 2
    info = dict()
    info['tt'] = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
    info['order_scale'] = np.linspace(1 / (order + 1), 1, order + 1)[::-1].astype('float32')[None, :]
    info['num_stocks'] = num_stocks
    tt_pred = ((1 + (np.arange(1, 1 + args.days) / t)) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')

    params = train_s(logp[:, :-args.days], info)

    phi, psi = params
    sm_loss = second_moment_loss(tt_pred, phi.numpy(), psi.numpy(), logp[:, -args.days:])

    info_pred = deepcopy(info)
    info_pred['tt'] = tt_pred
    pred_model = s_model(info_pred)
    nllkd = negative_loglikelihood(pred_model, params, logp[:, -args.days:])

    print('Second moment loss: ', sm_loss)
    print('Negative log-likelihood: ', nllkd)

    ## Model without clusters
    print("\nmsis_2 model evaluation...")
    order = 2
    info = extract_hierarchical_info(data['sectors'], data['industries'])
    info['tt'] = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
    info['order_scale'] = np.linspace(1 / (order + 1), 1, order + 1)[::-1].astype('float32')[None, :]
    tt_pred = ((1 + (np.arange(1, 1 + args.days) / t)) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')

    params = train(logp[:, :-args.days], info)

    phi, psi = params[-2:]
    sm_loss = second_moment_loss(tt_pred, phi.numpy(), psi.numpy(), logp[:, -args.days:])

    info_pred = deepcopy(info)
    info_pred['tt'] = tt_pred
    pred_model = define_model(info_pred)
    nllkd = negative_loglikelihood(pred_model, params, logp[:, -args.days:])

    print('Second moment loss: ', sm_loss)
    print('Negative log-likelihood: ', nllkd)

    ## Model wit clusters
    print("\nmsis_mcs_2 model evaluation...")
    ## Estimate clusters
    order = 51

    info = extract_hierarchical_info(data['sectors'], data['industries'])
    info['tt'] = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
    info['order_scale'] = np.linspace(1 / (order + 1), 1, order + 1)[::-1].astype('float32')[None, :]

    phi_m, psi_m, phi_s, psi_s, phi_i, psi_i, phi, psi = train(logp[:, :-args.days], info, num_steps=50000)
    clusters_id = estimate_clusters(tickers, phi.numpy(), info['tt'])

    order = 2
    info = extract_hierarchical_info(data['sectors'], data['industries'])
    info['tt'] = (np.linspace(1 / t, 1, t) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')
    info['order_scale'] = np.linspace(1 / (order + 1), 1, order + 1)[::-1].astype('float32')[None, :]
    info['clusters_id'] = clusters_id
    tt_pred = ((1 + (np.arange(1, 1 + args.days) / t)) ** np.arange(order + 1).reshape(-1, 1)).astype('float32')

    params = train(logp[:, :-args.days], info)
    phi, psi, chi_m, xi_m, chi_c, xi_c, chi, xi = params[6:]
    sm_loss = second_moment_loss(tt_pred, (phi + chi).numpy(), (psi + xi).numpy(), logp[:, -args.days:])

    info_pred = deepcopy(info)
    info_pred['tt'] = tt_pred
    pred_model = define_model(info_pred)
    nllkd = negative_loglikelihood(pred_model, params, logp[:, -args.days:])

    print('Second moment loss: ', sm_loss)
    print('Negative log-likelihood: ', nllkd)