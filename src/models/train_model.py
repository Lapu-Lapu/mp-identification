from src.models.globs import model
import pandas as pd
# import numpy as np
import autograd.numpy as np
from autograd import grad
from functools import partial, update_wrapper
from scipy.optimize import minimize


def sigmoid(x, c=0.5, a=0.2, mu=0):
    y = c / (1 + np.exp(-(x - mu) / a))
    return y


def nlogprobability(params, X):
    """
    negative log evidence:
    X: Data
    cost function for linear regression
    theta: [const, w1, w2, upper_limit, scale]
    """
    x = X[:, :-1]  # inputs, x[0] ist immer 1 (Konstante)
    y = X[:, -1]  # output
    m = y.size
    w = np.array(params[:2])  # Gewichte
    c = params[2] if len(params) > 2 else 0.5
    if len(params) > 3:
        a = params[3]
    else:
        a = 0.2
    h = sigmoid(np.dot(x, w), c=c, a=a)
    J = 1. / m * (-np.dot(y, np.log(h)) - np.dot((1 - y), np.log(1 - h)))
    return J


def nlogprobability_const(w, X):
    x = X[:, 0]
    y = X[:, -1]
    p = w * x
    m = y.size
    J = 1 / m * (-np.dot(y, np.log(p)) - np.dot((1 - y), np.log(1 - p)))
    return J


def fit(X, w_init=(0.0, 0.5, 0.1), const=False):
    if not const:
        cost = partial(nlogprobability, X=X)
        update_wrapper(cost, nlogprobability)
        gradient = grad(cost)
        jac, method, callback = gradient, 'L-BFGS-B', None
        res = minimize(cost, w_init, jac=jac, method=method, callback=callback,
                       bounds=[(None, None), (None, None), (0.1, 0.5)])
    else:
        mu = np.sum(X[:, -1]) / len(X)
        cost = nlogprobability_const(mu, X)
        res = {'fun': cost, 'x': mu}
    return res


def crossvalidate(X, block_n, const=False):
    N, D = X.shape
    X = X[np.random.permutation(range(N))].copy()
    block_len = N // block_n
    T = [X[i*block_len:(i+1)*block_len] for i in range(block_n)]
    fit_res, test_costs = [], []
    for i, test in enumerate(T):
        idx = list(range(block_n))
        idx.remove(i)
        train = np.concatenate([T[j] for j in idx])
        res = fit(X=train, const=const)
        fit_res.append(res)
        if const:
            mu = np.sum(train[:, -1]) / len(train)
            testcost = nlogprobability_const(mu, X=test)
        else:
            testcost = nlogprobability(params=res['x'], X=test)
        test_costs.append(testcost)
    return fit_res, test_costs


def compute_complete_splits(N):
    pn = []
    while len(pn) == 0:
        pn = [i for i in range(2, N) if (N/i) % 1 == 0]
        N = N - 1
    return pn


def transform_input(D, score='ELBO'):
    pre = pd.Series(np.ones((len(D),)), name='const',
                    index=D.index)
    D = pd.concat([pre, D], axis=1)
    D['score'] = D[score]
    D = np.array(D[['const', 'score', 'y']])
    return D


if __name__ == '__main__':
    Df = pd.read_json('data/processed/preprocessed_data.json')

    results = {}
    mp_types = ['all', 'tmp', 'dmp', 'vgpdm', 'vcgpdm']
    model['all'] = {}
    model['all']['scores'] = ['MSE']
    for mp_type in mp_types:
        mp = model[mp_type]
        df = Df[Df.mp_type == mp_type] if mp_type != 'all' else Df

        for score in mp['scores']:
            print('++++', mp_type, score, ':', end=' ')
            score_min = df[score].values.min()
            score_max = df[score].values.max()

            X = transform_input(df, score=score)

            X[:, 1] = 2*((X[:, 1]-score_min)/(score_max-score_min)-0.5)
            print(len(X), 'Datapoints ++++')

            # # Cross validations
            # choose number of blocks
            possible_n = compute_complete_splits(len(X))
            try:
                n_blocks = possible_n[3]
            except IndexError:
                print('possible n for crossvalidation:', possible_n)
                n_blocks = possible_n[0]
            # compute cv-scores for constant prediction
            fit_res_const, test_costs_const = crossvalidate(X, n_blocks,
                                                            const=True)

            fit_cost_const = [f['fun'] for f in fit_res_const]
            mus = [f['x'] for f in fit_res_const]
            # compute cv-scores with score-regressor
            fit_res, test_costs = crossvalidate(X, n_blocks)
            fit_costs = [f.fun for f in fit_res]
            ws = [f.x for f in fit_res]

            # Save result
            for i in range(len(fit_costs)):
                res = {'residual_cost': fit_costs[i],
                        'test_cost': test_costs[i],
                        'residual_cost_const': fit_cost_const[i],
                        'test_cost_const': test_costs_const[i],
                        'mu': mus[i],
                        'N': len(X)}
                for j in range(len(ws[i])):
                    res[f'w{j}'] = ws[i][j]
                results[(mp_type, score, i)] = res


    raw_results = pd.DataFrame(results).T
    raw_results.index = raw_results.index.set_names(['mp_type', 'score', 'n'])
    raw_results.to_pickle('models/logistic_regression_model_raw.pkl')

    # aggregate over crossvalidations
    group = raw_results.groupby(level=['mp_type', 'score'])
    results = group.mean()
    N_crossval = group.apply(len)
    N_crossval.name = 'blocks'
    results = pd.concat((results, N_crossval), axis=1)

    # compute logk = N_{Trials}*(llh_{test}(w^\star)-llh_{test}(w_0=mean))
    results = results.assign(llr_vconst=lambda df:
                             df.N*(-df.test_cost+df.test_cost_const))

    results.to_pickle('models/logistic_regression_model.pkl')
