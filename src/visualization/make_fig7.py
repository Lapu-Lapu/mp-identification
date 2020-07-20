import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.train_model import (compute_complete_splits,
                                    sigmoid,
                                    transform_input,
                                    crossvalidate)
from src.models.globs import pp
from src.visualization.make_fig5 import get_best_weights

df = pd.read_json('data/processed/preprocessed_data.json')
score = 'MSE'
score_min = df[score].values.min()
score_max = df[score].values.max()

X = transform_input(df, score=score)

X[:, 1] = 2*((X[:, 1]-score_min)/(score_max-score_min)-0.5)
print(len(X), 'Datapoints')

# # Cross validations
# choose number of blocks
possible_n = compute_complete_splits(len(X))
n_blocks = possible_n[3]
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
results = {}
for i in range(len(fit_costs)):
    res = {'residual_cost': fit_costs[i],
           'test_cost': test_costs[i],
           'residual_cost_const': fit_cost_const[i],
           'test_cost_const': test_costs_const[i],
           'mu': mus[i],
           'N': len(X)}
    for j in range(len(ws[i])):
        res[f'w{j}'] = ws[i][j]
    results[(score, i)] = res

raw_results = pd.DataFrame(results).T
raw_results.index = raw_results.index.set_names(['score', 'n'])

# aggregate over crossvalidations
group = raw_results.groupby(level=['score'])
results = group.mean()
N_crossval = group.apply(len)
N_crossval.name = 'blocks'
results = pd.concat((results, N_crossval), axis=1)
results = results.assign(llr_vconst=lambda df:
                         df.N*(-df.test_cost+df.test_cost_const))

# Plotting
fig, ax = plt.subplots()
fig.set_size_inches(8, 5)

marker = ['C0x', 'C1x', 'C2x', 'C3x', 'C4o', 'C5o']
models = ['vcgpdm', 'vgpdm', 'tmp', 'dmp', 'mapcgpdm', 'mapgpdm']
for model, m in zip(models, marker):
    df_model = df[df.mp_type == model]
    (_, caps, _) = ax.errorbar(
        x=df_model[score], y=df_model.confusion_rate,
        yerr=df_model['std'],
        ecolor='lightgray',
        fmt=m,
        markersize=4, elinewidth=0.25, capsize=2.5,
        mew=0.5,
        label=pp[model]
    )

best_weights = get_best_weights(raw_results, score)
x = np.linspace(df[score].min(), df[score].max())
t = ((x - x.min())/(x.max()-x.min())-0.5)*2
t = t[:, np.newaxis]
t = np.concatenate((np.ones_like(t), t), axis=1)
w = best_weights.values[:2, np.newaxis]
c = best_weights.values[2]
y = sigmoid(np.dot(t, w), c=c)
ax.plot(x, y, label='predicted')

ax.legend()

# w2 is sigmoid upper limit!
best_weights = best_weights[['w0', 'w1', 'w2']]

logk = results.xs(score)['llr_vconst']

tstr = 'All models'
ax.set_title(tstr)
ax.set_ylabel('Confusion rate')
ax.set_xlabel('MSE')
ax.ticklabel_format(style='sci', scilimits=(-2, 4), axis='both')
ax.grid()

plt.savefig('reports/figures/fig7.pdf')
