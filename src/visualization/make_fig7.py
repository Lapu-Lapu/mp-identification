import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.models.train_model import (compute_complete_splits,
                                    sigmoid,
                                    transform_input,
                                    crossvalidate)
from src.models.globs import pp, beta_std
from src.visualization.make_fig5 import get_best_weights

df = pd.read_json('data/processed/preprocessed_data.json')

results = pd.read_pickle('models/logistic_regression_model.pkl')
raw_results = pd.read_pickle('models/logistic_regression_model_raw.pkl')
score = 'MSE'


# Plotting
fig, ax = plt.subplots()
fig.set_size_inches(8, 5)

# std = groupby('model_id').y.apply(beta_std)
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

best_weights = get_best_weights(raw_results, ['all', score])
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

tstr = 'All models'
ax.set_title(tstr)
ax.set_ylabel('Confusion rate')
ax.set_xlabel('MSE')
ax.ticklabel_format(style='sci', scilimits=(-2, 4), axis='both')
ax.grid()

plt.savefig('reports/figures/fig7.pdf')
