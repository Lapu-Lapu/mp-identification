from src.models.globs import model, pp, beta_std
from src.models.train_model import sigmoid
from src.visualization.make_fig5 import plot_psychometric, get_best_weights
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_pickle('models/logistic_regression_model.pkl')
raw_results = pd.read_pickle('models/logistic_regression_model_raw.pkl')
Df = pd.read_json('data/processed/preprocessed_data.json')
scores = pd.read_json('data/processed/joint_scores.json').set_index('model_id')

m = {
    ('vcgpdm', 'MSE'): (0, 0),
    ('vcgpdm', 'ELBO'): (1, 0),
    ('vcgpdm', 'dyn_elbo'): (2, 0),
    ('vcgpdm', 'lvm_elbo'): (3, 0),
    ('vgpdm', 'MSE'): (0, 1),
    ('vgpdm', 'ELBO'): (1, 1),
    ('vgpdm', 'dyn_elbo'): (2, 1),
    ('vgpdm', 'lvm_elbo'): (3, 1),
}

fig, ax = plt.subplots(ncols=2, nrows=4, sharey='row')
fig.set_size_inches(8, 10)

logk_min = results['llr_vconst'].min()
logk_max = results['llr_vconst'].max()

mp_types = ['vcgpdm', 'vgpdm']
for mp_type in mp_types:
    mp = model[mp_type]
    df_scores = scores[scores.mp_type == mp_type]
    df = Df[Df.mp_type == mp_type]

    for score in mp['scores']:
        best_weights = get_best_weights(raw_results, (mp_type, score))

        # w2 is sigmoid upper limit!
        best_weights = best_weights[['w0', 'w1', 'w2']]

        df_s = df_scores.loc[:, mp['params'] + [score, 'confusion_rate']]
        logk = results.xs((mp_type, score))['llr_vconst']
        c_val = (logk - logk_min + 0.2) / (logk_max - logk_min + 0.2)
        txt_color = (1 - c_val, c_val, 0)

        plot_psychometric(mp, df, df_s, best_weights, ax[m[(mp_type,
                                                           score)]])

        tstr = pp[mp_type] + '\n'
        if m[(mp_type, score)][0] == 0:
            ax[m[(mp_type, score)]].set_title(tstr)
        if score == 'MSE':
            ax[m[(mp_type, score)]].set_xlim(0, 0.035)
        if score == 'ELBO':
            ax[m[(mp_type, score)]].set_xlim(79500, 120000)
        if score == 'dyn_elbo':
            ax[m[(mp_type, score)]].set_xlim(-12000, -900)
        if score == 'lvm_elbo':
            ax[m[(mp_type, score)]].set_xlim(95000, 125000)
        ax[m[(mp_type, score)]].ticklabel_format(
            style='sci', scilimits=(-2, 4), axis='both')
        ax[m[(mp_type, score)]].grid()
        ax[m[(mp_type, score)]].annotate(
            f'ln K: {logk:.1f}',
            xy=(0.01, 1.05),
            xycoords='axes fraction',
            size=12,
            color=txt_color)

        ax[m[(mp_type, score)]].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax[m[(mp_type, score)]].get_legend().remove()
ax[0, 0].legend(['Exp. 1', 'Exp. 2', 'predicted'])

plt.tight_layout()
plt.savefig('reports/figures/fig6.pdf')
