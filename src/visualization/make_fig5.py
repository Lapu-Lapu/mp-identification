from src.models.globs import model, pp, beta_std
from src.models.train_model import sigmoid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_confusion_score(D):
    g = D.groupby('model_id')

    conf = g.y.mean()
    conf.name = 'confusion_rate'

    std = g.y.apply(beta_std)
    std.name = 'std'
    return conf, std


def compute_confusion(D):
    """
    Adds confusion rate grouped by mp_type,
    to dataframe D.
    Additionally adds beta-standard deviation,
    and number of models of mp_type.
    """
    conf, std = compute_confusion_score(D)
    conf = D.apply(lambda df: conf[df.model_id], axis=1)
    std = D.apply(lambda df: std[df.model_id], axis=1)
    conf.name = 'confusion_rate'
    std.name = 'std'

    return pd.concat([D, conf, std], axis=1)


def _predict(mp, x, df_scores, weights):
    score = list(set(df_scores.columns).intersection(set(mp['scores'])))[0]
    params = df_scores.loc[:, mp['params']].sort_values(mp['params'])

    x = ((x - x.min())/(x.max()-x.min())-0.5)*2
    x = x[:, np.newaxis]
    t = np.concatenate((np.ones_like(x), x), axis=1)
    w = weights.values[:2, None]
    c = weights.values[2]
    y = sigmoid(np.dot(t, w), c=c)
    return params, y


def plot_psychometric(mp, df, df_scores, weights, ax=None):
    """
    Args:
      df: data of trials
      df_scores: scores of models
      weights: weights for prediction
    """
    scoreset = set(df_scores.columns).intersection(set(mp['scores']))
    score = list(scoreset-{'confusion_rate'})[0]

    x = np.linspace(df_scores[score].min(), df_scores[score].max())
    params, y = _predict(mp, x, df_scores, weights)

    cr = {k: compute_confusion(df[df.expName == k])
          for k in ['main', 'follow-up']}

    if ax is None:
        fig, ax = plt.subplots()
    ax = cr['main'].plot(x=score, y='confusion_rate', style='bx',
                         alpha=0.5, rot=45, lw=1, ax=ax,
                         rasterized=True, ms=4)
    cr['follow-up'].plot(x=score, y='confusion_rate', style='rx',
                         alpha=0.5, ax=ax, lw=1, ms=4,
                         rasterized=True)
    ax.plot(x, y, 'g-', rasterized=True)
    ax.set_ylabel('Confusion rate')
    ax.set_xlabel(pp[score])
    return ax


def get_best_weights(data, idx_labels):
    d = data.xs(idx_labels)
    bidx = d.test_cost.idxmin()
    return d.loc[bidx, ['w0', 'w1', 'w2', 'test_cost']]


results = pd.read_pickle('models/logistic_regression_model.pkl')
# raw_results = pd.read_json('models/logistic_regression_model_raw.json')
raw_results = pd.read_pickle('models/logistic_regression_model_raw.pkl')
Df = pd.read_json('data/processed/preprocessed_data.json')
scores = pd.read_json('data/processed/joint_scores.json').set_index('model_id')

m = {
     ('tmp', 'MSE'):  (0, 0),
     ('tmp', 'ELBO'): (1, 0),
     ('dmp', 'MSE'):  (0, 1)
}

fig, ax = plt.subplots(ncols=2, nrows=2, sharey='row')
fig.set_size_inches(8, 5)

logk_min = results['llr_vconst'].min()
logk_max = results['llr_vconst'].max()

mp_types = ['tmp', 'dmp']
for mp_type in mp_types:
    mp = model[mp_type]
    df_scores = scores[scores.mp_type == mp_type]
    df = Df[Df.mp_type == mp_type]

    for score in mp['scores']:
        # # Test weights TODO: Check selection of weights
        best_weights = get_best_weights(raw_results, (mp_type, score))
        best_weights = best_weights[['w0', 'w1', 'w2']]

        df_s = df_scores.loc[:, mp['params']+[score, 'confusion_rate']]
        logk = results.xs((mp_type, score))['llr_vconst']
        c_val = (logk-logk_min+0.2)/(logk_max-logk_min+0.2)
        txt_color = (1-c_val, c_val, 0)

        plot_psychometric(mp, df, df_s, best_weights,
                             ax[m[(mp_type, score)]])
        # tstr = f'{mp_type}: perceived naturalness and prediction from {score}'
        tstr = pp[mp_type] + '\n'
        if m[(mp_type, score)][0] == 0:
            ax[m[(mp_type, score)]].set_title(tstr)
        if score == 'MSE':
            ax[m[(mp_type, score)]].set_xlim(0.001, 0.007)
        ax[m[(mp_type, score)]].ticklabel_format(style='sci', scilimits=(-2,4), axis='both')
        ax[m[(mp_type, score)]].grid()
        ax[m[(mp_type, score)]].annotate(f'ln K: {logk:.1f}', xy=(0.01,
                                                                  1.05),
                                         xycoords='axes fraction',
                                         size=12, color=txt_color)

        ax[m[(mp_type, score)]].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                            0.6])
        ax[m[(mp_type, score)]].get_legend().remove()
ax[0, 0].legend(['Exp. 1', 'Exp. 2', 'predicted'])
ax[1, 1].set_visible(False)
plt.tight_layout()
plt.savefig('reports/figures/fig5.pdf')
