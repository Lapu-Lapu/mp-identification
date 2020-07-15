import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.make_dataset import parse_filename
from src.models.globs import pp


def score_to_grid(scores, mesh=None, step=1):
    if mesh is None:
        mesh = pd.DataFrame(
            index=range(scores['dyn'].astype(int).min(),
                        scores['dyn'].astype(int).max(), step),
            columns=range(scores['lvm'].astype(int).min(),
                          scores['lvm'].astype(int).max(), step)
        )
        mesh.index.name = '#IPs Dynamics'
        mesh.columns.name = '#IPs Pose'
    score_type = 'cf'
    mesh.name = score_type
    for i, row in scores.iterrows():
        mesh.loc[row.dyn, row.lvm] = row.loc[score_type]
    mesh.fillna(np.nan, inplace=True)
    return mesh


def plot(mesh, models, annot=False, vmax=None):
    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(8,  5)
    cbar_ax = fig.add_axes([0.85, 0.22, 0.025, 0.57])
    fig, ax = pvplot(mesh.values(), ax=ax, step=1, cbar_ax=cbar_ax, fig=fig,
                      vmax=vmax, annot=annot)
    for i, m in enumerate(models):
        ax[i].set_title(pp[m])
    plt.subplots_adjust(right=0.8)
    ax[1].set_ylabel('')
    return fig, ax


def pvplot(meshs, ax=None, step=5, cbar_ax=None, fig=None, vmax=None, annot=True):
    if ax is None:
        fig, ax = plt.subplots(1, len(meshs), sharey=True)
    for i, m in enumerate(meshs):
        im = sns.heatmap(m, vmin=0, vmax=vmax, annot=annot, ax=ax[i],
                        cbar_ax=cbar_ax, fmt='.2f', cmap='viridis')
        ax[i].invert_yaxis()
        ax[i].set_title(m.name)
        ax[i].set_aspect('equal', 'box')
    return fig, ax


def create_scores_table(d_exp):
    cscores = d_exp.groupby('mp_type').apply(
        lambda d: d.groupby('model_id').apply(
            lambda m: {'cf': m.y.mean()}
        )
    ).reset_index(level=1)
    cscores = cscores.loc[~cscores.index.isin(['mapgpdm', 'mapcgpdm'])]
    scores = cscores.apply(
        lambda row: pd.concat([pd.Series(row[0]),
                               parse_filename(row.model_id)]),
        axis=1
    )
    return scores


D = pd.read_json('data/processed/preprocessed_data.json')
d_exp1 = D[D.version == 'main']

scores_exp1 = create_scores_table(d_exp1)

models = ['vgpdm', 'vcgpdm']
mesh = {k: score_to_grid(scores_exp1.loc[k], step=5)
        for k in models}


fig, ax = plot(mesh, models, vmax=0.5, annot=True)
cbar_ax = fig.add_axes([0.85, 0.22, 0.025, 0.57])
cbar_ax.set_ylabel('Confusion rate')
plt.savefig('reports/figures/fig3.pdf')

scores_exp2 = create_scores_table(D[D.version == 'follow-up'])

# mesh = {}
for m in ['vgpdm', 'vcgpdm']:
    x = score_to_grid(scores_exp1.loc[m])
    x = score_to_grid(scores_exp2.loc[m], x)
    mesh[m] = x

fig, ax = plot(mesh, models, vmax=0.5, annot=False)
# fig.suptitle('Confusion Rates (Experiment 1 & 2)')
for a in ax:
    a.set_xticks(5*np.arange(7)+0.5)
    a.set_xticklabels(np.arange(5, 36, 5))
    a.set_yticks(5*np.arange(7)+0.5)
    a.set_yticklabels(np.arange(5, 36, 5))
    pass
cbar_ax = fig.add_axes([0.85, 0.22, 0.025, 0.57])
cbar_ax.set_ylabel('Confusion rate')
plt.savefig('reports/figures/fig4.pdf')
