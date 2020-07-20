import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models.globs import model, beta_std, pp
from src.data.make_dataset import remove_brackets

def get_param(s):
    inparenthesis = re.compile(r'\([a-zA-Z0-9]*\)')
    return remove_brackets(re.findall(inparenthesis, s)[0])

def plot_conf_vs_1dparam(df, param, score='confusion_rate', ax=None, yerr=None):
    df = df.sort_values(by=param)
    df = df.sort_values(by='version', ascending=False)
    cmap = {'main': 'blue', 'follow-up': 'red'}
    color = df.loc[:, 'version'].map(cmap)
    df[param] = df[param].astype(int)
    ax = df.plot(x=param, y=score, kind='bar', yerr=yerr,
                 legend=False, color=color, ax=ax)
    ax.set_xlabel(pp[model[df.iloc[0]['mp_type']]['params'][0]])
    ylabel = score.replace('_', ' ')
    ylabel = ylabel[0].upper() + ylabel[1:]
    ax.set_ylabel(ylabel)
    return ax


D = pd.read_json('data/processed/preprocessed_data.json')
scores = pd.read_json('data/processed/joint_scores.json')

df = D[D.mp_type.apply(lambda x: x in ['tmp', 'dmp'])]
scores = scores[scores.mp_type.apply(lambda x: x in ['tmp', 'dmp'])]
scores = scores[scores.model_id.apply(lambda x: x not in ['dmp_nspi(10)', 'dmp_npsi(5)'])]
df = df.groupby('version').apply(
        lambda m: m.groupby('mp_type').apply(
            lambda d: d.groupby('model_id').y.apply(
                lambda x: {'confusion_rate': x.mean(), 'std': beta_std(x)}
            )
        )
).unstack().reset_index()
df['param'] = df.model_id.apply(get_param)

fig, ax = plt.subplots(nrows=3, ncols=2, sharex='col',
                       sharey='row', figsize=(8, 9))
s = (0, 0)
ax[s] = plot_conf_vs_1dparam(df[df.mp_type=='tmp'], 'param', 'confusion_rate', yerr='std', ax=ax[s])
ax[s].grid()
s = (1, 0)
ax[s] = plot_conf_vs_1dparam(scores[scores.mp_type=='tmp'], 'numprim', 'MSE', ax=ax[s])
ax[s].grid()
ax[s].set_title('TMP')
s = (0, 1)
ax[s] = plot_conf_vs_1dparam(df[df.mp_type=='dmp'], 'param', 'confusion_rate', yerr='std', ax=ax[s])
ax[s].grid()
ax[s].set_title('DMP')
s = (1, 1)
ax[s] = plt.subplot(324, sharey=ax[1, 0])
ax[1, 1] = plot_conf_vs_1dparam(scores[scores.mp_type=='dmp'], 'npsi', 'MSE', ax=ax[1, 1])
ax[s].grid()
ax[s].set_visible(True)
ax[2, 0] = plot_conf_vs_1dparam(scores[scores.mp_type=='tmp'], 'numprim', 'ELBO', ax=ax[2, 0])
ax[2, 0].grid()
ax[2, 1].set_visible(False)
ax[2, 0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
plt.savefig('reports/figures/fig2.pdf')
