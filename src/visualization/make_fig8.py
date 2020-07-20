from src.models.globs import pp, model

import ordering_compare as oc
import pandas as pd
import numpy as np

import itertools
import matplotlib.pyplot as plt
import seaborn as sns


def get_mpname(s):
    mp_class = s.split('_')[0]
    if mp_class == 'map':
        return s
    else:
        return mp_class


def get_index(s):
    return list(best_models.index).index(s)


def marginal_hypothesis(first_idx, second_idx, verbose=False):
    marg = np.exp(
        np.logaddexp.reduce(
            list(
                map(
                    lambda k: order_post[k],
                    filter(
                        lambda i1: i1.index(first_idx) < i1.index(second_idx),
                        order_post.keys())))))
    if verbose:
        print("Marginal probability that ", first_idx, ">=", second_idx, ":", marg)
    return marg


df = pd.read_json("data/processed/preprocessed_data.json")
d = df[['model_id',  'y']]
mpname = d.model_id.apply(get_mpname).rename('mp_type')
d = pd.concat([mpname, d], axis=1)
d.set_index('model_id', inplace=True)
data = d.y.groupby(d.index.names).agg([len, sum]).rename(columns={'len': 'N', 'sum':  'P'})
data['mp_type'] = data.index.map(get_mpname)

data = data.assign(Q=lambda df: df.N - df.P)
data = data.assign(confusion_rate=lambda df: df.P/df.N)

g = data.groupby('mp_type')
best_models = g.confusion_rate.agg( np.argmax)

counts = np.array([g.get_group(row.mp_type).iloc[int(row.confusion_rate)][['P', 'Q']].values
                   for i, row in best_models.reset_index().iterrows()])
counts = counts.astype(float)

order_post = oc.p_orderings_given_counts(counts)
order_post = pd.DataFrame(order_post.values(), index=order_post.keys())
order_post = order_post.iloc[:, 0]

best_order = list(order_post.idxmax())
ordered_list = best_models.iloc[best_order].index
for prev, nxt in zip(ordered_list[:-1], ordered_list[1:]):
    first_idx, second_idx = get_index(prev), get_index(nxt)
    marginal_hypothesis(first_idx, second_idx, verbose=False)

marg = np.nan * np.empty((6, 6))
ordered_idx = [get_index(k) for k in ordered_list]
for first_idx, second_idx in itertools.product(ordered_idx, ordered_idx):
    if second_idx == first_idx:
        continue
    marg[first_idx, second_idx] = marginal_hypothesis(first_idx, second_idx)
pp_models = [pp[i] for i in best_models.index]
marg = pd.DataFrame(marg, columns=pp_models,
                   index=pp_models)

pp_ordered_list = [pp[k] for k in ordered_list]
sns.heatmap(marg.loc[pp_ordered_list, pp_ordered_list], annot=True, cmap=sns.color_palette('BrBG', 15))
fig = plt.gcf()
fig.set_size_inches(8, 5)
plt.tight_layout()
plt.savefig('reports/figures/fig8.pdf')
