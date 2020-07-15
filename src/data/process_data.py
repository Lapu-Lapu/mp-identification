import pandas as pd
from src.models.globs import pp
from src.data.make_dataset import model_id

D = pd.read_json('data/processed/joint_results.json')
exp = ['main', 'follow-up']

# Remove participants who failed attention check
att_check = D[D.stimquality=='bad']
att_check_exp = {k: att_check.query(f"expName=='{k}'") for k in exp}
att = {k: att_check_exp[k] for k in exp}
att['all'] = att_check
att_conf = {d:100*att[d].y.mean() for d in att}
max_tol = 2
bad_vps = att_check_exp['main'].groupby('participant').y.sum() > max_tol
bad_vps = list(bad_vps[bad_vps].index)
D = D[D.participant.apply(lambda x: x not in bad_vps)]

# Remove attention checks from data set.
att_stim = att_check.loc[:, 'mov_file_generated'].unique()
for x in att_stim:
    D = D[D.mov_file_generated != x]
d1 = D[D.expName == 'main']
d2 = D[D.expName == 'follow-up']

# Catchtrials
catchcf = d1[d1.mp_type=='catchtrial'].y.mean()
# Remove catchtrials from further data analysis.
D = D[D.mp_type != 'catchtrial']

# Save result for logistic regression
scores_exp = []
for i in [1, 2]:
    scr = pd.read_json(f'data/interim/scores_exp{i}.json')
    scr['model_id'] = scr.apply(model_id, axis=1)
    scr.loc[scr.model_id == 'mapgpdm', 'model_id'] = 'map_gpdm'
    scr.loc[scr.model_id == 'mapcgpdm', 'model_id'] = 'map_cgpdm'
    scr = scr.set_index('model_id', verify_integrity=True)
    scores_exp += [scr]
scores = pd.concat(scores_exp, sort=False).reset_index()
scores.to_json('data/processed/joint_scores.json')

d_exp = {s: D[D.expName == s] for s in exp}
reg_data = []
for i, k in enumerate(exp):
    new_cols = set(scores_exp[i]) - set(d_exp[k])
    outer = d_exp[k].apply(lambda df: scores_exp[i].loc[df.model_id],
                                                  axis=1)
    reg_data += [pd.concat([d_exp[k], outer[new_cols]], axis=1)]

reg_data = pd.concat(reg_data, axis=0)

reg_data.to_json('data/processed/preprocessed_data.json')
