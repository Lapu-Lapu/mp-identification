# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd

import os
import re
import glob
from itertools import chain

from src.data.globs import (
    model,
    exp1_columns_raw,
    exp2_columns_raw
)


@click.command()
@click.argument('command', type=click.STRING)
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(command, input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(command)
    funcd = {
        'exp1': load_exp1_data,
        'exp2': load_exp2_data,
        'scores1': load_scores1,
        'scores2': load_scores2,
        'join': join_everything
    }
    funcd[command](input_filepath, output_filepath)


def join_everything(input_filepath, output_filepath):
    data_exp1 = pd.read_json(os.path.join(input_filepath, 'temp_exp1.json'))
    data_exp2 = pd.read_json(os.path.join(input_filepath, 'temp_exp2.json'))
    all_data = pd.concat((data_exp1, data_exp2), sort=False,
                         ignore_index=True)

    all_data['model_id'] = all_data.apply(modelstr, axis=1)

    out_file = os.path.join(output_filepath, 'joint_results.json')
    all_data.to_json(out_file)
    return


def load_exp1_data(input_filepath, output_filepath) -> pd.DataFrame:
    input_filepath = os.path.join(input_filepath, 'psychophysics', 'exp1', '')
    files = [input_filepath + f
             for f in os.listdir(input_filepath)
             if f[-3:] == 'csv']
    df_list = [load_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    df.loc[:, 'y'] = 1 - df.loc[:, 'y']
    D = df.sort_index()

    # Due to technical difficulties, exclude vp 15
    D = D[D.participant != 'xx15w25']
    D['expName'] = 'main'
    D['stimquality'] = D.mov_file_generated.apply(
        get_stimquality_exp1)
    D.to_json(os.path.join(output_filepath, 'temp_exp1.json'))


def load_scores1(input_filepath, output_filepath):
    scores_exp1 = load_scores_exp1(
        os.path.join(input_filepath, 'scores', 'ELBO_MSE_table_nogaps.csv')
    )
    scores_exp1.to_json(os.path.join(output_filepath, 'scores_exp1.json'))


def load_exp2_data(input_filepath, output_filepath):
    input_filepath = os.path.join(input_filepath, 'psychophysics', 'exp2', '')
    pattern = os.path.join(input_filepath, '*.csv')
    fps = glob.glob(pattern)
    data_exp2 = load_exp2(fps)

    # Remove vp 4 (myself)
    data_exp2 = data_exp2[data_exp2.participant != 4]
    data_exp2.to_json(os.path.join(output_filepath, 'temp_exp2.json'))
    return


def load_scores2(input_filepath, output_filepath):
    scores_exp2 = pd.read_json(
        os.path.join(input_filepath, 'scores', 'modelscores_exp2.json')
    )
    scores_exp2['version'] = 'follow-up'
    scores_exp2.drop(['WRAP_DYN', 'WRAP_PATH', 'dataset',
                      'elbo', 'mode', 'timing'], axis=1, inplace=True)
    scores_exp2.to_json(os.path.join(output_filepath, 'scores_exp2.json'))
    return


def load_scores_exp1(fn):
    df = pd.read_csv(fn)
    df.rename(columns={'prm0_dyn': 'dyn', 'prm1_lvm': 'lvm'},
              inplace=True)

    params = set(chain(*[v['params'] for v in model.values()]))
    e = pd.DataFrame(np.zeros((df.shape[0], len(params)-2))*np.nan,
                     columns=params-{'dyn','lvm'})
    df = pd.concat((e, df), axis=1)
    def reprm(row):
        if row.mp_type in ["mapcgpdm", "mapgpdm"]:
            row.lvm = row.dyn = np.nan
        elif row.mp_type not in ['vcgpdm', 'vgpdm']:
            row[model[row.mp_type]['params']] = row.dyn
            row.lvm = row.dyn = np.nan
        return row
    df = df.apply(reprm, axis=1)
    df.drop(['confusion_rate', 'std'], axis=1, inplace=True)
    df['version'] = 'main'
    return df


def load_csv(fn):
    df = pd.read_csv(fn, usecols=exp1_columns_raw)

    # change trial index
    df = df[~np.isnan(df.trial_number)]

    # adapt to new experiment names
    df = df.rename({"trials.thisN": 'n_trial',
                    "trials.thisTrialN": "n_inblock"}, axis=1)

    # get not natural stimulusname
    df['mov_file_generated'] = df.apply(get_artificial, axis=1)
    df['mov_file_natural'] = df.apply(get_natural, axis=1)

    # catchtrials: correct specifies the unseen video
    e = df.loc[df.mov_file_generated=='catchtrial'].apply(catchtype_correct, axis=1)
    df.loc[df.mov_file_generated=='catchtrial', 'correct'] = e

    # read stimulus parameters
    stimulus_params = df.apply(parse_row, axis=1)

    # put parsed stimulus info into the dataframe
    df = pd.concat([stimulus_params, df], axis=1)

    # add the response y to the dataframe (1 = natural stimulus chosen)
    try:
        y = pd.Series(df.correct == df['participant_key_response'],
                    dtype=int, name='y')
    except KeyError:
        y = pd.Series(df.correct == df['key_response.keys'],
                    dtype=int, name='y')

    df = pd.concat([df, y], axis=1)
    return df


def load_exp2(filenames: list):
    """
    Loads data from list of csv files to DataFrame.
    """
    Data = pd.concat((pd.read_csv(fp, usecols=exp2_columns_raw)
                      for fp in filenames),
                     sort=False,
                     ignore_index=True)
    d = pd.concat((Data, Data.mov_file_generated.apply(parse_filename)),
                   axis=1, sort=False)
    y = 1 - d['response.corr']
    y.name = 'y'
    d.drop(columns='response.corr', inplace=True)

    # remove path in filename
    d['stimquality'] = d.mov_file_generated.apply(get_stimquality_exp2)
    d.loc[:, 'mov_file_generated'] = d.loc[:, 'mov_file_generated'].apply(
        lambda x: x.split('generated\\')[-1].split('.')[0])
    d.rename({'model': 'mp_type'}, axis=1, inplace=True)
    d.loc[d.parts==1, 'mp_type'] = 'vgpdm'
    d.drop(columns='parts', inplace=True)
    d['expName'] = 'follow-up'
    return pd.concat((d, y), axis=1, sort=False)


def get_stimquality_exp1(s):
    last = s.split('final')[-1].split('_')
    if len(last) > 1:
        return 'bad'
    else:
        return 'good'


def get_stimquality_exp2(s):
    fin = s.split('_')[-1]
    if fin == 'fail.mov':
        return 'bad'
    elif fin in ['hip-corrected.mov', 'hip-correted.mov']:
        return 'good'
    else:
        return 'good'


def get_artificial(row: pd.Series):
    """
    Returns name of generated stimulus.
    """
    inv = {'left': 'right', 'right': 'left'}
    return _get_stimulus_name(row, inv)


def get_natural(row: pd.Series):
    """
    Returns name of natural stimulus.
    """
    return _get_stimulus_name(row, {})


def _get_stimulus_name(row, transform_dict):
    try:
        return row[transform_dict.get(row.correct, row.correct)].split('.')[0]
    except KeyError:
        return 'catchtrial'


def catchtype_correct(row):
    r = row.right.split('(')[0]
    l = row.left.split('(')[0]
    if r == 'catch':
        return 'right'
    else:
        return 'left'


def parse_row(row: pd.Series):
    """
    Read stimulus params of main experiment.
    """
    stimulus_str = row.mov_file_generated
    model = model_from_str(stimulus_str)
    return pd.concat((model, parse_filename(stimulus_str)))


def model_from_str(s: str):
    l = s.split('_')
    if l[0] != 'map':
        return pd.Series({'mp_type': l[0]})
    else:
        return pd.Series({'mp_type': ''.join(l[:2])})


def parse_filename(s: str) -> pd.Series:
    """
    Finds numbers in brackets, associates this value
    with a key in front of the bracket.
    Returns Series like {'model': 'vcgpdm',...}
    """
    inparenthesis = re.compile(r'\([a-zA-Z0-9]*\)')
    beforeparenthesis = re.compile(r'[a-zA-Z0-9]*\(')
    A = map(remove_brackets, re.findall(inparenthesis, s))
    B = map(remove_brackets, re.findall(beforeparenthesis, s))
    return pd.Series({b: a for a, b in zip(A, B)})


def remove_brackets(s):
    s = s.replace('(', '').replace(')', '')
    try:
        return int(s)
    except ValueError:
        return s


def modelstr(row: pd.Series) -> str:
    fn = row.mov_file_generated.split('_final')[0]
    if row.expName == 'main':
        return fn
    elif row.expName == 'follow-up':
        id_dict = parse_filename(fn)
        s = row.mp_type + '_'
        mp = model[row.mp_type]
        for p in mp['params']:
            s += f'{p}({id_dict[p]})-'
        return s[:-1]
    else:
        raise NameError(f'{row.expName} unknown')


class UserError(Exception):
    pass


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
