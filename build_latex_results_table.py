"""
    This script is used in order to build the LaTex table containing the
    results for the three MONKS datasets. The table can be seen in the
    report's third chapter.
"""

import ipdb
import json
import numpy as np
import pandas as pd

epochs = raw_input('MAX EPOCHS[N/None]: ')
epochs = int(epochs) if epochs != 'None' else None

tab_columns = ['Task', 'Optimizer', 'sigma_1', 'sigma_2', 'rho', 'eta',
               'alpha', 'lambda', 'MSE (TR - TS)', 'Accuracy (TR - TS) (%)']

path_to_json = '../data/final_setup/' 
from_analytics = True if raw_input('RETRIEVE FROM ANALYTICS[Y/N]? ') == 'Y' \
    else False
data = '../data/final_setup/analytics/' if from_analytics else '../data/monks/'

paths = []

for opt in ['sgd', 'cgd']:
    methods = ['standard', 'nesterov'] if opt == 'sgd' else ['pr', 'hs', 'mhs','dy', 'dl', 'cd']

    for m in methods:
        p = '{}_{}_monks_statistics.csv'.format(opt, m)

        if epochs is not None:
                p = p.replace('.csv', '_max_epochs_{}.csv'.format(epochs))
        else:
            if from_analytics:
                p = p.replace('.csv', '_no_max_epochs.csv')

        paths.append(data + opt.upper() + '/' + m + '/' + p if
                     from_analytics else data + p)

datasets = {'cm': pd.read_csv(paths[0]), 'nag': pd.read_csv(paths[1]),
            'pr': pd.read_csv(paths[2]), 'hs': pd.read_csv(paths[3]),
            'mhs': pd.read_csv(paths[4])}

table = pd.DataFrame(columns=tab_columns)

for monk in [1, 2, 3]:
    for opt in ['cm', 'nag', 'pr', 'hs', 'mhs', 'dy', 'dl', 'cd']:
        opt_name, opt_type = '', ''
        if opt in ['cm', 'nag']:
            opt_name, opt_type = 'SGD ({})'.format(opt.upper()), 'SGD'
        else:
            opt_name, opt_type = 'CGD ({})'.format(opt.upper()), 'CGD'

        if opt_type == 'SGD':
            m = 'standard' if opt == 'cm' else 'nesterov'
            hps = path_to_json + \
                'SGD/{}/monks_{}_best_hyperparameters_sgd.json'.format(m, monk)
        else:
            hps = path_to_json + \
                'CGD/monks_{}_best_hyperparameters_cgd_{}.json'.format(monk,
                                                                       opt)

        with open(hps) as json_file:
            params = json.load(json_file)
            sigma_1 = params['hyperparameters']['sigma_1'] \
                if opt in ['pr', 'hs', 'mhs'] else np.nan
            sigma_2 = np.round(params['hyperparameters']['sigma_2'], 2) \
                if opt in ['pr', 'hs', 'mhs'] else np.nan
            rho = np.round(params['hyperparameters']['rho'], 2) \
                if opt in ['pr', 'hs', 'mhs'] else np.nan
            eta = np.round(params['hyperparameters']['eta'], 2) \
                if opt in ['cm', 'nag'] else np.nan
            alpha = np.round(params['hyperparameters']['alpha'], 2) \
                if opt in ['cm', 'nag'] else np.nan

            lamb = None
            if monk == 3:
                lamb = np.round(params['hyperparameters']['reg_lambda'], 4) \
                    if opt in ['cm', 'nag'] else np.nan
            else:
                lamb = 0.0 if opt in ['cm', 'nag'] else np.nan

            mse = '{:.2e} - {:.2e}'.format(datasets[opt].iloc[monk - 1, 1],
                                           datasets[opt].iloc[monk - 1, 3])
            acc = '{:.2e} % - {:.2e} %'.format(datasets[opt].iloc[monk - 1, 5],
                                               datasets[opt].iloc[monk - 1, 7])

            table.loc[table.shape[0]] = \
                ['MONK {}'.format(monk), opt_name, sigma_1, sigma_2, rho,
                 eta, alpha, lamb, mse, acc]

table_name = 'table.txt' if epochs is None \
    else 'table_max_epochs_{}.txt'.format(epochs)

table.to_latex(buf=data + table_name, index=False, na_rep='-',
               index_names=False,
               column_format=(('| c ' * len(table.columns)) + '|'))
