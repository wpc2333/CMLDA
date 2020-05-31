"""
    This script is used for the final test on the three MONKS dataset.
"""

import ipdb
import json
import nn
import numpy as np
import pandas as pd
import utils
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

###############################################################################
# EXPERIMENTAL SETUP ##########################################################

ntrials = raw_input('TRIALS[N/None]: ')
ntrials = int(ntrials) if ntrials != 'None' else None

split_percentage = 0.8

epochs = raw_input('MAX EPOCHS[N/None]: ')
epochs = int(epochs) if epochs != 'None' else None

path_to_json = '../data/final_setup/'
save_statistics = True if raw_input('SAVE STATISTICS?[Y/N] ') == 'Y' else False
saving_curve_stats = True if raw_input('SAVE CURVE STATS?[Y/N] ') == 'Y' \
    else False

statistics = pd.DataFrame(columns=['DATASET', 'MEAN_MSE_TR', 'STD_MSE_TR',
                                   'MEAN_MSE_TS', 'STD_MSE_TS',
                                   'MEAN_ACCURACY_TR', 'STD_ACCURACY_TR',
                                   'MEAN_ACCURACY_TS', 'STD_ACCURACY_TS',
                                   'CONVERGENCE', 'LS'])   # mod

statistics_time = pd.DataFrame(columns=['DATASET', 'TOT', 'BACKWARD', 'LS',
                                        'DIRECTION'])

###############################################################################
# LOADING DATASET #############################################################

fpath = '../data/monks/'

names = ['monks-1_train',
         'monks-1_test',
         'monks-2_train',
         'monks-2_test',
         'monks-3_train',
         'monks-3_test']

datasets = {name: pd.read_csv(fpath+name+'_bin.csv').values
            for name in names}

X_designs, y_designs = [], []
X_tests, y_tests = [], []

for ds in [1, 2, 3]:
    design_set = datasets['monks-{}_train'.format(ds)]
    test_set = datasets['monks-{}_test'.format(ds)]

    y_design, X_design = np.hsplit(design_set, [1])
    y_test, X_test = np.hsplit(test_set, [1])

    # simmetrized X_design:
    X_design = (X_design*2-1)
    X_test = (X_test*2-1)
    design_set = np.hstack((y_design, X_design))
    test_set = np.hstack((y_test, X_test))

    X_designs.append(X_design)
    y_designs.append(y_design)
    X_tests.append(X_test)
    y_tests.append(y_test)

###############################################################################
# OPTIMIZER SELECTIONS ########################################################

params, opt = None, raw_input('CHOOSE AN OPTIMIZER[SGD/CGD]: ')

###############################################################################
# PARAMETERS SELECTION AND TESTING ############################################

mse_tr, mse_ts = list(), list()
acc_tr, acc_ts = list(), list()
convergence_ts, acc_epochs_ts, ls_ts = list(), list(), list()   # mod
tot, bw, ls, dr = list(), list(), list(), list()   # mod
bw_p, ls_p, dr_p = list(), list(), list()   # mod

beta, momentum = None, None

if opt == 'CGD':
    beta = raw_input('CHOOSE A BETA[hs/mhs/fr/pr]: ')
    assert beta in ['hs', 'mhs', 'fr', 'pr']
else:
    momentum = raw_input('MOMENTUM TYPE[standard/nesterov]: ')
    assert momentum in ['standard', 'nesterov']

sample = raw_input('WHICH CURVE DO YOU WANT TO SAMPLE?[1/2/3/4/5] ')
sample = sample.split(',')
plot_mse = True if '1' in sample else False
plot_acc = True if '2' in sample else False
plot_norm = True if '3' in sample else False
plot_time = True if '4' in sample else False

if plot_time:
    epochs = None if raw_input('MAX EPOCHS?[Y/N] ') == 'N' else 1000
    ntrials = 1 if epochs is None else 10

sample = None if sample == [''] else np.random.randint(0, ntrials)

print 'SAMPLING ITERATION {}'.format(sample + 1) if sample is not None \
    else None

for ds in [0, 1, 2]:
    if opt == 'SGD':
        hps = path_to_json + \
            'SGD/{}/monks_{}_best_hyperparameters_sgd.json'.format(momentum,
                                                                   ds + 1)
    else:
        hps = path_to_json + \
            'CGD/monks_{}_best_hyperparameters_cgd_{}.json'.\
            format(ds + 1, beta)

    if epochs is None:
        hps = hps.replace('.json', '_no_max_epochs.json')

    with open(hps) as json_file:
        params = json.load(json_file)

    hidden_sizes = [int(i) for i in
                    params['hyperparameters']['topology'].split(' -> ')]
    hidden_sizes = hidden_sizes[1:-1]

    if opt == 'SGD':
        params['hyperparameters']['momentum'] = \
            {'type': params['hyperparameters']['momentum_type'],
             'alpha': params['hyperparameters']['alpha']}
        params['hyperparameters'].pop('momentum_type')
        params['hyperparameters'].pop('alpha')
    else:
        if plot_time and epochs is None:
            params['hyperparameters']['max_epochs'] = None

    params['hyperparameters'].pop('activation')
    params['hyperparameters'].pop('topology')

    for trial in tqdm(range(ntrials),
                      desc='TESTING DATASET {}'.format(ds + 1)):

        neural_net = nn.NeuralNetwork(X_designs[ds], y_designs[ds],
                                      hidden_sizes=hidden_sizes,
                                      activation='sigmoid', task='classifier')

        # ipdb.set_trace()

        neural_net.train(X_designs[ds], y_designs[ds], opt, epochs=epochs,
                         X_va=X_tests[ds],
                         y_va=y_tests[ds], **params['hyperparameters'])

        mse_tr.append(neural_net.optimizer.error_per_epochs[-1])
        mse_ts.append(neural_net.optimizer.error_per_epochs_va[-1])
        acc_tr.append(neural_net.optimizer.accuracy_per_epochs[-1])
        acc_ts.append(neural_net.optimizer.accuracy_per_epochs_va[-1])
        convergence_ts.append(neural_net.optimizer.statistics['epochs'])
        ls_ts.append(neural_net.optimizer.statistics['ls'])   # mod
        tot.append(neural_net.optimizer.statistics['time_train'])
        bw.append(neural_net.optimizer.statistics['time_bw'])
        ls.append(neural_net.optimizer.statistics['time_ls'])
        dr.append(neural_net.optimizer.statistics['time_dr'])
        #bw_p.append((bw[-1]/tot[-1])*100)
        #ls_p.append((ls[-1]/tot[-1])*100)
        #dr_p.append((dr[-1]/tot[-1])*100)

        neural_net.restore_weights()

        if sample is not None and sample == trial:
            saving_str = 'monks_{}_{}'.format(ds + 1, momentum) \
                if opt == 'SGD' else '{}_monks_{}'.format(beta, ds + 1)

            path = '../data/final_setup/' + str(opt)

            if momentum is not None:
                path += '/' + str(momentum)
            if beta is not None:
                path += '/' + str(beta)

            if saving_curve_stats:
                curves_fn = path + \
                    '/MONK{}_curves_{}_max_epochs_{}.json'.format(ds + 1,
                                                                  opt.lower(),
                                                                  epochs)

                with open(curves_fn, 'w') as json_file:
                    curves_data = {'error': neural_net.optimizer.
                                   error_per_epochs,
                                   'error_va': neural_net.optimizer.
                                   error_per_epochs_va,
                                   'accuracy': neural_net.optimizer.
                                   accuracy_per_epochs,
                                   'accuracy_va': neural_net.optimizer.
                                   accuracy_per_epochs_va,
                                   'gradient_norm': neural_net.optimizer.
                                   gradient_norm_per_epochs,
                                   'time': neural_net.optimizer.
                                   time_per_epochs}
                    json.dump(curves_data, json_file, indent=4)

            if plot_mse:
                utils.plot_learning_curve(
                    neural_net.optimizer,
                    [neural_net.optimizer.error_per_epochs,
                     neural_net.optimizer.error_per_epochs_va],
                    'TEST', 'MSE', neural_net.optimizer.params,
                    fname=saving_str)
            if plot_acc:
                utils.plot_learning_curve(
                    neural_net.optimizer,
                    [neural_net.optimizer.accuracy_per_epochs,
                     neural_net.optimizer.accuracy_per_epochs_va],
                    'TEST', 'ACCURACY', neural_net.optimizer.params,
                    fname=saving_str)
            if plot_norm:
                utils.plot_learning_curve(
                    neural_net.optimizer,
                    [neural_net.optimizer.gradient_norm_per_epochs],
                    'TEST', 'NORM', neural_net.optimizer.params,
                    fname=saving_str)
            if plot_time:
                utils.plot_learning_curve(
                    neural_net.optimizer,
                    [neural_net.optimizer.time_per_epochs,
                     neural_net.optimizer.error_per_epochs,
                     neural_net.optimizer.error_per_epochs_va],
                    'TEST', 'TIME', neural_net.optimizer.params,
                    fname=saving_str)

    statistics.loc[statistics.shape[0]] = ['MONKS_{}'.format(ds + 1),
                                           np.mean(mse_tr), np.std(mse_tr),
                                           np.mean(mse_ts), np.std(mse_ts),
                                           np.mean(acc_tr), np.std(acc_tr),
                                           np.mean(acc_ts), np.std(acc_ts),
                                           np.mean(convergence_ts),  # mod
                                           np.mean(ls_ts)]
    statistics_time.loc[statistics_time.shape[0]] = \
        ['MONKS_{}'.format(ds + 1), np.mean(tot), np.mean(bw), np.mean(ls),
         np.mean(dr)]

file_name = None

if opt == 'SGD':
    file_name = fpath + opt.lower() + \
        '_{}_monks_statistics.csv'.format(momentum)
    file_name_time = fpath + opt.lower() + \
        '_{}_monks_time_statistics.csv'.format(momentum)
else:
    file_name = fpath + opt.lower() + '_' + beta + \
        '_monks_statistics.csv'
    file_name_time = fpath + opt.lower() + '_' + beta + \
        '_monks_time_statistics.csv'

if epochs is None:
    file_name = file_name.replace('.csv', '_no_max_epochs.csv')
    file_name_time = file_name_time.replace('.csv', '_no_max_epochs.csv')
else:
    file_name = file_name.replace('.csv', '_max_epochs_{}.csv'.format(epochs))
    file_name_time = file_name_time. \
        replace('.csv', '_max_epochs_{}.csv'.format(epochs))

if save_statistics:
    statistics.to_csv(path_or_buf=file_name, index=False)
    statistics_time.to_csv(path_or_buf=file_name_time, index=False)
