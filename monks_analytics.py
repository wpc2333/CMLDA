"""
  This script is used in order to test and validate the network for the
  three MONKS datasets.
"""

import ipdb
import json
import nn
import numpy as np
import pandas as pd
import utils as u
import validation as val
import warnings

warnings.filterwarnings("ignore")

###############################################################################
# EXPERIMENTAL SETUP ##########################################################

ds, nfolds = int(raw_input('CHOOSE A MONK DATASET[1/2/3]: ')), 5
grid_size = 10
split_percentage = 0.8
epochs = None
momentum = None
beta_choice = None
save_statistics = False

fpath = '../data/monks/'
preliminary_path = '../images/monks_preliminary_trials/'

###############################################################################
# LOADING DATASET #############################################################

names = ['monks-1_train',
         'monks-1_test',
         'monks-2_train',
         'monks-2_test',
         'monks-3_train',
         'monks-3_test']

datasets = {name: pd.read_csv(fpath + name + '_bin.csv').values
            for name in names}

design_set = datasets['monks-{}_train'.format(ds)]
test_set = datasets['monks-{}_test'.format(ds)]

y_design, X_design = np.hsplit(design_set, [1])
y_test, X_test = np.hsplit(test_set, [1])

# simmetrized X_design:
X_design = (X_design*2-1)
X_test = (X_test*2-1)
design_set = np.hstack((y_design, X_design))
test_set = np.hstack((y_test, X_test))

###############################################################################
# DATASET PARTITIONING ########################################################

np.random.shuffle(design_set)

training_set = design_set[:int(design_set.shape[0]*split_percentage), :]
validation_set = design_set[int(design_set.shape[0]*split_percentage):, :]

y_training, X_training = np.hsplit(design_set, [1])
y_validation, X_validation = np.hsplit(validation_set, [1])
y_test, X_test = np.hsplit(test_set, [1])

###############################################################################
# NETWORK INITIALIZATION ######################################################

testing, testing_betas, validation = False, False, False

if raw_input('TESTING OR VALIDATION[testing/validation]? ') == 'validation':
    validation = True
else:
    save_statistics = True if raw_input('SAVE STATISTICS?[Y/N] ') == 'Y'\
     else False
    testing = True

neural_net, initial_W, initial_b = None, None, None

if testing or testing_betas:
    neural_net = nn.NeuralNetwork(X_training, y_training, hidden_sizes=[4, 8],
                                  activation='sigmoid')
    initial_W, initial_b = neural_net.W, neural_net.b

###############################################################################
# PRELIMINARY TRAINING ########################################################
#
pars = {}
betas = ['hs', 'pr', 'fr', 'mhs', 'dy', 'dl', 'cd']
errors, errors_std = [], []
acc, acc_std = [], []
mse_tr, mse_ts = list(), list()
acc_tr, acc_ts = list(), list()
convergence_ts, acc_epochs_ts, ls_ts = list(), list(), list()   # mod
tot, bw, ls, dr = list(), list(), list(), list()   # mod
bw_p, ls_p, dr_p = list(), list(), list()   # mod

statistics = pd.DataFrame(columns=['DATASET', 'MEAN_MSE_TR', 'STD_MSE_TR',
                                   'MEAN_MSE_TS', 'STD_MSE_TS',
                                   'MEAN_ACCURACY_TR', 'STD_ACCURACY_TR',
                                   'MEAN_ACCURACY_TS', 'STD_ACCURACY_TS',
                                   'CONVERGENCE', 'LS'])   # mod

statistics_time = pd.DataFrame(columns=['DATASET', 'TOT', 'BACKWARD', 'LS',
                                        'DIRECTION'])


opt = str(raw_input("OPTIMIZER[SGD/CGD]: "))

print type(opt)

if testing:
    if opt == 'SGD':
        momentum = str(raw_input("MOMENTUM[nesterov/standard]: "))

        pars = {'epochs': epochs,
                'batch_size': X_training.shape[0],
                'eta': 0.61,
                'momentum': {'type': momentum, 'alpha': 0.83},
                'reg_lambda': 0.0,
                'reg_method': 'l2'}

        # if ds == 3:
        #     pars['reg_lambda'] = 0.002
        neural_net.train(X_training, y_training, opt, X_va=X_validation,
                         y_va=y_validation, **pars)
    else:
        testing_betas = True \
            if raw_input('TESTING BETAS[Y/N]? ') == 'Y' else False
        if testing_betas is False:
            beta_choice = str(raw_input("BETA[hs/pr/fr/mhs/dy/dl/cd]: "))
        pars = {'max_epochs': epochs,
                'error_goal': 1e-4,
                'strong': True,
                'rho': 0.0}
        if testing_betas:
            for beta in betas:
                pars['beta_m'] = beta

                for d_m in ['standard', 'modified']:
                    print 'TESTING BETA {} WITH DIRECTION {}'.\
                        format(beta.upper(), d_m.upper())

                    pars['plus'] = True
                    pars['d_m'] = d_m
                    pars['sigma_2'] = 0.3

                    neural_net.train(X_training, y_training, opt,
                                     X_va=X_validation, y_va=y_validation,
                                     **pars)
                    neural_net.update_weights(initial_W, initial_b)
                    neural_net.update_copies()

                    if d_m == 'standard':
                        errors_std.\
                            append(neural_net.optimizer.error_per_epochs)
                        acc_std.\
                            append(neural_net.optimizer.
                                   accuracy_per_epochs_va)
                    else:
                        errors.\
                            append(neural_net.optimizer.error_per_epochs)
                        acc.\
                            append(neural_net.optimizer.
                                   accuracy_per_epochs_va)

        else:
            pars['plus'] = True
            pars['beta_m'] = beta_choice
            if beta_choice == 'mhs':
                pars['rho'] = 0.67
            neural_net.train(X_training, y_training, opt, X_va=X_validation,
                             y_va=y_validation, **pars)

    if testing_betas:
        u.plot_betas_learning_curves(ds, betas, [errors_std, errors],
                                     'ERRORS', 'MSE')
        u.plot_betas_learning_curves(ds, betas, [acc_std, acc],
                                     'ACCURACY', 'ACCURACY')
    else:
        print '\n'
        print 'INITIAL ERROR: {}'.\
            format(neural_net.optimizer.error_per_epochs[0])
        print 'FINAL ERROR: {}'.\
            format(neural_net.optimizer.error_per_epochs[-1])
        print 'INITIAL VALIDATION ERROR: {}'.format(neural_net.optimizer.
                                                    error_per_epochs_va[0])

        print 'FINAL VALIDATION ERROR: {}'.format(neural_net.optimizer.
                                                  error_per_epochs_va[-1])
        print 'EPOCHS OF TRAINING: {}'.format(len(neural_net.optimizer.
                                                  error_per_epochs))
        print 'CONVERGENCE EPOCH: {}'.format(neural_net.optimizer.
                                             statistics['epochs'])
        print '\n'

        path = '../data/final_setup/analytics/' + str(opt)

        if momentum is not None:
            path += '/' + str(momentum)
        if opt == 'CGD':
            path += '/' + str(beta_choice)

        with open(path + '/MONK{}_curves_{}.json'.
                  format(ds, opt.lower()), 'w') as json_file:
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

        statistics.loc[statistics.shape[0]] = ['MONKS_{}'.format(ds),
                                               np.mean(mse_tr), np.std(mse_tr),
                                               np.mean(mse_ts), np.std(mse_ts),
                                               np.mean(acc_tr), np.std(acc_tr),
                                               np.mean(acc_ts), np.std(acc_ts),
                                               np.mean(convergence_ts),  # mod
                                               np.mean(ls_ts)]

        statistics_time.loc[statistics_time.shape[0]] = \
            ['MONKS_{}'.format(ds), np.mean(tot), np.mean(bw), np.mean(ls),
                np.mean(dr)]

        file_name = None

        if opt == 'SGD':
            file_name = path + \
                '/{}_{}_monks_statistics.csv'.format(ds, momentum)
            file_name_time = path + \
                '/{}_{}_monks_time_statistics.csv'.format(ds, momentum)
        else:
            file_name = path + \
                '/{}_monks_statistics.csv'.format(ds)
            file_name_time = path + \
                '/{}_monks_time_statistics.csv'.format(ds)

        if save_statistics:
            statistics.to_csv(path_or_buf=file_name, index=False)
            statistics_time.to_csv(path_or_buf=file_name_time, index=False)

        # u.plot_learning_curve_with_info(
        #     neural_net.optimizer,
        #     [neural_net.optimizer.error_per_epochs,
        #      neural_net.optimizer.error_per_epochs_va], 'VALIDATION',
        #     'MSE', neural_net.optimizer.params)
        # u.plot_learning_curve_with_info(
        #     neural_net.optimizer,
        #     [neural_net.optimizer.accuracy_per_epochs,
        #      neural_net.optimizer.accuracy_per_epochs_va], 'VALIDATION',
        #     'ACCURACY', neural_net.optimizer.params)
        # u.plot_learning_curve(
        #     neural_net.optimizer,
        #     [neural_net.optimizer.gradient_norm_per_epochs],
        #     'TEST', 'NORM', neural_net.optimizer.params)
        # u.plot_all_learning_curves(ds + 1, betas, [[neural_net.optimizer.
        #                            time_per_epochs]], 'ERRORS', 'MSE',
        #                            time=True, semilogy=False)

###############################################################################
# VALIDATION ##################################################################

if validation:
    experiment = 1
    param_ranges = {}

    if opt == 'SGD':
        param_ranges['eta'] = (0.6, 0.8)

        type_m = str(raw_input('MOMENTUM TYPE[standard/nesterov]: '))
        assert type_m in ['standard', 'nesterov']
        param_ranges['type'] = type_m

        param_ranges['alpha'] = (0.5, 0.9)
        param_ranges['reg_method'] = 'l2'
        param_ranges['reg_lambda'] = (0.001, 0.01)
        param_ranges['epochs'] = epochs
    else:
        beta_m = str(raw_input('CHOOSE A BETA[hs/pr/fr/mhs/dy/dl/cd]: '))
        assert beta_m in betas
        param_ranges['beta_m'] = beta_m

        d_m = str(raw_input('CHOOSE A DIRECTION METHOD[standard/modified]: '))
        assert d_m in ['standard', 'modified']
        param_ranges['d_m'] = d_m

        param_ranges['max_epochs'] = epochs
        param_ranges['error_goal'] = 1e-4
        param_ranges['strong'] = True
        param_ranges['plus'] = True
        param_ranges['sigma_2'] = (0.1, 0.4)

        if beta_m == 'mhs':
            param_ranges['rho'] = (0., 1.)
        else:
            param_ranges['rho'] = 0.0

    param_ranges['optimizer'] = opt
    param_ranges['hidden_sizes'] = [4, 8]
    param_ranges['activation'] = 'sigmoid'
    param_ranges['task'] = 'classifier'

    grid = val.HyperGrid(param_ranges, grid_size, random=True)
    selection = val.ModelSelectionCV(grid,
                                     fname=fpath +
                                     'monks_{}_experiment_{}_results.json.gz'.
                                     format(ds, experiment))
    selection.search(X_design, y_design, nfolds=nfolds)
    best_hyperparameters = selection.select_best_hyperparams()

    json_name = ''

    if opt == 'SGD':
        json_name = '../data/final_setup/{}/{}/'.format(opt, type_m) + \
            'monks_{}_best_hyperparameters_{}.json'.format(ds, opt.lower())
    else:
        json_name = '../data/final_setup/{}/'.format(opt) + \
            'monks_{}_best_hyperparameters_{}_{}.json'.format(ds, opt.lower(),
                                                              param_ranges
                                                              ['beta_m'])

    if epochs is None:
        json_name = json_name.replace('.json', '_no_max_epochs.json')

    with open(json_name, 'w') as json_file:
        json.dump(best_hyperparameters, json_file, indent=4)
