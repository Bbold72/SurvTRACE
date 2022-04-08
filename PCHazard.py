from collections import defaultdict
from easydict import EasyDict
import numpy as np
from pathlib import Path
import pickle
from sksurv.metrics import concordance_index_ipcw
import time

import torchtuples as tt # Some useful functions

from pycox.models import PCHazard

from baselines.data_class import Data
from baselines.models import simple_dln


num_runs = 1
# datasets = ['metabric', 'support', ('seer', 'event_0'), ('seer', 'event_1')]
datasets = ['metabric', 'support']

# define the setup parameters
config_metabric = EasyDict({
    'data': 'metabric',
    'horizons': [.25, .5, .75],
    'batch_size': 64,
    'learning_rate': 0.01,
    'epochs': 50,
    'hidden_size': 32,
    'dropout': 0.1,
    'val_batch_size': None 
})
config_support = EasyDict({
    'data': 'support',
    'horizons': [.25, .5, .75],
    'batch_size': 128,
    'learning_rate': 0.01,
    'epochs': 50,
    'hidden_size': 32,
    'dropout': 0.1,
    'val_batch_size': None
})
config_seer = EasyDict({
    'data': 'seer',
    'horizons': [.25, .5, .75],
    'batch_size': 1024,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1,
    'val_batch_size': 10000,
    # event_0: Heart Disease
    # event_1: Breast Cancer
    'event_to_censor': 'event_0'
})
config_dic = {
    'metabric': config_metabric,
    'support': config_support,
    'seer': config_seer
}


for dataset_name in datasets:

    if type(dataset_name) == tuple:
        dataset_name, config.event_to_censor = dataset_name

    config = config_dic[dataset_name]
    config.model = 'PCHazard'


    try:
        event_name = '-' + config.event_to_censor
    except AttributeError:
        event_name = ''

    print('Running PC Hazard' + event_name + ' on ' + dataset_name)

    # store each run in list
    runs_list = []

    for i in range(num_runs):

        # load data
        data = Data(config)

        # define neural network
        net = simple_dln(config)

        # initalize model
        model = PCHazard(net, tt.optim.Adam, duration_index=np.array(config['duration_index'], dtype='float32'))
        model.optimizer.set_lr(config.learning_rate)
        callbacks = [tt.callbacks.EarlyStopping(patience=20)]

        # train model
        train_time_start = time.time()
        log = model.fit(data.x_train, data.y_train, config.batch_size, config.epochs, callbacks, val_data=data.val_data)
        train_time_finish = time.time()

        # calcuate metrics
        def evaluator(df, train_index, model, test_set, config):
            df_train_all = df.loc[train_index]
            get_target = lambda df: (df['duration'].values, df['event'].values)
            durations_train, events_train = get_target(df_train_all)
            et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                            dtype = [('e', bool), ('t', float)])
            times = config['duration_index'][1:-1]
            horizons = config['horizons']

            df_test, df_y_test = test_set
            surv = model.predict_surv_df(df_test, batch_size=config.val_batch_size)
            risk = np.array((1 - surv).transpose())
            
            durations_test, events_test = get_target(df_y_test)
            et_test = np.array([(events_test[i], durations_test[i]) for i in range(len(events_test))],
                        dtype = [('e', bool), ('t', float)])
            metric_dict = defaultdict(list)
            cis = []
            for i, _ in enumerate(times):
                cis.append(
                    concordance_index_ipcw(et_train, et_test, estimate=risk[:, i+1], tau=times[i])[0]
                    )
                metric_dict[f'{horizons[i]}_ipcw'] = cis[i]

            for horizon in enumerate(horizons):
                print(f"For {horizon[1]} quantile,")
                print("TD Concordance Index - IPCW:", cis[horizon[0]])
            
            return metric_dict

        run = evaluator(data.df, data.df_train.index, model, (data.x_test, data.df_y_test), config=config)
        run['train_time'] = train_time_finish - train_time_start
        run['epochs_trained'] = log.epoch
        run['time_per_epoch'] =  run['train_time'] / run['epochs_trained']

        runs_list.append(run)

    file_name = 'PCHazard'  + event_name  + '_' + config['data'] + '.pickle'
    with open(Path('results', file_name), 'wb') as f:
        pickle.dump(runs_list, f)

