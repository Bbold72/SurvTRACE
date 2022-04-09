from easydict import EasyDict
import numpy as np
import time

import torchtuples as tt # Some useful functions

from pycox.models import PCHazard

from baselines.data_class import Data
from baselines.models import simple_dln
from baselines.evaluator import Evaluator
from baselines.utils import export_results, update_run


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
    'dropout': 0.1
})
config_support = EasyDict({
    'data': 'support',
    'horizons': [.25, .5, .75],
    'batch_size': 128,
    'learning_rate': 0.01,
    'epochs': 50,
    'hidden_size': 32,
    'dropout': 0.1
})
config_seer = EasyDict({
    'data': 'seer',
    'horizons': [.25, .5, .75],
    'batch_size': 1024,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1,
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
        evaluator = Evaluator(data, model, config)
        run = evaluator.eval()
        run = update_run(run, train_time_start, train_time_finish, log.epoch)

        runs_list.append(run)

    export_results(runs_list, config)