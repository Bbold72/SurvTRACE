from easydict import EasyDict
import numpy as np
import time

import torchtuples as tt # Some useful functions

from pycox.models import DeepHitSingle

from baselines.data_class import Data
from baselines.models import simple_dln
from baselines.evaluator import Evaluator
from baselines.utils import export_results, update_run


num_runs = 1
datasets = ['metabric', 'support']

# define the setup parameters
config_metabric = EasyDict({
    'data': 'metabric',
    'horizons': [.25, .5, .75],
    'batch_size': 64,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})
config_support = EasyDict({
    'data': 'support',
    'horizons': [.25, .5, .75],
    'batch_size': 128,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1
})

for dataset_name in datasets:
    print('Running DeepHit on ' + dataset_name)

    config = config_metabric if dataset_name == 'metabric' else config_support
    config.model = 'DeepSurv'

    # store each run in list
    runs_list = []

    for i in range(num_runs):

        # load data
        data = Data(config)

        # define neural network
        net = simple_dln(config)

        # initialize model
        model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=np.array(config['duration_index'], dtype='float32'))
        model.optimizer.set_lr(config.learning_rate)
        callbacks = [tt.callbacks.EarlyStopping(patience=20)]

        # train model
        train_time_start = time.time()
        log = model.fit(data.x_train, data.y_train, config.batch_size, config.epochs, callbacks, val_data=data.val_data)
        train_time_finish = time.time()


        # calcuate metrics
        evaluator = Evaluator(data, model, config, offset=0)
        run = evaluator.eval()
        run = update_run(run, train_time_start, train_time_finish, log.epoch)

        runs_list.append(run)

    export_results(runs_list, config)

