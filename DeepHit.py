from easydict import EasyDict
import numpy as np
import time

import torchtuples as tt # Some useful functions

from pycox.models import DeepHitSingle, DeepHit

from baselines.data_class import Data
from baselines.models import simple_dln, CauseSpecificNet
from baselines.evaluator import EvaluatorSingle, EvaluatorCompeting
from baselines.utils import export_results, update_run


num_runs = 1
datasets = ['metabric', 'support', 'seer']

# define the setup parameters
config_metabric = EasyDict({
    'data': 'metabric',
    'horizons': [.25, .5, .75],
    'batch_size': 64,
    'learning_rate': 0.01,
    'epochs': 1,
    'hidden_size': 32,
    'dropout': 0.1
})
config_support = EasyDict({
    'data': 'support',
    'horizons': [.25, .5, .75],
    'batch_size': 128,
    'learning_rate': 0.01,
    'epochs': 1,
    'hidden_size': 32,
    'dropout': 0.1
})
config_seer = EasyDict({
    'data': 'seer',
    'horizons': [.25, .5, .75],
    'batch_size': 1024,
    'learning_rate': 0.01,
    'epochs': 1,
    'hidden_size_indiv': 32,
    'hidden_size_shared': 64,
    'dropout': 0.1
})
config_dic = {
    'metabric': config_metabric,
    'support': config_support,
    'seer': config_seer
}

for dataset_name in datasets:
    print('Running DeepHit on ' + dataset_name)

    config = config_dic[dataset_name]
    config.model = 'DeepHit'

    # store each run in list
    runs_list = []

    for i in range(num_runs):

        # load data
        data = Data(config)

        # define neural network
        if config.data == 'seer':
            net = CauseSpecificNet(config)
            optimizer = tt.optim.AdamWR(lr=0.01, decoupled_weight_decay=0.01,
                                cycle_eta_multiplier=0.8)
            # initialize model
            model = DeepHit(net, optimizer, 
                            alpha=0.2, 
                            sigma=0.1,
                            duration_index=config.duration_index
                            )
            
            Evaluator = EvaluatorCompeting
        else:
            net = simple_dln(config)

            # initialize model
            model = DeepHitSingle(net, tt.optim.Adam, 
                                    alpha=0.2, 
                                    sigma=0.1, 
                                    duration_index=np.array(config['duration_index'],
                                    dtype='float32')
                                    )
            model.optimizer.set_lr(config.learning_rate)

            Evaluator = EvaluatorSingle

        # add early stopping
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

