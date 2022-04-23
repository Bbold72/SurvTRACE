from easydict import EasyDict
import time

import torchtuples as tt # Some useful functions

from pycox.models import CoxPH

from baselines.data_class import Data
from baselines.models import simple_dln
from baselines.evaluator import EvaluatorSingle
from baselines.utils import export_results, update_run


num_runs = 10
datasets = ['metabric', 'support', ('seer', 'event_0'), ('seer', 'event_1')]
# datasets = ['metabric', 'support']
# datasets = [('seer', 'event_0'), ('seer', 'event_1')]
# datasets = [('seer', 'event_1')]

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
config_seer = EasyDict({
    'data': 'seer',
    'horizons': [.25, .5, .75],
    'batch_size': 128,
    'learning_rate': 0.01,
    'epochs': 100,
    'hidden_size': 32,
    'dropout': 0.1,
    # event_0: Breast Cancer 
    # event_1: Heart Disease
    'event_to_censor': 'event_0'
})
config_dic = {
    'metabric': config_metabric,
    'support': config_support,
    'seer': config_seer
}

for dataset_name in datasets:

    if type(dataset_name) == tuple:
        dataset_name, event_to_censor = dataset_name
        config_seer.event_to_censor = event_to_censor
        event_to_keep = '0' if config_seer.event_to_censor == 'event_1' else '1'
        config_seer.event_to_keep = 'event_' + event_to_keep
        censor_event = True
    else:
        censor_event = False


    config = config_dic[dataset_name]
    config.model = 'DeepSurv'

    try:
        event_name = '-' + config.event_to_keep
    except AttributeError:
        event_name = ''

    print(f'Running {config.model}{event_name} on {dataset_name}')

    # store each run in list
    runs_list = []

    for i in range(num_runs):

        # load data
        data = Data(config, censor_event)

        # define neural network
        config.out_feature = 1   # need to overwrite value set in load_data
        net = simple_dln(config)

        # initialize model
        model = CoxPH(net, tt.optim.Adam)
        model.optimizer.set_lr(config.learning_rate)
        callbacks = [tt.callbacks.EarlyStopping(patience=20)]

        # train model
        train_time_start = time.time()
        log = model.fit(data.x_train, data.y_train, config.batch_size, config.epochs, callbacks, verbose=True, val_data=data.val_data, val_batch_size=config.batch_size)
        train_time_finish = time.time()

        # calcuate metrics
        evaluator = EvaluatorSingle(data, model, config, offset=0)
        run = evaluator.eval()
        run = update_run(run, train_time_start, train_time_finish, log.epoch)

        runs_list.append(run)

    export_results(runs_list, config)

