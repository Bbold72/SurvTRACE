from easydict import EasyDict
import time


from sksurv.linear_model import CoxPHSurvivalAnalysis

from baselines.data_class import Data
from baselines.evaluator import EvaluatorCPH
from baselines.utils import export_results, update_run, df_to_event_time_array


import numpy as np
from sksurv.metrics import concordance_index_ipcw
from collections import defaultdict

num_runs = 1
datasets = ['metabric', 'support']
# datasets = ['support']


# define the setup parameters
config_metabric = EasyDict({
    'data': 'metabric',
    'horizons': [.25, .5, .75],
    'epochs': 200
})
config_support = EasyDict({
    'data': 'support',
    'horizons': [.25, .5, .75],
    'epochs': 200
})

for dataset_name in datasets:
    config = config_metabric if dataset_name == 'metabric' else config_support
    config.model = 'CPH'
    print(f'Running {config.model} on {dataset_name}')


    # store each run in list
    runs_list = []

    for i in range(num_runs):

        # load data
        data = Data(config)
        y_et_train = df_to_event_time_array(data.df_y_train)
        
        # initialize model
        CPH = CoxPHSurvivalAnalysis(n_iter=config.epochs)

        # train model
        train_time_start = time.time()
        model = CPH.fit(data.x_train, y_et_train)
        train_time_finish = time.time()        

        # calcuate metrics
        evaluator = EvaluatorCPH(data, model, config)
        run = evaluator.eval()
    #     run = update_run(run, train_time_start, train_time_finish, log.epoch)
    #     runs_list.append(run)

    # export_results(runs_list, config)

