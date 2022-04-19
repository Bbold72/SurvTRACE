from easydict import EasyDict
import time


from sksurv.linear_model import CoxPHSurvivalAnalysis

from baselines.data_class import Data
from baselines.evaluator import EvaluatorSingle
from baselines.utils import export_results, update_run, df_to_event_time_array


import numpy as np
from sksurv.metrics import concordance_index_ipcw
from collections import defaultdict

num_runs = 1
# datasets = ['metabric', 'support']
datasets = ['support']


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
        # evaluator = EvaluatorSingle(data, model, config, offset=0)
        # run = evaluator.eval()
    #     run = update_run(run, train_time_start, train_time_finish, log.epoch)
        def evaluator(df, train_index, model, test_set, config, val_batch_size=None):
            df_train_all = df_train_all = df.loc[train_index]
            get_target = lambda df: (df['duration'].values, df['event'].values)
            durations_train, events_train = get_target(df_train_all)
            print('durations_train', durations_train)
            et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                            dtype = [('e', bool), ('t', float)])
            print('et_train', et_train)
            times = config['duration_index'][1:-1]
            print('times', times)
            horizons = config['horizons']

            df_test, df_y_test = test_set
            surv = model.predict_survival_function(df_test)
            # surv = model.predict_cumulative_hazard_function(df_test)
            surv = np.array([f.y for f in surv])
            risk = 1 - surv
            print('risk', risk)
            print(risk.shape)
            
            durations_test, events_test = get_target(df_y_test)
            print('durations_test', durations_test)
            print('events_test', events_test)
            et_test = np.array([(events_test[i], durations_test[i]) for i in range(len(events_test))],
                        dtype = [('e', bool), ('t', float)])
            print('et_test', et_test)
            metric_dict = defaultdict(list)
            cis = []
            for i, _ in enumerate(times):
                # print('iteration', i)
                # print('risk', risk[:, i])
                # print(times)
                cis.append(
                    concordance_index_ipcw(et_train, et_test, estimate=risk[:, i], tau=times[i])[0]
                    )
                metric_dict[f'{horizons[i]}_ipcw'] = cis[i]


            for horizon in enumerate(horizons):
                print(f"For {horizon[1]} quantile,")
                print("TD Concordance Index - IPCW:", cis[horizon[0]])
            
            return metric_dict

        evaluator(data.df, data.df_train.index, model, (data.x_test, data.df_y_test), config=config)
    #     runs_list.append(run)

    # export_results(runs_list, config)

