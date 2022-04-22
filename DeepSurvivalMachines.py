from easydict import EasyDict
import time

from auton_survival.models.dsm import DeepSurvivalMachines
from sklearn.model_selection import ParameterGrid

from baselines.data_class import Data
from baselines.evaluator import EvaluatorSingleDSM
from baselines.evaluator import EvaluatorCompetingDSM
from baselines.utils import export_results, update_run

import numpy as np

num_runs = 10
datasets = ['metabric', 'support', 'seer']

# define the setup parameters
config_metabric = EasyDict({
    'data': 'metabric',
    'horizons': [.25, .5, .75],
    'batch_size': 64,
    'epochs': 100,
    'hyperparameters': EasyDict({
        'hidden_size': [[100, 100]],
        'k': [4],
        'distribution': ['Weibull'],
        'learning_rate': [1e-3],
        'discount': [0.5]
        # 'hidden_size': [[50], [50, 50], [100], [100, 100]],
        # 'k': [4, 6, 8],
        # 'distribution': ['LogNormal', 'Weibull'],
        # 'learning_rate': [1e-3, 1e-4],
        # 'discount': [0.5, 0.75, 1]
    })
})
config_support = EasyDict({
    'data': 'support',
    'horizons': [.25, .5, .75],
    'batch_size': 128,
    'epochs': 200,
    'hyperparameters': EasyDict({
        'hidden_size': [[100, 100]],
        'k': [4],
        'distribution': ['Weibull'],
        'learning_rate': [1e-3],
        'discount': [0.5]
        # 'hidden_size': [[50], [50, 50], [100], [100, 100]],
        # 'k': [4, 6, 8],
        # 'distribution': ['LogNormal', 'Weibull'],
        # 'learning_rate': [1e-3, 1e-4],
        # 'discount': [0.5, 0.75, 1]
    })
})
config_seer = EasyDict({
    'data': 'seer',
    'horizons': [.25, .5, .75],
    'batch_size': 1024,
    'epochs': 200,
    'hyperparameters': EasyDict({
        'hidden_size': [[100, 100]],
        'k': [4],
        'distribution': ['Weibull'],
        'learning_rate': [1e-4],
        'discount': [0.5]
        # 'hidden_size': [[50], [50, 50], [100], [100, 100]],
        # 'k': [4, 6, 8],
        # 'distribution': ['LogNormal', 'Weibull'],
        # 'learning_rate': [1e-3, 1e-4],
        # 'discount': [0.5, 0.75, 1]
    })
})
config_dic = {
    'metabric': config_metabric,
    'support': config_support,
    'seer': config_seer
}

for dataset_name in datasets:
    config = config_dic[dataset_name]
    config.model = 'DSM'
    print('Running DSM on ' + dataset_name)


    # select evaluator based on dataset
    Evaluator = EvaluatorCompetingDSM if config.data == 'seer' else EvaluatorSingleDSM
    

    # store each run in list
    runs_list = []

    for i in range(num_runs):

        # load data
        data = Data(config)
        print(np.unique(data.train_outcomes))
        # params = create_parameter_grid(config, ['learning_rate', 'hidden_size', 'k', 'distribution'])
        params = ParameterGrid(dict(config.hyperparameters))

        # train model
        models = []
        n_params = len(params)
        for pi, param in enumerate(params):
            train_time_start = time.time() 
            # initialize model
            print(f'Hyperparameters - {pi+1}/{n_params}:', param)
            model = DeepSurvivalMachines(k=param['k'],
                                    distribution=param['distribution'],
                                    layers=param['hidden_size']
                                    )
            model.fit(data.x_train, data.train_times, data.train_outcomes, 
                        val_data=(data.x_val, data.val_times, data.val_outcomes), 
                        iters=config.epochs, 
                        learning_rate=param['learning_rate'],
                        batch_size=config.batch_size
                        )
            train_time_finish = time.time()
            evaluator = Evaluator(data, model, config, test_set=False)
            run = evaluator.eval()
            mean_td = sum(x for x in run.values())/len(run)
            val_loss = model.compute_nll(data.x_val, data.val_times, data.val_outcomes)
            models.append([mean_td, model, train_time_start, train_time_finish, param])
            print('Loss:', val_loss)
            print(run)
            print('Average TD Concordance Index:', mean_td)
        best_model = max(models, key=lambda x: x[0])
        print('Best Model Hyperparameters:', best_model[4])

        # calcuate metrics
        evaluator = Evaluator(data, best_model[1], config)
        run = evaluator.eval()
        run = update_run(run, best_model[2], best_model[3], config.epochs)

        runs_list.append(run)

    export_results(runs_list, config)

