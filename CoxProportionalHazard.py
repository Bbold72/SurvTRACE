from easydict import EasyDict
import time


from sksurv.linear_model import CoxPHSurvivalAnalysis

from baselines.data_class import Data
from baselines.evaluator import EvaluatorCPH
from baselines.utils import export_results, update_run, df_to_event_time_array


num_runs = 10
datasets = ['metabric', 'support', ('seer', 'event_0'), ('seer', 'event_1')]
# datasets = ['metabric', 'support']
# datasets = [('seer', 'event_0'), ('seer', 'event_1')]
# datasets = [('seer', 'event_1')]


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
config_seer = EasyDict({
    'data': 'seer',
    'horizons': [.25, .5, .75],
    'epochs': 200,
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
    config.model = 'CPH'

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

        # initialize model
        CPH = CoxPHSurvivalAnalysis(n_iter=config.epochs, verbose=1)

        # train model
        train_time_start = time.time()
        model = CPH.fit(data.x_train, data.y_et_train)
        train_time_finish = time.time()        

        # calcuate metrics
        evaluator = EvaluatorCPH(data, model, config)
        run = evaluator.eval()
        run = update_run(run, train_time_start, train_time_finish, config.epochs)
        runs_list.append(run)

    export_results(runs_list, config)

