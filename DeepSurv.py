import time

import torchtuples as tt # Some useful functions


from baselines.data_class import Data
from baselines.models import simple_dln
from baselines.evaluator import EvaluatorSingle
from baselines.utils import export_results, update_run
from baselines import configurations
from baselines.models import DeepSurv


num_runs = 10
# event_0: Breast Cancer 
# event_1: Heart Disease
datasets = ['metabric', 'support', ('seer', 'event_0'), ('seer', 'event_1')]
# datasets = ['metabric', 'support']
# datasets = [('seer', 'event_0'), ('seer', 'event_1')]
# datasets = [('seer', 'event_1')]
model_name = 'DeepSurv'


for dataset_name in datasets:
    if type(dataset_name) == tuple:
        dataset_name, event_to_censor = dataset_name
        censor_event = True
    else:
        censor_event = False

    config = getattr(configurations, f'{model_name}_{dataset_name}')
    config.model = model_name
    print(f'Running {config.model} on {dataset_name}')
    print(config)

    if censor_event:
        config.event_to_censor = event_to_censor
        event_to_keep = '0' if config.event_to_censor == 'event_1' else '1'
        config.event_to_keep = 'event_' + event_to_keep

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
        model = DeepSurv(config)

        # train model
        train_time_start = time.time()
        log = model.train(data)
        train_time_finish = time.time()

        # calcuate metrics
        evaluator = EvaluatorSingle(data, model.model, config, offset=0)
        run = evaluator.eval()
        run = update_run(run, train_time_start, train_time_finish, log.epoch)

        runs_list.append(run)

    export_results(runs_list, config)

