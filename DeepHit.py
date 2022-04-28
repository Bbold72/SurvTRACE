import time


from baselines.data_class import Data
from baselines.evaluator import EvaluatorSingle, EvaluatorCompeting
from baselines.utils import export_results, update_run
from baselines import configurations
from baselines.models import DeepHitSingleEvent, DeepHitCompeting


num_runs = 10
datasets = ['metabric', 'support', 'seer']
model_name = 'DeepHit'


for dataset_name in datasets:
    config = getattr(configurations, f'{model_name}_{dataset_name}')
    config.model = model_name
    print(f'Running {config.model} on {dataset_name}')
    print(config)

    # store each run in list
    runs_list = []

    for i in range(num_runs):

        # load data
        data = Data(config)

        # initialize model
        if config.data == 'seer':
            model = DeepHitCompeting(config)
            Evaluator = EvaluatorCompeting
        else:
            model = DeepHitSingleEvent(config)
            Evaluator = EvaluatorSingle

        # train model
        train_time_start = time.time()
        log = model.train(data)
        train_time_finish = time.time()


        # calcuate metrics
        evaluator = Evaluator(data, model.model, config, offset=0)
        run = evaluator.eval()
        run = update_run(run, train_time_start, train_time_finish, log.epoch)

        runs_list.append(run)

    export_results(runs_list, config)

