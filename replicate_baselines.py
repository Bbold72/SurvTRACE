import time

from baselines.data_class import Data
from baselines.evaluator import EvaluatorCPH, EvaluatorSingle
from baselines.utils import export_results, update_run
from baselines import configurations
from baselines.models import CPH, PCHazard






def run_experiment(dataset_name, model_name, num_runs=10, event_to_censor=None):

    censor_event = True if event_to_censor else False
    print(censor_event)
    
    # intialize configuration
    config = getattr(configurations, f'{model_name}_{dataset_name}')
    config.model = model_name
    print(config)


    # add event to censor and to keep into config
    if censor_event:
        config.event_to_censor = event_to_censor
        event_to_keep = '0' if config.event_to_censor == 'event_1' else '1'
        config.event_to_keep = 'event_' + event_to_keep
    
    try:
        event_name = '-' + config.event_to_keep
    except AttributeError:
        event_name = ''
    config.epochs=1
    print(f'Running {config.model}{event_name} on {dataset_name}')


    # get corresponding model and evaluator
    if config.model == 'CPH':
        Model = CPH
        Evaluator = EvaluatorCPH
    elif config.model == 'PCHazard':
        Model = PCHazard
        Evaluator = EvaluatorSingle
    else:
        raise('Wrong model name provided')

    # store each run in list
    runs_list = []
    for i in range(num_runs):

        # load data
        data = Data(config, censor_event)

        # initalize model
        m = Model(config)

        # train model
        train_time_start = time.time()
        m.train(data)
        train_time_finish = time.time()

        # calcuate metrics
        evaluator = Evaluator(data, m.model, config)
        run = evaluator.eval()
        print(m.epochs_trained)
        run = update_run(run, train_time_start, train_time_finish, m.epochs_trained)

        runs_list.append(run)

    export_results(runs_list, config)


def main():
    # models that require an event to be censored for competing events
    cause_specific_models = set(['CPH', 'PCHazard'])  
    number_runs = 1

    datasets = ['metabric', 'support', 'seer']
    # datasets = ['metabric', 'support']
    models = ['CPH', 'PCHazard']
    models = ['PCHazard']

    for dataset_name in datasets:
        for model_name in models:
            if dataset_name == 'seer' and model_name in cause_specific_models:
                run_experiment(dataset_name, model_name, number_runs, event_to_censor='event_0')
                run_experiment(dataset_name, model_name, number_runs, event_to_censor='event_1')
            else:
                run_experiment(dataset_name, model_name, number_runs)


if __name__ == '__main__':
    main()