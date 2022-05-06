import time
import click
import logging
from typing import Optional

from experiments import configurations
from experiments.data_class import Data
from experiments.evaluator import EvaluatorSingle, EvaluatorCompeting
from experiments.models import CPH, DH, DeepSurv, DSM, PCHazard, RSF, SurvTRACE
from experiments.utils import export_results, update_run

logger = logging.getLogger(__name__)

DATASETS = ['metabric', 'support', 'seer']
# MODELS = ['CPH', 'DeepHit', 'DeepSurv', 'DSM', 'PCHazard', 'RSF', \
#                 'survtrace', 'survtrace-woMTL', 'survtrace-woIPS', 'survtrace-woIPS-woMTL']
#                 'survtrace', 'survtrace-woMTL', 'survtrace-woIPS', 'survtrace-woIPS-woMTL']
MODELS = ['survtrace', 'survtrace-woMTL', 'survtrace-woIPS', 'survtrace-woIPS-woMTL']
CAUSE_SPECIFIC_MODELS = set(['CPH', 'DeepSurv', 'PCHazard', 'RSF'])
MAP_NAME_TO_MODEL = {
    'CPH': CPH,
    'DeepHit': DH,
    'DeepSurv': DeepSurv,
    'DSM': DSM,
    'PCHazard': PCHazard,
    'RSF': RSF,
    'survtrace': SurvTRACE,
    'survtrace-woMTL': SurvTRACE,
    'survtrace-woIPS': SurvTRACE,
    'survtrace-woIPS-woMTL': SurvTRACE,
}

@click.command()
@click.option('--num_runs', default=10, type=int, help='Number of runs')
def run_all_experiments(num_runs):
    for model_name in MODELS:
        for dataset_name in DATASETS:
            if dataset_name == 'seer' and model_name in CAUSE_SPECIFIC_MODELS:
                run_experiment(dataset_name, model_name, num_runs, event_to_censor='event_0')
                run_experiment(dataset_name, model_name, num_runs, event_to_censor='event_1')
            else:
                run_experiment(dataset_name, model_name, num_runs)

def run_experiment(dataset_name: str, model_name: str, num_runs=10, event_to_censor: Optional[str]=None):
    '''
    Runs baseline model experiment from beginning to end.

    Loads, processes, and split data into train, validation, test dataframes.
    These dataframes are stored in baselines.Data class.
    If censoring data, event_to_censor will have its values turned to zero.
    Additional post-processing to transform data into correct format for each model.
    A model from baselines.models is instantiated and trained, and then a trained
    model is passed to a baselines.evaluators class to calculate the
    time dependent concordance index. Then results are exported to a pickle file
    and saved in directory 'results/'.

    Models:
        - CPH: Cox Proportional Hazards
        - DeepHit
        - DeepSurv
        - DSM: Deep Survival Machines
        - PCHazard: PC-Hazard
        - RSF: Random Survival Forests
        - survtrace: SurvTRACE with MTL & IPS
        - survtrace-woMTL: SurvTRACE without MTL but with IPS
        - survtrace-woIPS: SurvTRACE with MTL but without IPS
        - survtrace-woIPS-woMTL: SurvTRACE without MTL and without IPS

    Args:
        dataset_name (str): Name of dataset to use. One of ['metabric', 'support', 'seer']
        model_name (str): Name of model/experiment to run.
            One of ['CPH', 'DeepHit', 'DeepSurv', 'DSM', 'PCHazard', 'RSF']
        num_runs (int): Number of times to run experiment.
            Default is 10 since the paper evaluates each model 10 times.
        event_to_censor (str): Name of event to censore when running cause specific analysis on SEER
            One of ['event_0', 'event_1', None]. Default is None.
            Valid for these models SEER: ['CPH', 'DeepSurv', 'PCHazard', 'RSF'].

    Return:
        Does not return a value.
        Outputs results as a pickle file to the directory 'results/'
        structure of results is a list of dictionaries where each element of the list is the results
        of a run and each run contains a dictionary of metrics, total training time, number of
        epochs trained for, and time per epoch.
    '''
    censor_event = True if event_to_censor else False

    # intialize configuration
    if model_name.startswith('survtrace'):
        config_model_name = model_name.split('-')[0]
    else:
        config_model_name = model_name
    config = getattr(configurations, f'{config_model_name}_{dataset_name}')
    config.model = model_name

    # add event to censor and to keep into config
    if censor_event:
        config.event_to_censor = event_to_censor
        event_to_keep = '0' if config.event_to_censor == 'event_1' else '1'
        config.event_to_keep = 'event_' + event_to_keep

    # add event name if censoring
    try:
        event_name = '-' + config.event_to_keep
    except AttributeError:
        event_name = ''
    logger.info(f'Running {config.model}{event_name} on {dataset_name}')

    # config.epochs=1
    model = MAP_NAME_TO_MODEL[config_model_name]

    # get corresponding evaluator
    Evaluator = EvaluatorSingle
    if config_model_name in ['DeepHit', 'DSM', 'survtrace'] and dataset_name == 'seer':
        Evaluator = EvaluatorCompeting

    # store each run in list
    runs_list = []
    for i in range(num_runs):
        logger.info(f'Run {i+1}/{num_runs}')

        # load data
        data = Data(config, dataset=dataset_name, run_num=i, censor_event=censor_event)

        # initalize model
        m = model(config)

        # train model
        train_time_start = time.time()
        m.train(data)
        train_time_finish = time.time()

        # calcuate metrics
        evaluator = Evaluator(data, m, config)
        try:
            run = evaluator.eval()
            run = update_run(run, train_time_start, train_time_finish, m.epochs_trained)
            runs_list.append(run)
        except ValueError as e:
            logger.error(f'ERROR: Could not evaluate {model_name} on {dataset_name} data for run {i}. Skipping.')
            logger.error(e)
    export_results(runs_list, config)

def main():
    run_all_experiments()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()