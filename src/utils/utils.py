from easydict import EasyDict
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import List, Dict

def export_results(results_list: List[Dict], config: EasyDict):
    """
    Exports results as a pickle file.

    Naming of file:
        - w/o censoring:
            '{model name}_{dataset name}.pickle'.
        - w/ censoring:
            '{model name}_{dataset name}_{event_#}.pickle' where # is '0' or '1'.

    Args:
        results_list (list of dicts): list of dictionary of metrics
        config (EasyDict): configuration file.

    Returns:
        Does not return anything.
        Outputs a pickle file with a list of dictionaries where each element of the list is the results
        of a run and each run contains a dictionary of metrics, total training time, number of
        epochs trained for, and time per epoch.
    """
    if config.data == 'seer' and config.model in ['PCHazard', 'CPH', 'RSF', 'DeepSurv']:
        event_name = '_' + config.event_to_keep
    else:
        event_name = ''

    file_name = config['model'] + '_' + config['data'] + f'{event_name}.pickle'
    with open(Path('results', file_name), 'wb') as f:
        pickle.dump(results_list, f)


def update_run(run_dict:dict, train_time_start: float, train_time_finish: float, epochs_trained: int):
    """
    Adds computation stats of run to metrics dictionary.

    Adds the following computation stats:
        - total training time
        - total epochs trained
        - time per epoch

    Args:
        run_dict (dict): dictionary of metrics returned from evaluator class
        train_time_start (float): time at start of training
        train_time_finish (float): time at end of training
        epochs_trained (int): number of epochs trained

    Returns:
        A dictionary of results with a dictionary of metrics, total training time, number of
        epochs trained for, and time per epoch.
    """
    run_dict['train_time'] = train_time_finish - train_time_start
    run_dict['epochs_trained'] = epochs_trained
    run_dict['time_per_epoch'] =  run_dict['train_time'] / run_dict['epochs_trained']
    return run_dict


def df_to_event_time_array(df: pd.DataFrame, event_var_name='event'):
    """
    Creates array of tuples.

    Creates a structured array with two fields:
        1) binary class event indicator.
        2) the time of the event/censoring.
    This format is used in scikit-survival.

    Args:
        df (pandas dataframe): data to subset duration and event from
        event_var_name (str): name of event variable in df
            Default is 'event'

    Returns:
        event_time_array (np.arrray): e.g. [(True, 1), (False, 0)].
    """
    durations, events = (df['duration'].values, df[event_var_name].values)

    event_time_array = np.array([(events[i], durations[i]) for i in range(len(events))],
                    dtype = [('e', bool), ('t', float)])
    return event_time_array