import numpy as np
from pathlib import Path
import pickle

def export_results(results_list, config):

    if config.data == 'seer' and config.model in ['PCHazard', 'CPH', 'RSF']:
        event_name = '_' + config.event_to_keep
    else:
        event_name = ''

    file_name = config['model'] + '_' + config['data'] + f'{event_name}.pickle'
    with open(Path('results', file_name), 'wb') as f:
        pickle.dump(results_list, f)

def update_run(run_dict, train_time_start, train_time_finish, epochs_trained):
    run_dict['train_time'] = train_time_finish - train_time_start
    run_dict['epochs_trained'] = epochs_trained
    run_dict['time_per_epoch'] =  run_dict['train_time'] / run_dict['epochs_trained']
    return run_dict


def df_to_event_time_array(df, event_var_name='event'):
    """
    creates a structured array with two fields:
        1) binary class event indicator 
        2) the time of the event/censoring
    This format is used in scikit-survival
    
    Input:
        df (dataframe): data to subset duration and event from
        event_var_name (str): name of event variable in df

    Returns:
        event_time_array (np.arrray): [(True, 1), (False, 0)]
    """
    durations, events = (df['duration'].values, df[event_var_name].values)

    event_time_array = np.array([(events[i], durations[i]) for i in range(len(events))],
                    dtype = [('e', bool), ('t', float)])
    return event_time_array