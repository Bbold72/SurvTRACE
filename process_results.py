# process and format all of the pickle files in '/results'.

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pathlib import Path
import pickle
import os
from typing import Tuple


# TODO: maybe export aggregated results either in this function or within main
def aggregate_raw_data() -> pd.DataFrame:
    '''
    Processes all of the pickle files in '/results' into a dataframe.

    Each file contains the results of a model evaluated on a dataset. 
    It is a list of dictionaries of metrics where each element in the
    list represents a run of the experiment. 

    Metrics:
        - 0.25_ipcw: time-dependent concordance index at 25% quantile
        - 0.50_ipcw: time-dependent concordance index at 50% quantile
        - 0.75_ipcw: time-dependent concordance index at 75% quantile
        - 0.##_ipcw_#: _# indicates which competing event metric belongs to
        - train_time: total training time
        - epochs_trained: number epochs trained for
        - time_per_epoch: training time per epoch (train_time / epochs_trained)

    Naming of file: 
    - w/o censoring:
        '{model name}_{dataset name}.pickle'.
    - w/ censoring:
        '{model name}_{dataset name}_{event_#}.pickle' where # is '0' or '1'.
    
    For each file/experiment, dictionary is converted to a dataframe and the runs
    are aggregated by calculating mean and standard deviation. 
    Each resulting dataframe in concatenated at the end with all other experiments.

    Resulting dataframe structure:
        - index: metric name
        - columns:
            - mean
            - std
            - file_name
            - model
            - dataset
            - event: indicates cause-specific run
            - event_num: event number of cause-specific run

    Args:
        None
    
    Return:
        A dataframe of aggregated results of all experiments.
    '''
    df_list = []
    for file_name in os.listdir(Path('results')):

        with open(Path('results', file_name), 'rb') as f:
            result = pickle.load(f)
        result = pd.DataFrame(result)
        file_name = file_name.split('.')[0]

        agg_df = (result.agg(['mean', 'std'])
                        .transpose()
                        .assign(file_name = lambda x: file_name)
                        )

        agg_df = agg_df.join(agg_df['file_name'].str.split('_', expand=True).rename(
            columns={0:'model', 1:'dataset', 2: 'event', 3: 'event_num'}
        ))

        df_list.append(agg_df)

    df = (pd.concat(df_list)
            .assign(model_name = lambda x: x['model'])
            .replace({'model_name': {'survtrace': 'SurvTRACE',
                                        'survtrace-woMTL': 'SurvTRACE w/o MTL',
                                        'survtrace-woIPS': 'SurvTRACE w/o IPS',
                                        'survtrace-woIPS-woMTL': 'SurvTRACE w/o IPS & MTL',
                                        'PCHazard': 'PC-Hazard'
                                        }
                            })
                        )

    # add cause-specific label
    df['model_name'] = np.where(df['event_num'].isna(),
                            df['model_name'],
                            'CS-' + df['model_name']
                            )
    return df


# TODO: this should export latex file
def format_df(df: pd.DataFrame, is_compare_df: bool=False):
    '''
    Formats metrics and and reshapes dataframe to table format used in paper.

    Rounds mean and standard deviation to three decimal places and combines
    value into one value as a string with standard deviation in parantheses.

    Table format:
        - Each row is a model.
        - Column is a multi-index.
            - Level 1: Dataset name.
            - Level 2: Metric name.

    Args:
        df: (pd.Dataframe): dataframe of results in long format.
        is_compare_df (bool): if comparing our results with the authors, 
            format percents. 
    
    Return:
        # TODO: Export latex file
        prints table of results for paper in latex format
    '''
    df = (df.round({'mean': 3, 
                    'std': 3
                    })
            .assign(metric = lambda x: x['mean'].apply(str) + '(' + x['std'].apply(str) + ')',
                        )
            .drop(columns=['mean', 'std'])
            .sort_values(by=['dataset', 'model_name', 'horizon'])
            ) 
    if is_compare_df:
        df['metric'] = df['metric'].str.replace(r'\(.*\)', '%', regex=True)

    mask = df['horizon'].str.contains('brier', regex=False)
    df = (df[~mask].pivot(index=['model_name'], 
                                        columns=['dataset', 'horizon'], 
                                        values=['metric']
                                        )
                    )       
    print(df.to_latex())


def subset_on_seer(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Split dataframe into two based on 'SEER' dataset or not.

    Args:
        - df (pd.Dataframe): dataframe of results in long format.

    Returns:
        - Tuple of dataframes
            - Tuple[0]: dataframe w/o 'SEER' dataset.
            - Tuple[1]: dataframe w 'SEER' dataset.
    '''
    mask = df['dataset'] == 'seer'
    return df[~mask], df[mask]


def subset_computational_requirements(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Split dataframe into two based on whether metric relates to
        computational requirements or not.

    Args:
        - df (pd.Dataframe): dataframe of results in long format.

    Returns:
        - Tuple of dataframes
            - Tuple[0]: dataframe w/o computational requirements dataset.
            - Tuple[1]: dataframe w/ computational requirements dataset.
    '''
    comp_reqs = ['train_time', 'epochs_trained', 'time_per_epoch']
    mask = df['horizon'].isin(comp_reqs)
    return df[~mask], df[mask]


def label_competing_events_seer(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Label competing events in SEER dataset. Relabel as dataset so
    can reshape on dataset in format_df()

    event_num == 0: Breast Cancer
    event_num == 1: Heart Disease

    Args:
        - df (pd.Dataframe): dataframe of results in long format.

    Returns:
        - df (pd.Dataframe): dataframe of results in long format with labeling.
    '''
    df['dataset'] = np.where(df['event_num'] == '0',
                                'Breast Cancer',
                                np.where(df['event_num'] == '1',
                                            'Heart Disease',
                                            df['dataset']
                                            )
                                        )
    return df


def order_comp_horizon(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Order computational requirements for table.

    Args:
        - df (pd.Dataframe): dataframe of results in long format.

    Returns:
        - df (pd.Dataframe): dataframe of results in long format with horizons ordered.
    '''
    cat = CategoricalDtype(categories=["train_time", "epochs_trained", "time_per_epoch"], ordered=True)
    df['horizon'] = df['horizon'].astype(cat)
    return df


def compare_author_results(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Compare our results with authors' results.

    Reads in 'author_results.csv' and merges dataframe of our results 
    on authors' results, and thencalcuate percent difference of 
    our results with authors' results

    Args:
        - df (pd.Dataframe): dataframe of results in long format.
    
    Returns:
        - df (pd.Dataframe): dataframe of comarision in long format.
    '''
    df_authors = pd.read_csv('author_results.csv')
    df['event_num'] = df['event_num'].apply(float)

    # merge our results on authors' results
    #   {col name}_x: our result
    #   {col name}_y: authors' results
    compare_df = pd.merge(df, df_authors, on=['horizon', 'model', 'dataset', 'event_num'])

    calc_pch = lambda df, var: ((df[var + '_x'] / df[var + '_y']) - 1) * 100
    compare_df['mean'] = calc_pch(compare_df, 'mean')
    compare_df['std'] = calc_pch(compare_df, 'std')

    compare_df = compare_df.drop(columns=['mean_x', 'std_x', 'mean_y', 'std_y'])

    return compare_df 

def main():
    agg_df = (aggregate_raw_data().reset_index()
                            .rename(columns = {'index': 'horizon'})
                            .drop(columns=['event', 'file_name'])
                            )

    # find which event for models that can handel competing events
    #   their metric will have an event number appended to it
    agg_df['event_index'] = agg_df['horizon'].str.slice(-1)
    agg_df['event_num'] = np.where((agg_df['event_index'] == '0') | (agg_df['event_index'] == '1'),
                                            agg_df['event_index'],
                                            agg_df['event_num']
                                            )
    agg_df = agg_df.drop(columns=['event_index'])
    agg_df['horizon'] = agg_df['horizon'].str.replace(r'_\d', '', regex=True)


    # separate dataframe into single and competing events
    df_single, df_competing = subset_on_seer(agg_df)

    # remove compuational requirements from metrics
    df_single, df_single_comp = subset_computational_requirements(df_single)
    df_competing, df_competing_comp = subset_computational_requirements(df_competing)
    
    # label competing events
    df_competing = label_competing_events_seer(df_competing)


    # compare our results with authors results
    compare_df = compare_author_results(agg_df)
    compare_df_single, compare_df_competing = subset_on_seer(compare_df)  
    compare_df_competing['event_num'] = compare_df_competing['event_num'].apply(int).apply(str)
    compare_df_competing = label_competing_events_seer(compare_df_competing)


    # format computational requirements
    df_competing_comp_censoring, df_competing_comp = subset_on_seer(label_competing_events_seer(df_competing_comp))
    order_comps_df_list = list(map(order_comp_horizon, [df_single_comp, df_competing_comp, df_competing_comp_censoring]))


    # format all tables
    list(map(format_df, [df_single, compare_df_single, order_comps_df_list[0], df_competing, compare_df_competing, *order_comps_df_list[1:]],
                        [False, True, False, False, True, False, False]))



    # some rando stats
    N = len(compare_df)

    def within_percent(percent):
        num_within_percent = sum(compare_df['mean'].abs() <= percent)
        print(f"Percent of Results within {percent}% of Authors':", round((num_within_percent/N)*100, 0))
    
    list(map(within_percent, [5, 3, 1]))

if __name__ == '__main__':
    main()


