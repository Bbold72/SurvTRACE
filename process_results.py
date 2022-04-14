import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pathlib import Path
import pickle
import os

from torch import double, float64


def aggregate_raw_data():
    df_list = []
    for file_name in os.listdir(Path('results')):

        with open(Path('results', file_name), 'rb') as f:
            result = pickle.load(f)
            
        result = pd.DataFrame(result)

        file_name = file_name.split('.')[0]
        print(file_name)
        agg_df = (result.agg(['mean', 'std'])
                        .transpose()
                        .assign(file_name = lambda x: file_name)

                        )
        agg_df = agg_df.join(agg_df['file_name'].str.split('_', expand=True).rename(
            columns={0:'model', 1:'dataset', 2: 'event', 3: 'event_num'}
        ))
        if file_name == 'PCHazard_seer_event_0':
            print(agg_df)

        df_list.append(agg_df)

    df = pd.concat(df_list)
    return df


def format_df(df, is_compare_df=False):

    df = (df.round({'mean': 3, 
                                    'std': 3
                                    })
                            .assign(metric = lambda x: x['mean'].apply(str) + '(' + x['std'].apply(str) + ')',
                                     )
                            .drop(columns=['mean', 'std'])
                            .sort_values(by=['dataset', 'model', 'horizon'])
                            ) 
    if is_compare_df:
        df['metric'] = df['metric'].str.replace(r'\(.*\)', '%', regex=True)

    mask = df['horizon'].str.contains('brier', regex=False)
    df = (df[~mask].replace({'model': {'DeepHitCompeting': 'DeepHit',
                                                        'survtrace': 'SurvTRACE',
                                                        'PCHazard': 'PC-Hazard',
                                                        'DeepHitSingle': 'DeepHit'
                                                        }
                                            }
                                        ).pivot(index=['model'], 
                                                columns=['dataset', 'horizon'], 
                                                values=['metric']
                                                )
                    )       
    print(df.to_latex())


def subset_datasets(df):
    mask = df['dataset'] == 'seer'
    return df[~mask], df[mask]

def subset_computational_requirements(df):
    comp_reqs = ['train_time', 'epochs_trained', 'time_per_epoch']
    mask = df['horizon'].isin(comp_reqs)
    return df[~mask], df[mask]

def competing_dataset(df):
    df['dataset'] = np.where(df['event_num'] == '0',
                                'Breast Cancer',
                                np.where(df['event_num'] == '1',
                                            'Heart Disease',
                                            df['dataset']
                                            )
                                        )
    return df


def compare_author_results(df):
    df_authors = pd.read_csv('author_results.csv')
    df['event_num'] = df['event_num'].apply(float)
    compare_df = pd.merge(df, df_authors, on=['horizon', 'model', 'dataset', 'event_num'])

    calc_pch = lambda df, var: ((df[var + '_x'] / df[var + '_y']) - 1) * 100
    compare_df['mean'] = calc_pch(compare_df, 'mean')
    compare_df['std'] = calc_pch(compare_df, 'std')

    compare_df = compare_df.drop(columns=['mean_x', 'std_x', 'mean_y', 'std_y'])

    return compare_df 

def main():
    df = (aggregate_raw_data().reset_index()
                            .rename(columns = {'index': 'horizon'})
                            .drop(columns=['event', 'file_name'])
                            )

    # find which event for CS-PC Hazard
    df['event_index'] = df['horizon'].str.slice(-1)
    df['event_num'] = np.where((df['event_index'] == '0') | (df['event_index'] == '1'),
                                            df['event_index'],
                                            df['event_num']
                                            )
    df['horizon'] = df['horizon'].str.replace(r'_\d', '', regex=True)


    # separate dataframe into single and competing events
    df_single, df_competing = subset_datasets(df)

    df_single, df_single_comp = subset_computational_requirements(df_single)
    df_competing, df_competing_comp = subset_computational_requirements(df_competing)


    format_df(df_single)

    df_competing = competing_dataset(df_competing)
    format_df(df_competing)


    compare_df = compare_author_results(df)
    compare_df_single, compare_df_competing = subset_datasets(compare_df)
    format_df(compare_df_single, True)


    compare_df_competing['event_num'] = compare_df_competing['event_num'].apply(int).apply(str)
    compare_df_competing = competing_dataset(compare_df_competing)
    format_df(compare_df_competing, True)


    def order_comp_horizon(df):
        cat = CategoricalDtype(categories=["train_time", "epochs_trained", "time_per_epoch"], ordered=True)
        df['horizon'] = df['horizon'].astype(cat)
        return df

    format_df(order_comp_horizon(df_single_comp))


    df_competing_comp_censoring, df_competing_comp = subset_datasets(competing_dataset(df_competing_comp))
    
    format_df(df_competing_comp)
    format_df(df_competing_comp_censoring)




    # some rando stats
    N = len(compare_df)

    def within_percent(percent):
        num_within_percent = sum(compare_df['mean'].abs() <= percent)
        print(f"Percent of Results within {percent}% of Authors':", round((num_within_percent/N)*100, 0))
    
    within_percent(3)
    within_percent(1)
    

if __name__ == '__main__':
    main()


