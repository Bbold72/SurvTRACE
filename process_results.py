import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import os


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

        df_list.append(agg_df)

    df = pd.concat(df_list)
    return df


def format_single_events_df(df_single):

    cols_unpivot = ['metric']
    cols_to_drop = ['event', 'event_num', 'file_name', 'mean', 'std']
    df_single = (df_single.round({'mean': 3, 
                                    'std': 3
                                    })
                            .assign(metric = lambda x: x['mean'].apply(str) + '(' + x['std'].apply(str) + ')')
                            .drop(columns=cols_to_drop)
                            .sort_values(by=['dataset', 'model', 'horizon'])
                            ) 

    mask = df_single['horizon'].str.contains('brier', regex=False)
    df_single = df_single[~mask]
    df_single = df_single.pivot(index=['model'], columns=['dataset', 'horizon'], values=['metric'])
        
    print(df_single)
    df_single.to_csv('single.csv')
    print(df_single.to_latex())

def format_competing_events_df(df_competing):
    print(df_competing)

def main():
    df = (aggregate_raw_data().reset_index()
                            .rename(columns = {'index': 'horizon'})
                            )

    # separate dataframe into single and competing events
    mask = df['dataset'] == 'seer'
    df_single = df[~mask]
    df_competing = df[mask]

    format_single_events_df(df_single)


    df_competing['event_index'] = df_competing['horizon'].str.slice(-1)
    df_competing['event_num'] = np.where((df_competing['event_index'] == '0') | (df_competing['event_index'] == '1'),
                                            df_competing['event_index'],
                                            df_competing['event_num']
                                            )
    df_competing['dataset'] = np.where(df_competing['event_num'] == '0',
                                        'Breast Cancer',
                                        np.where(df_competing['event_num'] == '1',
                                            'Heart Disease',
                                            df_competing['dataset']
                                            )
                                        )


    format_competing_events_df(format_single_events_df(df_competing))

if __name__ == '__main__':
    main()


