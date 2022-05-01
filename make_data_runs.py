from survtrace.dataset import load_data
from easydict import EasyDict
import os
from pathlib import Path
import pickle

# duration quantiles
HORIZONS = [.25, .5, .75]
ROOT_DIR = os.path.realpath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data/processed')



def make_data_run(dataset: str, num_runs: int=10):
    '''
    Clean datafile and split into training, validation, and test sets.
    '''
    # define data conifg to pass to load_data
    data_config = EasyDict({
        'data': dataset,
        'horizons': HORIZONS
    })

    # make a directory for each dataset
    DATASET_DIR = os.path.join(DATA_DIR, dataset)
    if not os.path.isdir(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    for i in range(num_runs):
        print(f'Creating {dataset} data for run {i}')

        df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(data_config)
        
        # export data for run as pickle file
        with open(Path(DATASET_DIR, f'run_{i}'), 'wb') as f:
            pickle.dump((df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val, data_config), f)


def main():

    datasets = ['metabric', 'support', 'seer']

    for data in datasets:
        make_data_run(dataset=data,
                        num_runs=10)
 

if __name__ == '__main__':
    main()


