import os
import click
import pickle
import logging
from pathlib import Path
from easydict import EasyDict
from src.models.survtrace.dataset import load_data

logger = logging.getLogger(__name__)

# duration quantiles
HORIZONS = [.25, .5, .75]
DATASETS = ['metabric', 'support', 'seer']
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data','processed')


@click.command()
@click.option('--num_runs', default=10, type=int, help='Number of runs')
def make_data_run(num_runs):
    '''
    Clean datafile and split into training, validation, and test sets.
    '''
    for dataset in DATASETS:
        logger.info(f'Creating {dataset} data for {num_runs} runs')

        # define data conifg to pass to load_data
        data_config = EasyDict({
            'data': dataset,
            'horizons': HORIZONS
        })

        # make a directory for each dataset
        dataset_dir = os.path.join(DATA_DIR, dataset)
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)

        for i in range(num_runs):
            df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(data_config)

            # export data for run as pickle file
            with open(Path(dataset_dir, f'run_{i}.pickle'), 'wb') as f:
                pickle.dump((df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val, data_config), f)

def main():
    make_data_run()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
