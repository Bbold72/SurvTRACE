from pathlib import Path
import pickle
import time

from survtrace.dataset import load_data
from survtrace.evaluate_utils import Evaluator
from survtrace.utils import set_random_seed
from survtrace.model import SurvTraceSingle
from survtrace.model import SurvTraceMulti
from survtrace.train_utils import Trainer
from survtrace.config import STConfig

from survtrace.losses import NLLLogistiHazardLoss, NLLPCHazardLoss
from torch.nn import BCELoss, MSELoss

from baselines.utils import update_run


num_runs = 10
datasets = ['metabric', 'support', 'seer']
datasets = ['metabric', 'support']
datasets = ['seer']


data_hyperparams = {
            'metabric': {
                'batch_size': 64,
                'weight_decay': 1e-4,
                'learning_rate': 1e-3,
                'epochs': 100,
                'variants': ['woMTL', ''],
                },
            'support': {
                'batch_size': 128,
                'weight_decay': 0,
                'learning_rate': 1e-3,
                'epochs': 100,
                'variants': ['woMTL', ''],
                },
            'seer': {
                'batch_size': 1024,
                'weight_decay': 0,
                'learning_rate': 1e-4,
                'epochs': 100,
                # 'variants': ['woIPS-woMTL', 'woMTL', 'woIPS', ''],
                'variants': ['woIPS'],
                }
            }

for dataset_name in datasets:
    hparams = data_hyperparams[dataset_name]

    for variant in hparams['variants']:
        if variant != '':
            variant_name = '-' + variant
        else:
            variant_name = variant
        print(f'Running SurvTrace{variant_name} on {dataset_name}')

        # define the setup parameters
        STConfig['data'] = dataset_name
        if dataset_name == 'seer':
            STConfig['num_hidden_layers'] = 2
            STConfig['hidden_size'] = 16
            STConfig['intermediate_size'] = 64
            STConfig['num_attention_heads'] = 2
            STConfig['initializer_range'] = .02
            STConfig['early_stop_patience'] = 5

            hparams['val_batch_size'] = 10000


        # store each run in list
        runs_list = []
        for i in range(num_runs):
            print(f'Run {i+1}/{num_runs}')

            # load data - also splits data
            df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(STConfig)

            # get model
            set_random_seed(STConfig['seed'])
            model = SurvTraceMulti(STConfig) if dataset_name == 'seer' else SurvTraceSingle(STConfig)

            # initialize a trainer
            if variant == 'woIPS-woMTL':
                trainer = Trainer(model, metrics=[NLLLogistiHazardLoss(),])
            elif variant == 'woMTL':
                trainer = Trainer(model, metrics=[NLLPCHazardLoss()])
            elif variant == 'woIPS':
                trainer = Trainer(model, metrics=[NLLLogistiHazardLoss(), BCELoss(), MSELoss()])
            # complete survtrace
            else:
                trainer = Trainer(model, metrics=[NLLPCHazardLoss(), BCELoss(), MSELoss()])

            train_time_start = time.time()
            train_loss, val_loss, last_epoch = trainer.fit((df_train, df_y_train), (df_val, df_y_val), **hparams,)
            train_time_finish = time.time()

            # evaluate model
            evaluator = Evaluator(df, df_train.index)
            run = evaluator.eval(model, (df_test, df_y_test))
            run['train_time'] = train_time_finish - train_time_start
            run = update_run(run, train_time_start, train_time_finish, last_epoch)
            runs_list.append(run)

        file_name = f'survtrace{variant_name}_{STConfig["data"]}.pickle'
        with open(Path('results', file_name), 'wb') as f:
            pickle.dump(runs_list, f)
    
