import numpy as np
from easydict import EasyDict
from collections import defaultdict
from sksurv.metrics import concordance_index_ipcw

import torch # For building the networks 
import torchtuples as tt # Some useful functions

from pycox.models import PCHazard

from survtrace.dataset import load_data

# define the setup parameters
# config = EasyDict({
#     'data': 'metabric',
#     'horizons': [.25, .5, .75],
#     'batch_size': 64,
#     'learning_rate': 0.01,
#     'epochs': 50,
#     'hidden_size': 32
# })
config = EasyDict({
    'data': 'support',
    'horizons': [.25, .5, .75],
    'batch_size': 128,
    'learning_rate': 0.01,
    'epochs': 50,
    'hidden_size': 32
})
# config = EasyDict({
#     'data': 'seer',
#     'horizons': [.25, .5, .75],
#     'batch_size': 1024,
#     'learning_rate': 0.01,
#     'epochs': 100,
#     'hidden_size': 32
# })
# event_0: Heart Disease
# event_1: Breast Cancer
event_to_censor = 'event_0'


# load data
df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(config)

# censor event
if config.data == 'seer':

    censor_event = lambda df: df.drop(event_to_censor, axis=1).rename(columns={event_to_keep: 'event'})

    event_to_keep = '0' if event_to_censor.split('_')[1] == '1' else '1'
    event_to_keep = 'event_' + event_to_keep

    df = censor_event(df)
    df_y_train = censor_event(df_y_train)
    df_y_val = censor_event(df_y_val)
    df_y_test = censor_event(df_y_test)

# convet data to format necessary for pycox
x_train = np.array(df_train, dtype='float32')
x_val = np.array(df_val, dtype='float32')
x_test = np.array(df_test, dtype='float32')

y_df_to_tuple = lambda df: tuple([np.array(df['duration'], dtype='int64'), np.array(df['event'], dtype='float32'), np.array(df['proportion'], dtype='float32')])

y_train = y_df_to_tuple(df_y_train)
y_val = y_df_to_tuple(df_y_val)


# define neural network
hidden_size = config.hidden_size
dropout = 0.1

net = torch.nn.Sequential(
    torch.nn.Linear(config.num_feature, hidden_size),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(hidden_size),
    torch.nn.Dropout(dropout),
    
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.BatchNorm1d(hidden_size),
    torch.nn.Dropout(dropout),
    
    torch.nn.Linear(hidden_size, config.out_feature)
)

# initalize model
model = PCHazard(net, tt.optim.Adam, duration_index=np.array(config['duration_index'], dtype='float32'))
model.optimizer.set_lr(config.learning_rate)
callbacks = [tt.callbacks.EarlyStopping(patience=20)]

# train model
log = model.fit(x_train, y_train, config.batch_size, config.epochs, callbacks, val_data=tuple([x_val, y_val]))

# calcuate metrics
class Evaluator:
    def __init__(self, df, train_index):
        '''the input duration_train should be the raw durations (continuous),
        NOT the discrete index of duration.
        '''
        self.df_train_all = df.loc[train_index]

    def eval_single(self, model, test_set, config, val_batch_size=None):
        df_train_all = self.df_train_all
        get_target = lambda df: (df['duration'].values, df['event'].values)
        durations_train, events_train = get_target(df_train_all)
        print('durations_train', durations_train)
        et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                        dtype = [('e', bool), ('t', float)])
        print('et_train', et_train)
        times = config['duration_index'][1:-1]
        print('times', times)
        horizons = config['horizons']

        df_test, df_y_test = test_set
        surv = model.predict_surv_df(df_test, batch_size=val_batch_size)
        risk = np.array((1 - surv).transpose())
        print('risk', risk)
        
        durations_test, events_test = get_target(df_y_test)
        print('durations_test', durations_test)
        print('events_test', events_test)
        et_test = np.array([(events_test[i], durations_test[i]) for i in range(len(events_test))],
                    dtype = [('e', bool), ('t', float)])
        print('et_test', et_test)
        metric_dict = defaultdict(list)
        cis = []
        for i, _ in enumerate(times):
            print('iteration', i)
            print('risk', risk[:, i+1])
            print(times)
            cis.append(
                concordance_index_ipcw(et_train, et_test, estimate=risk[:, i+1], tau=times[i])[0]
                )
            metric_dict[f'{horizons[i]}_ipcw'] = cis[i]


        for horizon in enumerate(horizons):
            print(f"For {horizon[1]} quantile,")
            print("TD Concordance Index - IPCW:", cis[horizon[0]])
        
        return metric_dict

evaluator = Evaluator(df, df_train.index)
evaluator.eval_single(model, (x_test, df_y_test), config=config, val_batch_size=10000)


