from easydict import EasyDict
from collections import defaultdict
import numpy as np
from sksurv.metrics import concordance_index_ipcw

import torch # For building the networks 
import torchtuples as tt # Some useful functions

from pycox.models import CoxPH

from survtrace.dataset import load_data


# define the setup parameters
config = EasyDict({
    'data': 'metabric',
    'horizons': [.25, .5, .75],
    'batch_size': 64,
    'learning_rate': 0.01,
    'epochs': 50,
    'hidden_size': 32
})
config = EasyDict({
    'data': 'support',
    'horizons': [.25, .5, .75],
    'batch_size': 128,
    'learning_rate': 0.01,
    'epochs': 50,
    'hidden_size': 32
})


# load data
df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val = load_data(config)


# convet data to format necessary for pycox
x_train = np.array(df_train, dtype='float32')
x_val = np.array(df_val, dtype='float32')
x_test = np.array(df_test, dtype='float32')

y_df_to_tuple = lambda df: tuple([np.array(df['duration'], dtype='int64'), np.array(df['event'], dtype='float32')])

y_train = y_df_to_tuple(df_y_train)
y_val = y_df_to_tuple(df_y_val)


# define neural network
in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = 1
batch_norm = True
dropout = 0.1
output_bias = False

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                              dropout, output_bias=output_bias)

model = CoxPH(net, tt.optim.Adam)
model.optimizer.set_lr(0.001)
callbacks = [tt.callbacks.EarlyStopping(patience=20)]

# train model
log = model.fit(x_train, y_train, config.batch_size, config.epochs, callbacks, verbose=True, val_data=tuple([x_val, y_val]), val_batch_size=config.batch_size)


# calcuate metrics
def evaluator(df, train_index, model, test_set, config):
    df_train_all = df_train_all = df.loc[train_index]
    get_target = lambda df: (df['duration'].values, df['event'].values)
    durations_train, events_train = get_target(df_train_all)
    et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                    dtype = [('e', bool), ('t', float)])
    times = config['duration_index'][1:-1]
    horizons = config['horizons']

    df_test, df_y_test = test_set
    _ = model.compute_baseline_hazards()
    surv = model.predict_surv(df_test)
    risk = np.array((1 - surv))
    
    durations_test, events_test = get_target(df_y_test)
    et_test = np.array([(events_test[i], durations_test[i]) for i in range(len(events_test))],
                dtype = [('e', bool), ('t', float)])

    metric_dict = defaultdict(list)
    cis = []
    for i, _ in enumerate(times):
        cis.append(
            concordance_index_ipcw(et_train, et_test, estimate=risk[:, i], tau=times[i])[0]
            )
        metric_dict[f'{horizons[i]}_ipcw'] = cis[i]

    for horizon in enumerate(horizons):
        print(f"For {horizon[1]} quantile,")
        print("TD Concordance Index - IPCW:", cis[horizon[0]])
    
    return metric_dict


evaluator(df, df_train.index, model, (x_test, df_y_test), config=config)


