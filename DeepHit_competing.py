from easydict import EasyDict
import time 

import torchtuples as tt # Some useful functions

from pycox.models import DeepHit

from baselines.data_class import Data
from baselines.models import CauseSpecificNet
from baselines.evaluator import EvaluatorCompeting
from baselines.utils import export_results, update_run


config = EasyDict({
    'data': 'seer',
    'model': 'DeepHitCompeting',
    'horizons': [.25, .5, .75],
    'batch_size': 1024,
    'learning_rate': 0.01,
    'epochs': 50,
    'hidden_size_indiv': 32,
    'hidden_size_shared': 64,
    'dropout': 0.1
})

num_runs = 10

# load data
data = Data(config)
print('Running DeepHit on ' + config.data)

# store each run in list
runs_list = []

for i in range(num_runs):

    # initialize model
    net = CauseSpecificNet(config)
    optimizer = tt.optim.AdamWR(lr=0.01, decoupled_weight_decay=0.01,
                                cycle_eta_multiplier=0.8)
    callbacks = [tt.callbacks.EarlyStoppingCycle(patience=10)]

    model = DeepHit(net, optimizer, alpha=0.2, sigma=0.1,
                    duration_index=config.duration_index)

    # train model
    train_time_start = time.time()
    log = model.fit(data.x_train, data.y_train, config.batch_size, config.epochs, callbacks, val_data=tuple([data.x_val, data.y_val]))
    train_time_finish = time.time()


    # evaluator = Evaluator(data.df, data.df_train.index)
    # evaluator.eval_multi(model, (data.x_test, data.df_y_test), config=config, val_batch_size=10000)
    evaluator = EvaluatorCompeting(data, model, config, offset=0)
    run = evaluator.eval()
    run = update_run(run, train_time_start, train_time_finish, log.epoch)

    runs_list.append(run)

export_results(runs_list, config)


