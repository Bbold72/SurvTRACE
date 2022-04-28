import numpy as np
import torchtuples as tt # Some useful functions

from sksurv.linear_model import CoxPHSurvivalAnalysis
from pycox.models import PCHazard as PCH
from sksurv.ensemble import RandomSurvivalForest

from baselines.dlns import simple_dln



class CPH:

    def __init__(self, config):

        self.model = CoxPHSurvivalAnalysis(n_iter=config.epochs, verbose=1)

    def train(self, data):
        self.model.fit(data.x_train, data.y_et_train)
  


class PCHazard:

    def __init__(self, config):

        # define neural network
        net = simple_dln(config)

        # initalize model
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.model = PCH(net, tt.optim.Adam, duration_index=np.array(config['duration_index'], dtype='float32'))
        self.model.optimizer.set_lr(config.learning_rate)
        self.callbacks = [tt.callbacks.EarlyStopping(patience=20)]


    def train(self, data):
        log = self.model.fit(data.x_train, 
                            data.y_train,
                            self.batch_size, 
                            self.epochs, 
                            self.callbacks, 
                            val_data=data.val_data
                            )
        return log

class RSF:

    def __init__(self, config):

        self.model = RandomSurvivalForest(n_estimators=config.epochs, 
                                            verbose=1,
                                            max_depth=4,
                                            n_jobs=-1
                                            )

    def train(self, data):
        self.model.fit(data.x_train, data.y_et_train)