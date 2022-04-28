import numpy as np
import torchtuples as tt # Some useful functions

from sksurv.linear_model import CoxPHSurvivalAnalysis
from pycox.models import DeepHitSingle, DeepHit
from pycox.models import PCHazard as PCH
from sksurv.ensemble import RandomSurvivalForest

from baselines.dlns import simple_dln, CauseSpecificNet



class CPH:

    def __init__(self, config):

        self.model = CoxPHSurvivalAnalysis(n_iter=config.epochs, verbose=1)

    def train(self, data):
        self.model.fit(data.x_train, data.y_et_train)
  

class DeepHitCompeting:

    def __init__(self, config):

        self.batch_size = config.batch_size
        self.epochs = config.epochs
        net = CauseSpecificNet(config)
        optimizer = tt.optim.AdamWR(lr=0.01, 
                                        decoupled_weight_decay=0.01,
                                        cycle_eta_multiplier=0.8
                                        )
        self.callbacks = [tt.callbacks.EarlyStopping(patience=20)]

        # initialize model
        self.model = DeepHit(net, optimizer, 
                        alpha=0.2, 
                        sigma=0.1,
                        duration_index=config.duration_index
                        )
        
    def train(self, data):
        log = self.model.fit(data.x_train, data.y_train, self.batch_size, self.epochs, self.callbacks, val_data=data.val_data)
        return log
            
    
class DeepHitSingleEvent:

    def __init__(self, config):

        self.batch_size = config.batch_size
        self.epochs = config.epochs
        net = simple_dln(config)
        self.callbacks = [tt.callbacks.EarlyStopping(patience=20)]

        # initialize model
        self.model = DeepHitSingle(net, tt.optim.Adam, 
                                alpha=0.2, 
                                sigma=0.1, 
                                duration_index=np.array(config['duration_index'],
                                dtype='float32')
                                )
        self.model.optimizer.set_lr(config.learning_rate)
        
    def train(self, data):
        log = self.model.fit(data.x_train, data.y_train, self.batch_size, self.epochs, self.callbacks, val_data=data.val_data)
        return log


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