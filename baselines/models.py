import abc
import numpy as np
import torchtuples as tt # Some useful functions

from sksurv.linear_model import CoxPHSurvivalAnalysis
from pycox.models import CoxPH, DeepHitSingle, DeepHit
from pycox.models import PCHazard as PCH
from sksurv.ensemble import RandomSurvivalForest

from baselines.dlns import simple_dln, CauseSpecificNet

# TODO: add Deep Survival Machines to models

class BaseModel:

    def __init__(self):
        self.epochs_trained = 0
        self.model = None

    @abc.abstractclassmethod
    def train(self, data):
        pass


class BasePycox(BaseModel):
    
    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.callbacks = [tt.callbacks.EarlyStopping(patience=20)]

    def train(self, data):
        self.log = self.model.fit(data.x_train, 
                            data.y_train,
                            self.batch_size, 
                            self.epochs, 
                            self.callbacks, 
                            val_data=data.val_data
                            )
        self.epochs_trained = self.log.epoch


class BaseSksurv(BaseModel):
    def __init__(self, config):
        super().__init__()

        # TODO: there doesn't seem to be a good way to get out how many epochs actually ran
        self.epochs_trained = config.epochs

    def train(self, data):
        self.model.fit(data.x_train, data.y_et_train) 


class CPH(BaseSksurv):

    def __init__(self, config):
        super().__init__(config)
        self.model = CoxPHSurvivalAnalysis(n_iter=config.epochs, verbose=1)

  
class DeepHitCompeting(BasePycox):

    def __init__(self, config):
        super().__init__(config)
        net = CauseSpecificNet(config)
        optimizer = tt.optim.AdamWR(lr=0.01, 
                                        decoupled_weight_decay=0.01,
                                        cycle_eta_multiplier=0.8
                                        )
        # initialize model
        self.model = DeepHit(net, 
                        optimizer, 
                        alpha=0.2, 
                        sigma=0.1,
                        duration_index=config.duration_index
                        )
            
    
class DeepHitSingleEvent(BasePycox):

    def __init__(self, config):
        super().__init__(config)
        net = simple_dln(config)

        # initialize model
        self.model = DeepHitSingle(net, tt.optim.Adam, 
                                alpha=0.2, 
                                sigma=0.1, 
                                duration_index=np.array(config['duration_index'],
                                dtype='float32')
                                )
        self.model.optimizer.set_lr(config.learning_rate)


class DeepSurv(BasePycox):
    def __init__(self, config):
        super().__init__(config)
        config.out_feature = 1   # need to overwrite value set in load_data
        net = simple_dln(config)

        # initialize model
        self.model = CoxPH(net, tt.optim.Adam)
        self.model.optimizer.set_lr(config.learning_rate)

    # overwrite train method
    def train(self, data):
        self.log = self.model.fit(data.x_train, data.y_train, self.batch_size, self.epochs, self.callbacks, verbose=True, val_data=data.val_data)
        self.epochs_trained = self.log.epoch


class PCHazard(BasePycox):

    def __init__(self, config):
        super().__init__(config)

        # define neural network
        net = simple_dln(config)

        # initalize model
        self.model = PCH(net, tt.optim.Adam, duration_index=np.array(config['duration_index'], dtype='float32'))
        self.model.optimizer.set_lr(config.learning_rate)



class RSF(BaseSksurv):

    def __init__(self, config):
        super().__init__(config)
        self.model = RandomSurvivalForest(n_estimators=config.epochs, 
                                            verbose=1,
                                            max_depth=4,
                                            n_jobs=-1
                                            )
