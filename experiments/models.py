# provides consistent interface to instantiate and train each model used in the experiments

from abc import ABC, abstractclassmethod
from easydict import EasyDict
import numpy as np
import torchtuples as tt 

from auton_survival.models.dsm import DeepSurvivalMachines
from sksurv.linear_model import CoxPHSurvivalAnalysis
from pycox.models import CoxPH, DeepHitSingle, DeepHit
from pycox.models import PCHazard as PCH
from sksurv.ensemble import RandomSurvivalForest

from survtrace.model import SurvTraceSingle, SurvTraceMulti
from survtrace.train_utils import Trainer

from survtrace.losses import NLLLogistiHazardLoss, NLLPCHazardLoss
from torch.nn import BCELoss, MSELoss

from torch.nn import BCELoss, MSELoss, ReLU
from torchtuples.practical import MLPVanilla

from experiments.dlns import simple_dln, simple_dln_sequential, CauseSpecificNet
from experiments.data_class import Data

class BaseModel(ABC):
    '''
    Defines interface for other classes.

    Attributes:
        - epochs_trained (int): number of epochs model trained for
        - model: model from pycox, DSM, sksurv, or SurvTRACE.
        - eval_offset (int): number of columns to skip in model's risk calculation.
    '''
    def __init__(self):
        self.epochs_trained = 0

        # TODO: makes these abstract attributes
        self.model = None
        self.eval_offset = None

    @abstractclassmethod
    def train(self):
        '''
        Trains self.model
        '''
        pass
    

class BasePycox(BaseModel):
    '''
    Defines interface for models from pycox.
    
    Child of BaseModel.

    Attributes:
        - batch_size (int)
        - epochs (int): number of epochs to train for
        - callbacks (List): list of additional functionality to add to each model
            - adds EarlyStopping where model quits training after not improving for 20 epochs
    '''
    def __init__(self, config: EasyDict):
        '''
        Args:
            - config: configuration dictionary from experiments.configurations
        '''
        super().__init__()
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.callbacks = [tt.callbacks.EarlyStopping(patience=20)]


    def train(self, data: Data) -> None:
        '''
        Trains self.model.

        Args: 
            - data: Data class from experiments.data_class
        
        Returns:
            Nothing.
            self.model is trained.
            Updates self.epochs_trained.
        '''
        self.log = self.model.fit(data.x_train, 
                            data.y_train,
                            self.batch_size, 
                            self.epochs, 
                            self.callbacks, 
                            val_data=data.val_data
                            )
        self.epochs_trained = self.log.epoch



class BaseSksurv(BaseModel):
    '''
    Defines interface for models from sksurv.
    
    Child of BaseModel.

    Attributes:
        - epochs (int): number of epochs to train for
    '''
    def __init__(self, config: EasyDict):
        '''
        Args:
            - config: configuration dictionary from experiments.configurations
        '''
        super().__init__()
        self.epochs = config.epochs


    def train(self, data: Data):
        '''
        Trains self.model.

        Args: 
            - data: Data class from experiments.data_class
        
        Returns:
            Nothing.
            self.model is trained.
            Updates self.epochs_trained with total number of epochs from config.
        '''
        self.model.fit(data.x_train, data.y_et_train) 

        # TODO: there doesn't seem to be a good way to get out how many epochs actually ran
        self.epochs_trained = self.epochs


class CPH(BaseSksurv):
    '''
    Cox Proportional Hazards

    Child of BaseSksurv.
    '''
    def __init__(self, config):
        '''
        Args:
            - config: configuration dictionary from experiments.configurations
        '''
        super().__init__(config)
        self.eval_offset = 0
        self.model = CoxPHSurvivalAnalysis(n_iter=config.epochs, verbose=1)

  
class DeepHitCompeting(BasePycox):
    '''
    DeepHit for competing events.

    Child of BasePycox.
    '''
    def __init__(self, config):
        '''
        Args:
            - config: configuration dictionary from experiments.configurations
        '''
        super().__init__(config)
        self.eval_offset = 0
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
    '''
    DeepHit for single events.

    Child of BasePycox.
    '''
    def __init__(self, config):
        '''
        Args:
            - config: configuration dictionary from experiments.configurations
        '''
        super().__init__(config)
        self.eval_offset = 0
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
    '''
    DeepSurv.

    Child of BasePycox.
    '''
    def __init__(self, config):
        '''
        Args:
            - config: configuration dictionary from experiments.configurations
        '''
        super().__init__(config)
        self.eval_offset = 0

        config.out_feature = 1   # need to overwrite value set in load_data
        net = simple_dln(config)

        # initialize model
        self.model = CoxPH(net, tt.optim.Adam)
        self.model.optimizer.set_lr(config.learning_rate)

    # overwrite train method
    def train(self, data):
        self.log = self.model.fit(data.x_train, data.y_train, self.batch_size, self.epochs, self.callbacks, verbose=True, val_data=data.val_data)
        self.epochs_trained = self.log.epoch


class DSM(BaseModel):
    '''
    Deep Survival Machines.

    Child of BaseModel.
    '''
    def __init__(self, config):
        '''
        Args:
            - config: configuration dictionary from experiments.configurations
        '''
        super().__init__()
        self.eval_offset = 0
        self.epochs = config.epochs
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size

        self.model = DeepSurvivalMachines(k=config['k'],
                                    distribution=config['distribution'],
                                    layers=config['hidden_size']
                                    )
    def train(self, data):
        self.model.fit(data.x_train, data.train_times, data.train_outcomes, 
                    val_data=(data.x_val, data.val_times, data.val_outcomes), 
                    iters=self.epochs, 
                    learning_rate=self.learning_rate,
                    batch_size=self.batch_size
                    )
        
        # TODO: find a way to get number of epochs trained
        self.epochs_trained = np.nan
    


class PCHazard(BasePycox):
    '''
    PC-Hazard.

    Child of BasePycox.
    '''
    def __init__(self, config):
        '''
        Args:
            - config: configuration dictionary from experiments.configurations
        '''
        super().__init__(config)
        self.eval_offset = 1

        # define neural network
        net = MLPVanilla(in_features=config.num_feature, 
                        num_nodes=config.hidden_layers_size, 
                        out_features=config.out_feature, 
                        batch_norm=True, 
                        dropout=config.dropout, 
                        output_bias=True
                        )

        # AdamWR optimizer
        optimizer = tt.optim.AdamWR(lr=config.learning_rate, 
                                    decoupled_weight_decay=config.decoupled_weight_decay,
                                    cycle_multiplier=config.cycle_multiplier
                                    )

        # initalize model
        self.model = PCH(net, optimizer, duration_index=np.array(config['duration_index'], dtype='float32'))


class RSF(BaseSksurv):
    '''
    Random Survival Forests.

    Child of BaseSksurv.
    '''
    def __init__(self, config):
        '''
        Args:
            - config: configuration dictionary from experiments.configurations
        '''
        super().__init__(config)
        self.eval_offset = 0
        self.model = RandomSurvivalForest(n_estimators=config.epochs, 
                                            verbose=1,
                                            max_depth=4,
                                            n_jobs=-1
                                            )

class SurvTRACE(BaseModel):
    '''
    Class to run SurvTRACE and its variants.
    
    Child of BaseModel.

    Attributes:
        - variant_name (str): name of variant
        - hyperparameters (dictionary):
            - contains: batch_size, weight_decay, learning_rate, epochs
        - trainer (survtrace.Trainer clsass): trains self.model
    '''
    def __init__(self, config):
        '''
        Args:
            - config: configuration dictionary from experiments.configurations
        '''
        super().__init__()
        self.eval_offset = 1
        try:
            self.variant_name = config.model.split('_')[1]
        except IndexError:
            self.variant_name = ''
        self.hyperparameters = {
            'batch_size': config.batch_size,
            'weight_decay': config.weight_decay,
            'learning_rate': config.learning_rate,
            'epochs': config.epochs,
        }
        get_model = lambda has_mtl: SurvTraceMulti(config, has_mtl) if config.data == 'seer' else SurvTraceSingle(config, has_mtl)

        # initialize a model and determine loss functions to use
        if self.variant_name == 'woIPS-woMTL':
            self.model = get_model(has_mtl=False)
            metrics_list = [NLLLogistiHazardLoss(),]
        elif self.variant_name == 'woMTL':
            self.model = get_model(has_mtl=False)
            metrics_list = [NLLPCHazardLoss()]
        elif self.variant_name == 'woIPS':
            self.model = get_model(has_mtl=True)
            metrics_list = [NLLLogistiHazardLoss(), BCELoss(), MSELoss()]
        # complete survtrace
        else:
            self.model = get_model(has_mtl=True)
            metrics_list = [NLLPCHazardLoss(), BCELoss(), MSELoss()]

        # initialize trainer
        self.trainer = Trainer(self.model, metrics=metrics_list)


    def train(self, data: Data):
        '''
        Trains self.model.

        Args: 
            - data: Data class from experiments.data_class
        
        Returns:
            Nothing.
            self.model is trained.
            Updates self.epochs_trained with total number of epochs from config.
        '''
        train_loss, val_loss, last_epoch = self.trainer.fit((data.df_train, data.df_y_train), 
                                                            (data.df_val, data.df_y_val), 
                                                            **self.hyperparameters
                                                            )
        self.epochs_trained = last_epoch