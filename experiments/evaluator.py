# calculates time dependent concordanane index

import abc
from collections import defaultdict
from typing import Tuple
from easydict import EasyDict
import numpy as np
from sksurv.metrics import concordance_index_ipcw
from experiments.utils import df_to_event_time_array
from experiments.data_class import Data

class EvaluatorBase:
    """
    Base class that for caluculating time dependent concordance index

    Attributes:
        - model_name (str): name of model evaluated on
        - model: class from experiments.model
        - offset: number columns to skip from the left of risk calculation
        - df_train_all: dataframe with features and events & durations trained on
        - x_eval: features data to evaluate on. either validation or test set
        - df_y_eval: dataframe of events and durations to evaluate on.
        - times: duration indexes
        - horionzs: quantiles 
        - metric_dict: (defaultdict): stores calculated metrics
    """
    def __init__(self, data: Data, model, config: EasyDict, test_set: bool=True):
        '''
        Args:
            - data: class from experiments.Data
            - model: class from experiments.models
            - config: configuration file from experiments.configurations
            - test_set (bool): 
                - True: evaluate on test set
                - False: evaluate on validation set
        '''
        self.model_name = config.model
        self.model = model
        self.offset = model.eval_offset
        self.df_train_all = data.df.loc[data.df_train.index]

        if test_set:
            self.x_eval = data.df_test if self.model_name.startswith('survtrace') else data.x_test
            self.df_y_eval = data.df_y_test
        # use validation set
        else:
            self.x_eval = data.df_val if self.model_name.startswith('survtrace') else data.x_val
            self.df_y_eval = data.df_y_val


        self.times = config['duration_index'][1:-1]
        self.horizons = config['horizons']
        self.metric_dict = defaultdict(list)
    

    def _make_event_time_array(self, event_var_name: str) -> Tuple[np.array, np.array]:
        '''
        helper function to convert training data and evaluation data

        Args:
            - event_var_name (str): variable name of event
        
        Returns:
            - tuple of event time arrays
            - Tuple[0]: training data
            - Tuple[1]: evaluation data
        '''
        def helper(df):
            return df_to_event_time_array(df, event_var_name=event_var_name)
        
        return helper(self.df_train_all), helper(self.df_y_eval)


    def _calc_concordance_index_ipcw_base(self, risk, event_var_name: str, event_dict_label: str='', event_print_label: str=''):
        '''
        Calculates time dependent concordance index.

        Use sksurv.concordance_index_ipcw.

        Args:
            - risk: array of models risk predicts for each time horizon.
            - event_var_name (str): variable name of event.
            - event_dict_label (str):
            - event_print_label (str):

        Return:
            Nothing.
            Adds each metric to attribute metric_dict.
        '''
        et_train, et_test = self._make_event_time_array(event_var_name)

        cis = []
        for i, _ in enumerate(self.times):
            cis.append(
                concordance_index_ipcw(et_train, et_test, estimate=risk[:, i+self.offset], tau=self.times[i])[0]
                )
            self.metric_dict[f'{self.horizons[i]}_ipcw{event_dict_label}'] = cis[i]

        for horizon in enumerate(self.horizons):
            print(f"{event_print_label}For {horizon[1]} quantile,")
            print("TD Concordance Index - IPCW:", cis[horizon[0]])
    
    
    @abc.abstractclassmethod
    def _calc_risk(self):
        '''
        Calcuates risk based on provided trained model.
        '''
        pass

    @abc.abstractclassmethod
    def calc_concordance_index_ipcw(self):
        '''
        Calculates time dependent concordance index.
        '''
        pass

    @abc.abstractclassmethod
    def eval(self):
        '''
        Calculates time dependent concordance index and return metrics.
        '''
        pass
        

class EvaluatorSingle(EvaluatorBase):
    """
    Caluculating time dependent concordance index on single events.

    Child of EvaluatorBase.
    """
    def __init__(self, data: Data, model, config: EasyDict, test_set: bool=True):
        '''
        Args:
            - data: class from experiments.Data
            - model: class from experiments.models
            - config: configuration file from experiments.configurations
            - test_set (bool): 
                - True: evaluate on test set
                - False: evaluate on validation set
        '''
        super().__init__(data, model, config, test_set)
   
    # TODO: move to baselinse.models
    #   ideally it would be nice to have a generic interface for calculating risk in experiments.models 
    #   like the train method. This may requrie additional refactor of evaluators and models to make that work
    # Note: self.model.model.{function}:
    #   - first model references a class from experiments.models
    #   - second model references a model from pycox, sksurv, DSM
    def _calc_risk(self):
        '''
        Calculates risk based on trained model provided.

        Returns:
            Array of risk per time horizon.
        '''
        if self.model_name == 'DSM':
            return 1 - self.model.model.predict_survival(self.x_eval.astype('float64'), self.times.tolist())
        elif self.model_name == 'RSF':
            return 1 - self.model.model.predict_survival_function(self.x_eval)
        elif self.model_name == 'CPH':
            surv = self.model.model.predict_survival_function(self.x_eval)
            surv = np.array([f.y for f in surv])
            return 1 - surv
        elif self.model_name == 'DeepSurv':
            _ = self.model.model.compute_baseline_hazards()
            return 1 - self.model.model.predict_surv(self.x_eval)
        elif self.model_name.startswith('survtrace'):
            return 1 - self.model.model.predict_surv(self.x_eval, batch_size=None).cpu()
        else:
            return 1 - self.model.model.predict_surv(self.x_eval)


    def calc_concordance_index_ipcw(self, event_var_name: str='event'):
        '''
        Calculates time dependent concordance index.

        Args:
            - event_var_name (str): variable name of event.
                - Defualt = 'event'

        Return:
            Nothing.
            Adds each metric to attribute metric_dict.
        '''
        risk = self._calc_risk()
        self._calc_concordance_index_ipcw_base(risk, event_var_name)


    def eval(self):
        '''
        Calculates time dependent concordance index and return metrics.
        '''
        self.calc_concordance_index_ipcw()
        return self.metric_dict
        


class EvaluatorCompeting(EvaluatorBase):
    """
    Caluculating time dependent concordance index on competing events.

    Child of EvaluatorBase.

    Attributes:
        - num_event (int): number of competing events
    """
    def __init__(self, data: Data, model, config: EasyDict, test_set: bool=True):
        '''
        Args:
            - data: class from experiments.Data
            - model: class from experiments.models
            - config: configuration file from experiments.configurations
            - test_set (bool): 
                - True: evaluate on test set
                - False: evaluate on validation set
        '''
        super().__init__(data, model, config, test_set=test_set)
        self.num_event = config.num_event

    # TODO: move to baselinse.models
    #   ideally it would be nice to have a generic interface for calculating risk in experiments.models 
    #   like the train method. This may requrie additional refactor of evaluators and models to make that work
    # Note: self.model.model.{function}:
    #   - first model references a class from experiments.models
    #   - second model references a model from pycox, sksurv, DSM
    def _calc_risk(self, event_idx):
        '''
        Calculates risk based on trained model provided.

        Returns:
            Array of risk per time horizon.
        '''
        if self.model_name == 'DSM':
            return 1 - self.model.model.predict_survival(self.x_eval.astype('float64'), self.times.tolist(), risk=event_idx+1)
        elif self.model_name == 'DeepHit':
            return self.model.model.predict_cif(self.x_eval)[event_idx, :, :].transpose()
        elif self.model_name.startswith('survtrace'):
            return 1 - self.model.model.predict_surv(self.x_eval, batch_size=10000, event=event_idx).cpu()
        else:
            raise('Model not implemented')


    def calc_concordance_index_ipcw(self, event_idx: int, event_var_name: str):
        '''
        Calculates time dependent concordance index.

        Args:
            - event_idx: index of competing event.
            - event_var_name (str): variable name of event.

        Return:
            Nothing.
            Adds each metric to attribute metric_dict.
        '''
        risk = self._calc_risk(event_idx)
        event_dict_label = f'_{event_idx}'
        event_print_label = f'Event: {event_idx} ' 
        self._calc_concordance_index_ipcw_base(risk, event_var_name, event_dict_label, event_print_label)
    

    def eval(self):
        '''
        Calculates time dependent concordance index for each competing event and return metrics.
        '''
        for event_idx in range(self.num_event):
            event_var_name = f'event_{event_idx}'
            self.calc_concordance_index_ipcw(event_idx, event_var_name)
        return self.metric_dict