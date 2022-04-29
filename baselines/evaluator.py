# calculate time dependent concordanane index

import abc
from collections import defaultdict
import numpy as np
from sksurv.metrics import concordance_index_ipcw
from baselines.utils import df_to_event_time_array


class EvaluatorBase:

    def __init__(self, data, model, config, test_set: bool=True):

        self.model_name = config.model
        self.model = model
        self.offset = model.eval_offset
        self.df_train_all = data.df.loc[data.df_train.index]

        if test_set:
            self.x_eval = data.x_test
            self.df_y_eval = data.df_y_test
        # use validation set
        else:
            self.x_eval = data.x_val
            self.df_y_eval = data.df_y_val


        self.times = config['duration_index'][1:-1]
        self.horizons = config['horizons']
        self.metric_dict = defaultdict(list)
    
    def _make_event_time_array(self, event_var_name):
        def helper(df):
            return df_to_event_time_array(df, event_var_name=event_var_name)
        
        return helper(self.df_train_all), helper(self.df_y_eval)


    def _calc_concordance_index_ipcw_base(self, risk, event_var_name, event_dict_label='', event_print_label=''):

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
    
    
    @ abc.abstractclassmethod
    def _calc_risk(self):
        pass

    def calc_concordance_index_ipcw(self, event_var_name='event'):
        risk = self._calc_risk()
        self._calc_concordance_index_ipcw_base(risk, event_var_name)

    def eval(self):
        self.calc_concordance_index_ipcw()
        return self.metric_dict
        

class EvaluatorSingle(EvaluatorBase):

    def __init__(self, data, model, config, test_set=True):
        super().__init__(data, model, config, test_set)
   
    # TODO: move to baselinse.models
    #   ideally it would be nice to have a generic interface for calculating risk in baselines.models 
    #   like the train method. This may requrie additional refactor of evaluators and models to make that work
    # Note: self.model.model.{function}:
    #   - first model references a class from baselines.models
    #   - second model references a model from pycox, sksurv, DSM
    def _calc_risk(self):
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
        else:
            return 1 - self.model.model.predict_surv(self.x_eval)


# TODO: move to baselinse.models
#   ideally it would be nice to have a generic interface for calculating risk in baselines.models 
#   like the train method. This may requrie additional refactor of evaluators and models to make that work
# Note: self.model.model.{function}:
#   - first model references a class from baselines.models
#   - second model references a model from pycox, sksurv, DSM
class EvaluatorCompeting(EvaluatorBase):

    def __init__(self, data, model, config, test_set=True):
        super().__init__(data, model, config, test_set=test_set)
        self.num_event = config.num_event


    def _calc_risk(self, event_idx):
        if self.model_name == 'DSM':
            return 1 - self.model.model.predict_survival(self.x_eval.astype('float64'), self.times.tolist(), risk=event_idx+1)
        elif self.model_name == 'DeepHit':
            return self.model.model.predict_cif(self.x_eval)[event_idx, :, :].transpose()
        else:
            raise('Model not implemented')

    def calc_concordance_index_ipcw(self, event_idx, event_var_name):
        risk = self._calc_risk(event_idx)
        event_dict_label = f'_{event_idx}'
        event_print_label = f'Event: {event_idx} ' 
        self._calc_concordance_index_ipcw_base(risk, event_var_name, event_dict_label, event_print_label)
    
    def eval(self):
        for event_idx in range(self.num_event):
            event_var_name = f'event_{event_idx}'
            self.calc_concordance_index_ipcw(event_idx, event_var_name)
        return self.metric_dict