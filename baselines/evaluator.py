# calculate time dependent concordanane index

import abc
from cgi import test
from collections import defaultdict
import numpy as np
from sksurv.metrics import concordance_index_ipcw
from baselines.utils import df_to_event_time_array


# TODO: have model parameter accept model class from baselines.models
# probably need to add a method in baselines.models to calculate survival functions
# then could likely reduce the number of evaluator classes since would not need to
# overwrite calc_survival_function
class EvaluatorBase:

    def __init__(self, data, model, config, offset: int, test_set: bool=True):

        self.model = model
        self.offset = offset
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
    
    @abc.abstractclassmethod
    def calc_survival_function(self):
        pass

    def _calc_risk(self):
        surv = self.calc_survival_function()
        return 1 - surv

    def calc_concordance_index_ipcw(self, event_var_name='event'):
        risk = self._calc_risk()
        self._calc_concordance_index_ipcw_base(risk, event_var_name)

    def eval(self):
        self.calc_concordance_index_ipcw()
        return self.metric_dict
        

class EvaluatorSingle(EvaluatorBase):

    def __init__(self, data, model, config, offset=1):
        super().__init__(data, model, config, offset)
        self.compute_baseline_hazards = True if config.model == 'DeepSurv' else False

    def calc_survival_function(self):
        if self.compute_baseline_hazards:
            _ = self.model.compute_baseline_hazards()
        return self.model.predict_surv(self.x_eval)


class EvaluatorBaseV2:

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
        surv = self.calc_survival_function()
        return 1 - surv

    def calc_concordance_index_ipcw(self, event_var_name='event'):
        risk = self._calc_risk()
        self._calc_concordance_index_ipcw_base(risk, event_var_name)

    def eval(self):
        self.calc_concordance_index_ipcw()
        return self.metric_dict
        

class EvaluatorSingleV2(EvaluatorBaseV2):

    def __init__(self, data, model, config, test_set=True):
        super().__init__(data, model, config, test_set)
   
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


class EvaluatorCompeting(EvaluatorBase):

    def __init__(self, data, model, config, offset=0, test_set=True):
        super().__init__(data, model, config, offset, test_set)
        self.num_event = config.num_event

    def calc_survival_function(self):
        return self.model.predict_surv(self.x_eval)

    def predict_cif(self, event_idx):
        return self.model.predict_cif(self.x_eval)[event_idx, :, :].transpose()

    def _calc_risk(self, event_idx):
        return self.predict_cif(event_idx)

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


class EvaluatorCPH(EvaluatorBase):

    def __init__(self, data, model, config, offset=0):
        super().__init__(data, model, config, offset)

    def calc_survival_function(self):
        surv = self.model.predict_survival_function(self.x_eval)
        surv = np.array([f.y for f in surv])
        return surv


class EvaluatorRSF(EvaluatorBase):

    def __init__(self, data, model, config, offset=0):
        super().__init__(data, model, config, offset)

    def calc_survival_function(self):
        return self.model.predict_survival_function(self.x_eval)


class EvaluatorSingleDSM(EvaluatorBase):

    def __init__(self, data, model, config, offset=0, test_set=True):
        super().__init__(data, model, config, test_set=test_set, offset=offset)

    def calc_survival_function(self):
        return self.model.predict_survival(self.x_eval.astype('float64'), self.times.tolist())


class EvaluatorCompetingDSM(EvaluatorBase):

    def __init__(self, data, model, config, offset=0, test_set=True):
        super().__init__(data, model, config, test_set=test_set, offset=offset)
        self.num_event = config.num_event

    def calc_survival_function(self, event_idx):
        return self.model.predict_survival(self.x_eval.astype('float64'), self.times.tolist(), risk=event_idx+1)

    def _calc_risk(self, event_idx):
        return 1 - self.calc_survival_function(event_idx)

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


class EvaluatorCompetingV2(EvaluatorBaseV2):

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