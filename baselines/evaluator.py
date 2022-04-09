from collections import defaultdict
import numpy as np
from sksurv.metrics import concordance_index_ipcw


class Evaluator:

    def __init__(self, data, model, config, offset=1):

        self.model = model
        self.offset = offset
        self.x_test = data.x_test
        self.times = config['duration_index'][1:-1]
        self.horizons = config['horizons']
        self.metric_dict = defaultdict(list)
        self.compute_baseline_hazards = True if config.model == 'DeepSurv' else False

        get_target = lambda df: (df['duration'].values, df['event'].values)
                
        df_train_all = data.df.loc[data.df_train.index]
        durations_train, events_train = get_target(df_train_all)
        self.et_train = np.array([(events_train[i], durations_train[i]) for i in range(len(events_train))],
                        dtype = [('e', bool), ('t', float)])
        
        durations_test, events_test = get_target(data.df_y_test)
        self.et_test = np.array([(events_test[i], durations_test[i]) for i in range(len(events_test))],
                    dtype = [('e', bool), ('t', float)])

    def calc_survival_function(self):
        if self.compute_baseline_hazards:
            _ = self.model.compute_baseline_hazards()
        self.surv = self.model.predict_surv(self.x_test)

    def calc_concordance_index_ipcw(self):
        self.risk = (1 - self.surv)
        cis = []
        for i, _ in enumerate(self.times):
            cis.append(
                concordance_index_ipcw(self.et_train, self.et_test, estimate=self.risk[:, i+self.offset], tau=self.times[i])[0]
                )
            self.metric_dict[f'{self.horizons[i]}_ipcw'] = cis[i]

        for horizon in enumerate(self.horizons):
            print(f"For {horizon[1]} quantile,")
            print("TD Concordance Index - IPCW:", cis[horizon[0]])
        
        
    def eval(self):
        self.calc_survival_function()
        self.calc_concordance_index_ipcw()
        return self.metric_dict

