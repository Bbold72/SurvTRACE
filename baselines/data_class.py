from easydict import EasyDict
import numpy as np
from survtrace.dataset import load_data

from baselines.utils import df_to_event_time_array


class Data:
    '''
    loads, processes data, and splits into train, validation, and test sets
    Data is accessed through this class.
    '''

    def __init__(self, config: EasyDict, censor_event=False):

        # load data
        # TODO: splitting the data into train, validation, test should be separate process. Each run of each experiment
        # should use the same train, validation, test datasets so the results are comparable. currently each run, of each
        # dataset uses its own train, validation, test.
        self.df, self.df_train, self.df_y_train, self.df_test, self.df_y_test, self.df_val, self.df_y_val = load_data(config)
        
        # additional post processing for models that are not SurvTRACE
        if not config.model.startswith('SurvTRACE'):
            # censor event
            if censor_event:
                censor_event = lambda df: df.drop(config.event_to_censor, axis=1).rename(columns={config.event_to_keep: 'event'})

                self.df = censor_event(self.df)
                self.df_y_train = censor_event(self.df_y_train)
                self.df_y_val = censor_event(self.df_y_val)
                self.df_y_test = censor_event(self.df_y_test)

            # competing events
            if ((config.model == 'DeepHit' or config.model == 'DSM') and config.data == 'seer'):
                # make event variable for pycox model
                # 0: right censored
                # 1: Heart Disease (event_0)
                # 2: Breast Cancer (event_1)
                def make_event_col(df):
                    df['event'] = np.where(df['event_0'] == 1,
                                            1,
                                            np.where(df['event_1'] == 1,
                                                    2,
                                                    0
                                                    )
                                            )
                    return df

                self.df, self.df_y_train, self.df_y_val, self.df_y_test = map(make_event_col, [self.df, self.df_y_train, self.df_y_val, self.df_y_test])
        

            # convet data to format necessary for pycox
            self.x_train = np.array(self.df_train, dtype='float32')
            self.x_val = np.array(self.df_val, dtype='float32')
            self.x_test = np.array(self.df_test, dtype='float32')

            if config.model == 'PCHazard':
                y_df_to_tuple = lambda df: tuple([np.array(df['duration'], dtype='int64'), np.array(df['event'], dtype='int64'), np.array(df['proportion'], dtype='float32')])
            else:
                y_df_to_tuple = lambda df: tuple([np.array(df['duration'], dtype='int64'), np.array(df['event'], dtype='int64')])

            self.y_train = y_df_to_tuple(self.df_y_train)
            self.y_val = y_df_to_tuple(self.df_y_val)

            # package in tuple to pass to pycox models
            self.val_data = tuple([self.x_val, self.y_val])

            # process data for DSM
            if config.model == 'DSM':
                _, self.train_outcomes = self.y_train
                self.train_times = np.array(self.df.loc[self.df_train.index]['duration'])
                _, self.val_outcomes = self.y_val
                self.val_times = np.array(self.df.loc[self.df_val.index]['duration'])

            if config.model == 'CPH' or config.model == 'RSF':
                self.y_et_train = df_to_event_time_array(self.df_y_train)
