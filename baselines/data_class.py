import numpy as np
from survtrace.dataset import load_data

class Data:

    def __init__(self, config):

        # load data
        self.df, self.df_train, self.df_y_train, self.df_test, self.df_y_test, self.df_val, self.df_y_val = load_data(config)

        # censor event
        if config.data == 'seer':

            censor_event = lambda df: df.drop(config.event_to_censor, axis=1).rename(columns={event_to_keep: 'event'})

            event_to_keep = '0' if config.event_to_censor.split('_')[1] == '1' else '1'
            event_to_keep = 'event_' + event_to_keep

            self.df = censor_event(self.df)
            self.df_y_train = censor_event(self.df_y_train)
            self.df_y_val = censor_event(self.df_y_val)
            self.df_y_test = censor_event(self.df_y_test)

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

