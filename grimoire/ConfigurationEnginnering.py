from grimoire.LoggerEnginnering import LoggerEnginnering

from time import gmtime, strftime
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder

import numpy as np


class ConfigurationEnginnering:

    __slots__ = ('chunck', 'autoclean', 'str_data_format',
                 'preprocessing_enable', 'preprocessing_data',
                 'preprocessing_target', 'encoder_X', 'encoder_y',
                 'save_matrix_prediction',  'file_name_matrix_prediction',
                 'format_data_predict', 'delimit_data_predict',
                 'start_logging', 'name_file_log', 'drop_old_log', 'logger')

    def __init__(self):
        self.chunck = 32
        self.autoclean = False
        self.str_data_format = "%%d-%B-%Y_%H-%M-%S"

        # Configuration for prepossing
        self.preprocessing_enable = False
        self.preprocessing_data = False
        self.preprocessing_target = False
        self.encoder_X = None
        self.encoder_y = None
        self.scaler = None

        # Configuration for save data predict
        self.save_matrix_prediction = True
        self.file_name_matrix_prediction = ''
        self.format_data_predict = '%d'
        self.delimit_data_predict = ','

        # Configuraton for logging
        self.start_logging = True
        self.name_file_log = ''
        self.drop_old_log = True
        self.logger = LoggerEnginnering()

    def __del__(self):
        del self.chunck
        del self.enable_preprocessing
        del self.autoclean
        del self.save_matrix_prediction
        del self.file_name_matrix_prediction
        del self.format_data_predict
        del self.delimit_data_predict
        del self.start_logging
        del self.name_file_log
        del self.drop_old_log

    def run_logging(self):
        if self.start_logging:
            if self.name_file_log == '':
                local_time = strftime(self.str_data_format, gmtime())
                self.name_file_log = 'enginnering_{0}'.format(local_time)

            self.logger = LoggerEnginnering(log_file=self.name_file_log,
                                            drop_old=self.drop_old_log)

    def run_save_predict(self, data_predict) -> None:
        if self.save_matrix_prediction:
            if self.file_name_matrix_prediction == '':
                local_time = strftime(self.str_data_format, gmtime())
                self.file_name_matrix_prediction = 'matrix_prediction_{0}.txt'
                self.file_name_matrix_prediction.format(local_time)

            np.savetxt(fname=self.file_name_matrix_prediction, X=data_predict,
                       fmt=self.format_data_predict,
                       delimiter=self.delimit_data_predict)

    def run_encoder_data(self, encoder_type, my_encoder=None):
        if encoder_type == 0:
            self.encoder_X = LabelEncoder()
        elif encoder_type == 1:
            self.encoder_X = OneHotEncoder(categories='auto',
                                           drop=None,
                                           sparse=False,
                                           dtype=np.float64,
                                           handle_unknown='ignore',
                                           n_values='auto',
                                           categorical_features='all')
        elif encoder_type == 2:
            self.encoder_X = OrdinalEncoder(categories='auto',
                                            dtype=np.float64)
        elif encoder_type == 3:
            self.encoder_X = my_encoder
        else:
            raise TypeError("Don't you specified encoder?")

    def run_transformer(self):
        if self.preprocessing_data & (self.enconder_X is None):
            self.encoder_X = OneHotEncoder(categories='auto',
                                           drop=None,
                                           sparse=False,
                                           dtype=np.float64,
                                           handle_unknown='ignore',
                                           n_values='auto',
                                           categorical_features='all')
        if self.preprocessing_target & (self.enconder_y is None):
            self.encoder_y = LabelEncoder()
            
