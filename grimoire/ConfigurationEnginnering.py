from grimoire.LoggerEnginnering import LoggerEnginnering

from time import gmtime, strftime
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklear.preprocessing import StandardScaler, RobustScaler

import numpy as np


class ConfigurationEnginnering:

    __slots__ = ('chunck', 'autoclean', 'str_data_format',
                 'preprocessing_enable', 'preprocessing_data',
                 'preprocessing_target', 'preprocessing_scaler',
                 'encoder_X', 'encoder_y', 'normalize_enable', 'scaler',
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
        self.preprocessing_scaler = False
        self.encoder_X = None
        self.encoder_y = None
        self.normalize_enable = False
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
        del self.autoclean
        del self.str_data_format
        del self.preprocessing_enable
        del self.preprocessing_data
        del self.preprocessing_target
        del self.preprocessing_scaler
        del self.encoder_X
        del self.encoder_y
        del self.normalize_enable
        del self.scaler
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

    def run_scaler_data(self, scaler_type, my_scaler=None):
        if scaler_type == 0:
            self.scaler = StandardScaler(copy=True,
                                         with_mean=True,
                                         with_std=True)
        elif scaler_type == 1:
            self.scaler = RobustScaler(with_centering=True,
                                       with_scaling=True,
                                       quantile_range=(25.0, 75.0),
                                       copy=True)
        elif scaler_type == 2:
            self.scaler = my_scaler
        else:
            raise TypeError("Don't you specified scaler?")

    def run_encoder(self):
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

        if self.preprocessing_scaler & (self.scaler is None):
            self.scaler = StandardScaler(copy=True,
                                         with_mean=True,
                                         with_std=True)
