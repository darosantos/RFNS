from grimoire.LoggerEnginnering import LoggerEnginnering

from time import gmtime, strftime
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, RobustScaler

import numpy as np


class ConfigurationEnginnering:

    __slots__ = ('chunck', 'autoclean', 'str_data_format',
                 'preprocessing_enable', 'encoder_enable', 'encoder_data',
                 'encoder_target', 'encoder_not_type', 'encoder_feature',
                 'encoder_flag', 'normalize_enable', 'normalize_flag',
                 'encoder_X', 'encoder_y', 'normalize_scaler',
                 'save_matrix_prediction',  'file_name_matrix_prediction',
                 'format_data_predict', 'delimit_data_predict',
                 'start_logging', 'name_file_log', 'drop_old_log', 'logger')

    def __init__(self):
        self.chunck = 32
        self.autoclean = False
        self.str_data_format = "%%d-%B-%Y_%H-%M-%S"

        # Configuration for prepossing
        self.preprocessing_enable = False
        # Usado para criar os codificadores
        self.encoder_enable = False
        # Usado para codificar os dados
        self.encoder_data = False
        # Usado para codificar os rótulos
        self.encoder_target = False
        # Usado para definir os tipos de dados que não são codificados
        self.encoder_not_type = [int, float, complex,
                                 np.int8, np.int16, np.int32, np.int64,
                                 np.float, np.float64, np.complex64]
        # Usado para armazenar os atributos e valores codificados
        self.encoder_feature = {}
        # Usado para contorlar se X ou y foram codificados
        self.encoder_flag = [0, 0]
        # Usado para habilitar a normalização
        self.normalize_enable = False
        # Usado para controle da normalização caso feita ou não
        self.normalize_flag = 0
        # Objeto usado para codificar X
        self.encoder_X = None
        # Objeto para codificar y
        self.encoder_y = None
        # Objeto para normalziar os dados em X
        self.normalize_scaler = None

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
        del self.encoder_enable
        del self.encoder_data
        del self.encoder_target
        del self.encoder_not_type
        del self.encoder_feature
        del self.encoder_flag
        del self.normalize_enable
        del self.normalize_flag
        del self.encoder_X
        del self.encoder_y
        del self.normalize_scaler

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

    def run_encoder_data(self, encoder_type=1, my_encoder=None, force=False):
        if (self.encoder_enable & (self.encoder_X is None)) | force:
            if encoder_type == 0:
                self.encoder_X = LabelEncoder()
            elif encoder_type == 1:
                self.encoder_X = OneHotEncoder(categories='auto',
                                               drop=None,
                                               sparse=False,
                                               dtype=np.float64,
                                               handle_unknown='ignore',
                                               n_values='auto')
            elif encoder_type == 2:
                self.encoder_X = OrdinalEncoder(categories='auto',
                                                dtype=np.float64)
            elif encoder_type == 3:
                self.encoder_X = my_encoder
            else:
                raise TypeError("Don't you specified encoder for data?")

    def run_encoder_target(self, encoder_type=0, my_encoder=None, force=False):
        if (self.encoder_enable & (self.encoder_y is None)) | force:
            if encoder_type == 0:
                self.encoder_y = LabelEncoder()
            elif encoder_type == 1:
                self.encoder_y = my_encoder
            else:
                raise TypeError("Don't you specified encoder for target?")

    def run_scaler_data(self, scaler_type=0, my_scaler=None, force=False):
        if (self.normalize_enable & (self.normalize_scaler is None)) | force:
            if scaler_type == 0:
                self.normalize_scaler = StandardScaler(copy=True,
                                                       with_mean=True,
                                                       with_std=True)
            elif scaler_type == 1:
                p_qr = (25.0, 75.0)
                self.normalize_scaler = RobustScaler(with_centering=True,
                                                     with_scaling=True,
                                                     quantile_range=p_qr,
                                                     copy=True)
            elif scaler_type == 2:
                self.normalize_scaler = my_scaler
            else:
                raise TypeError("Don't you specified scaler?")

    def run_preprocessing(self):
        if self.preprocessing_data & (self.enconder_X is None):
            self.encoder_X = OneHotEncoder(categories='auto',
                                           drop=None,
                                           sparse=False,
                                           dtype=np.float64,
                                           handle_unknown='ignore',
                                           n_values='auto')

        if self.preprocessing_target & (self.enconder_y is None):
            self.encoder_y = LabelEncoder()

        if self.preprocessing_scaler & (self.scaler is None):
            self.scaler = StandardScaler(copy=True,
                                         with_mean=True,
                                         with_std=True)
