from grimoire.LoggerEnginnering import LoggerEnginnering

from time import gmtime, strftime

class ConfigurationEnginnering(object):

    __slots__ = ('chunck', 'enable_preprocessing', 'save_matrix_prediction', 
                 'file_name_matrix_prediction', 'autoclean',
                 'start_logging', 'name_file_log', 'drop_old_log', 'logger')
    
    def __init__(self):
        self.chunck = 32
        self.enable_preprocessing = False
        self.autoclean = False
        
        # Configuration for save data predict
        self.save_matrix_prediction = True
        self.file_name_matrix_prediction = ''
        self.format_data_predict = '%d'
        self.delimit_data_predict = ','
        
        # Configuraton for logging
        self.start_logging = False
        self.name_file_log = ''
        self.drop_old_log = True
        self.logger = LoggerEnginnering()
                                        
    def __del__(self):
        del self.chunck
        del self.enable_preprocessing
        del self.save_matrix_prediction
        del self.file_name_matrix_prediction
        del self.autoclean
        del self.start_logging
        del self.name_file_log
        del self.drop_old_log
        del self.logger
    
    def run_logging(self):
        if self.start_logging:
            if self.name_file_log == '':
                local_time = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
                self.name_file_log =  'enginnering_{0}'.format(local_time)
                
            self.logger = LoggerEnginnering(log_file=self.name_file_log,
                                            drop_old=self.drop_old_log)
                                            
    def run_save_predict(self, data_predict):
        if self.save_matrix_prediction:
            if self.file_name_matrix_prediction == '':
                local_time = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
                self.file_name_matrix_prediction = 'matrix_prediction_{0}.txt'.format(local_time)
                
            np.savetxt(fname=self.file_name_matrix_prediction, X=data_predict,
                       fmt=self.format_data_predict, delimiter=self.delimit_data_predict)