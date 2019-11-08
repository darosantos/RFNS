# -*- coding: utf-8 -*-
import logging


class LoggerEnginnering(object):

    __slots__ = ('name', 'level', 'path', 'log_file',
                 'default_extension', 'formatter', 'logger')

    def __init__(self, log_file='', path='',
                 level=logging.DEBUG, drop_old=False, name='enginnering'):
        self.name = name
        self.level = level
        self.path = path
        self.log_file = log_file
        self.default_extension = '.log'
        self.formatter = '%(asctime)s %(levelname)s %(message)s'
        self.logger = None

        if drop_old:
            self.reset()

        self.run()

    def __del__(self):
        del self.name
        del self.level
        del self.path
        del self.log_file
        del self.default_extension
        del self.formatter

    def get_log_file(self):
        log_content = ''
        log_file = self.get_log_name()
        with open(log_file) as hfile:
            log_content = hfile.read()
        return log_content

    def get_log_name(self):
        log_name = ''
        if (self.path != ''):
            log_name = self.path
        if (self.log_file is not None):
            log_name = log_name + self.log_file
        elif(self.name is not None):
            log_name = log_name + self.name + self.default_extension
        else:
            raise TypeError('Log name not empty')
        return log_name

    def reset(self):
        from os import remove
        from os.path import isfile

        if isfile(self.get_log_name()):
            remove(self.get_log_name())

    def run(self):
        if self.log_file == '':
            return

        handler = logging.FileHandler(self.get_log_name())
        handler.setFormatter(logging.Formatter(self.formatter))

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        self.logger.addHandler(handler)

    def add(self, type, msg):
        if self.log_file == '':
            return

        msg = str(msg)
        if type == 'info':
            self.logger.info(msg)
        elif type == 'debug':
            self.logger.debug(msg)
        else:
            raise TypeError('Log level not corret')
