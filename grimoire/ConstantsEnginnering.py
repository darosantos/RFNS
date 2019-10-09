# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:19:58 2019

@author: Danilo Santos
"""
import enum


class ConstantsEnginnering(enum.Enum):
    ESTRATEGY_TRAINNING_SINGLE = 0
    ESTRATEGY_TRAINNING_BLOCK = 1

    __const__ = ('ESTRATEGY_TRAINNING_SINGLE', 'ESTRATEGY_TRAINNING_BLOCK')
    
    __slots__ = ()

    def __setattr__(self, name, value):
        if str(name) not in self.__const__:
            self.__slots__.__setattr__(name, value)
            # super().__setattr__(name, value)
        else:
            raise TypeError('Access const impossible for ', name)

    @staticmethod
    def get_constant(const_name):
        pass
