#!/usr/bin/env python

from distutils.core import setup

setup(name='EnginneringForest',
      version='1.0.0',
      description='Machine Learning Algoritm with Ensemble-based Decision Trees',
      author='Danilo Santos',
      author_email='danilo_santosrs@hotmail.com',
      url='https://github.com/darosantos/RFNS/',
	  license='Apache-2.0',
	   zip_safe=False,
      packages=['grimoiri', 'grimoiri.BaseEnginnering', 'grimoiri.CacheEnginnering', 
				'grimoiri.ClassifierEnginneringForest', 
				'grimoiri.ConfigurationEnginnering', 
				'grimoiri.EnginneringForest', 'grimoiri.LoggerEnginnering'],
     )