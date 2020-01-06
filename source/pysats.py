#!C:\Users\jakob\Anaconda3\pythonw.exe
# -*- coding: utf-8 -*-

"""
FILE DESCRIPTION:

This file is a wrapper for the Java-Python "bridges" Lsvm.py, Gsvm.py and Mrvm.py. With the class PySats you can create instances of one of the follwowing three valuations models from the sprctrum auction test suite (SATS):
    The Local Synergy Value Model (LSVM): PySats.getInstance().create_lsvm() creates a instance from the class _Lsvm (see Lsvm.py)
    The Global Synergy Value Model (GSVM): PySats.getInstance().create_gsvm() creates a instance from the class _Gsvm (see Gsvm.py)
    The Multi Region Value Model (MRVM): PySats.getInstance().create_mrvm() creates a instance from the class _Mrvm (see Mrvm.py)

See example_javabridge.py for an example of how to use the class PySats.
"""

# Libs
import os

__author__ = 'Fabio Isler, Jakob Weissteiner'
__copyright__ = 'Copyright 2019, Deep Learning-powered Iterative Combinatorial Auctions: Jakob Weissteiner and Sven Seuken'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Jakob Weissteiner'
__email__ = 'weissteiner@ifi.uzh.ch'
__status__ = 'Dev'


# %%
class PySats:
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if PySats.__instance is None:
            PySats()
        return PySats.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if PySats.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            import jnius_config
            jnius_config.set_classpath(
                '.', os.path.join('lib', '*'))
            PySats.__instance = self

    def create_lsvm(self, seed=None, number_of_national_bidders=1, number_of_regional_bidders=5):
        from source.lsvm import _Lsvm
        return _Lsvm(seed, number_of_national_bidders, number_of_regional_bidders)

    def create_gsvm(self, seed=None, number_of_national_bidders=1, number_of_regional_bidders=6):
        from source.gsvm import _Gsvm
        return _Gsvm(seed, number_of_national_bidders, number_of_regional_bidders)

    def create_mrvm(self, seed=None, number_of_national_bidders=3, number_of_regional_bidders=4, number_of_local_bidders=3):
        from source.mrvm import _Mrvm
        return _Mrvm(seed, number_of_national_bidders, number_of_regional_bidders, number_of_local_bidders)
