import unittest
from unittest import TestSuite
import qneat.logger as log
import logging
import sys

def load_tests(loader, tests, pattern):
    ''' Discover and load all unit tests in all files named ``*_test.py`` in ``./src/``
    '''
    suite = TestSuite()
    for all_test_suite in unittest.defaultTestLoader.discover('./tests', pattern='*_test.py'):
        for test_suite in all_test_suite:
            suite.addTests(test_suite)
    return suite

if __name__ == '__main__':
    log.QNEATLogger("test", mode="w")
    unittest.main()