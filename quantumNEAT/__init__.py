import logging
from quantumneat.logger import setup_logger

debugging = False

if debugging:
    setup_logger(test_level=logging.DEBUG, file_level=logging.DEBUG)
else:
    setup_logger(test_level=logging.INFO, file_level=logging.INFO)