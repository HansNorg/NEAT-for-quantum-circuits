import logging
from quantumneat.logger import setup_logger

debugging = True

if debugging:
    setup_logger(test_level=logging.DEBUG, file_level=logging.DEBUG)
else:
    setup_logger(test_level=logging.INFO, file_level=logging.INFO)

logger = logging.getLogger("quantumNEAT")
test_logger = logging.getLogger("test_quantumNEAT")
logger.info("==================quantumneat.__init__==================")
test_logger.info("==================quantumneat.__init__==================")