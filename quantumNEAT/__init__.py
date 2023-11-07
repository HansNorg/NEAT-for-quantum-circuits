import logging
from quantumneat.logger import setup_logger

debugging = True

if debugging:
    setup_logger(main_file_level=logging.DEBUG)
else:
    setup_logger(main_file_level=logging.INFO)

logger = logging.getLogger("quantumNEAT")
quantumneat_logger = logging.getLogger("quantumNEAT.quantumneat")
test_logger = logging.getLogger("quantumNEAT.tests")
logger.info("==================quantumneat.__init__==================")
quantumneat_logger.info("==================quantumneat.__init__==================")
test_logger.info("==================quantumneat.__init__==================")