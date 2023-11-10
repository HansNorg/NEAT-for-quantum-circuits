import logging
from quantumneat.logger import default_logger

debugging = True

default_logger(debugging)

logger = logging.getLogger("quantumNEAT")
quantumneat_logger = logging.getLogger("quantumNEAT.quantumneat")
test_logger = logging.getLogger("quantumNEAT.tests")
experiments_logger = logging.getLogger("quantumNEAT.experiments")
logger.info("==================quantumneat.__init__==================")
quantumneat_logger.info("==================quantumneat.__init__==================")
test_logger.info("==================quantumneat.__init__==================")
experiments_logger.info("==================quantumneat.__init__==================")