import unittest
import logging

class TestLogger(unittest.TestCase):
    @unittest.skipUnless(__name__=='__main__',"Logging testing performed manually")
    def test_main_logger(self):
        logger = logging.getLogger("quantumNEAT")
        logger.debug("Main logger debug message")
        logger.info("Main logger info message")
        logger.warning("Main logger warning message")
        logger.error("Main logger error message")
        logger.critical("Main logger critical message")

    @unittest.skipUnless(__name__=='__main__',"Logging testing performed manually")
    def test_quantumneat_logger(self):
        logger = logging.getLogger("quantumNEAT.quantumneat")
        logger.debug("quantumneat logger debug message")
        logger.info("quantumneat logger info message")
        logger.warning("quantumneat logger warning message")
        logger.error("quantumneat logger error message")
        logger.critical("quantumneat logger critical message")

    @unittest.skipUnless(__name__=='__main__',"Logging testing performed manually")
    def test_test_logger(self):
        logger = logging.getLogger("quantumNEAT.tests")
        logger.debug("Test logger debug message")
        logger.info("Test logger info message")
        logger.warning("Test logger warning message")
        logger.error("Test logger error message")
        logger.critical("Test logger critical message")

if __name__ == '__main__':
    unittest.main()