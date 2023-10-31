import unittest
import logging

class TestLogger(unittest.TestCase):
    @unittest.skip("Logging testing performed manually")
    def test_main_logger(self):
        logger = logging.getLogger("quantumNEAT")
        logger.debug("Main logger debug message")
        logger.info("Main logger info message")
        logger.warning("Main logger warning message")
        logger.error("Main logger error message")
        logger.critical("Main logger critical message")

    @unittest.skip("Logging testing performed manually")
    def test_test_logger(self):
        logger = logging.getLogger("test_quantumNEAT")
        logger.debug("Test logger debug message")
        logger.info("Test logger info message")
        logger.warning("Test logger warning message")
        logger.error("Test logger error message")
        logger.critical("Test logger critical message")

if __name__ == '__main__':
    unittest.main()