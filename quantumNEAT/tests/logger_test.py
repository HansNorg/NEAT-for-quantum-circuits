import unittest
import logging

class TestLogger(unittest.TestCase):
    # @unittest.skip("Logging testing interferes with logging")
    def test_main_logger(self):
        self.logger = logging.getLogger("quantumNEAT")
        self.logger.debug("Main logger debug message")
        self.logger.info("Main logger info message")
        self.logger.warning("Main logger warning message")
        self.logger.error("Main logger error message")
        self.logger.critical("Main logger critical message")

    # @unittest.skip("Logging testing interferes with logging of other tests")
    def test_test_logger(self):
        self.logger = logging.getLogger("test_quantumNEAT")
        self.logger.debug("Test logger debug message")
        self.logger.info("Test logger info message")
        self.logger.warning("Test logger warning message")
        self.logger.error("Test logger error message")
        self.logger.critical("Test logger critical message")

if __name__ == '__main__':
    unittest.main()