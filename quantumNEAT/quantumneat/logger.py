import sys
import logging
import traceback

def setup_logger(name = "quantumNEAT", console_level = logging.WARNING, main_file_level = logging.INFO, quantumneat_level = logging.INFO, 
                 test_level = logging.DEBUG, experiments_level = logging.INFO, mode ="a", print_start = True, extra_file_name = "", log_errors = True):
    # print(extra_file_name+"setup_logger")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(filename)s - %(funcName)s - line: %(lineno)d\n%(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, main_file_level))
    
    sh = logging.StreamHandler()
    sh.setLevel(console_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    try:
        fh = logging.FileHandler("logs/"+extra_file_name+"main.log", mode=mode, encoding = "utf-8")    
    except FileNotFoundError:
        fh = logging.FileHandler("quantumneat/logs/"+extra_file_name+"main.log", mode=mode, encoding = "utf-8")
    fh.setLevel(main_file_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    quantumneat_logger = logging.getLogger(f"{name}.quantumneat")
    quantumneat_logger.setLevel(min(console_level, main_file_level, quantumneat_level))

    try:
        fh = logging.FileHandler("logs/"+extra_file_name+"quantumneat.log", mode=mode, encoding = "utf-8")    
    except FileNotFoundError:
        fh = logging.FileHandler("quantumneat/logs/"+extra_file_name+"quantumneat.log", mode=mode, encoding = "utf-8")
    fh.setLevel(quantumneat_level)
    fh.setFormatter(formatter)
    quantumneat_logger.addHandler(fh)
    
    test_logger = logging.getLogger(f"{name}.tests")
    test_logger.setLevel(min(console_level, main_file_level, test_level))
    
    try:
        fh = logging.FileHandler("logs/"+extra_file_name+"test.log", mode=mode, encoding = "utf-8")    
    except FileNotFoundError:
        fh = logging.FileHandler("quantumneat/logs/"+extra_file_name+"test.log", mode=mode, encoding = "utf-8")
    fh.setLevel(test_level)
    fh.setFormatter(formatter)
    test_logger.addHandler(fh)

    experiments_logger = logging.getLogger(f"{name}.experiments")
    experiments_logger.setLevel(min(console_level, main_file_level, experiments_level))

    try:
        fh = logging.FileHandler("logs/"+extra_file_name+"experiments.log", mode=mode, encoding = "utf-8")    
    except FileNotFoundError:
        fh = logging.FileHandler("quantumneat/logs/"+extra_file_name+"experiments.log", mode=mode, encoding = "utf-8")
    fh.setLevel(experiments_level)
    fh.setFormatter(formatter)
    experiments_logger.addHandler(fh)

    if print_start:
        logger.info("==================logger_setup==================")
        quantumneat_logger.info("==================logger_setup==================")
        test_logger.info("==================logger_setup==================")
        experiments_logger.info("==================logger_setup==================")
    
    # log uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            logger.error("================= Keyboard interruption =================")
            sys.__excepthook__(exc_type, exc_value, exc_traceback) # calls default excepthook
            return
        logger.error("================= Uncaught exception =================", exc_info=(exc_type, exc_value, exc_traceback))
        
    if log_errors:
        sys.excepthook = handle_exception

def default_logger(debugging = False, print_start = True, extra_file_name=""):
    if debugging:
        setup_logger(main_file_level=logging.DEBUG, quantumneat_level=logging.DEBUG, experiments_level=logging.DEBUG, print_start=print_start, extra_file_name=extra_file_name)
    else:
        setup_logger(print_start=print_start, extra_file_name=extra_file_name)