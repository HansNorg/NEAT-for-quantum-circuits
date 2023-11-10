import logging
    
def setup_logger(name = "quantumNEAT", console_level = logging.WARNING, main_file_level = logging.INFO, quantumneat_level = logging.INFO, test_level = logging.DEBUG, experiments_level = logging.INFO, mode ="a"):
    print("setup_logger")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(min(console_level, main_file_level))
    
    sh = logging.StreamHandler()
    sh.setLevel(console_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    try:
        fh = logging.FileHandler("logs/main.log", mode=mode, encoding = "utf-8")    
    except FileNotFoundError:
        fh = logging.FileHandler("quantumneat/logs/main.log", mode=mode, encoding = "utf-8")
    fh.setLevel(main_file_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    quantumneat_logger = logging.getLogger(f"{name}.quantumneat")
    quantumneat_logger.setLevel(min(console_level, main_file_level, quantumneat_level))

    try:
        fh = logging.FileHandler("logs/quantumneat.log", mode=mode, encoding = "utf-8")    
    except FileNotFoundError:
        fh = logging.FileHandler("quantumneat/logs/quantumneat.log", mode=mode, encoding = "utf-8")
    fh.setLevel(quantumneat_level)
    fh.setFormatter(formatter)
    quantumneat_logger.addHandler(fh)
    
    test_logger = logging.getLogger(f"{name}.tests")
    test_logger.setLevel(min(console_level, main_file_level, test_level))
    
    try:
        fh = logging.FileHandler("logs/test.log", mode=mode, encoding = "utf-8")    
    except FileNotFoundError:
        fh = logging.FileHandler("quantumneat/logs/test.log", mode=mode, encoding = "utf-8")
    fh.setLevel(test_level)
    fh.setFormatter(formatter)
    test_logger.addHandler(fh)

    experiments_logger = logging.getLogger(f"{name}.experiments")
    experiments_logger.setLevel(min(console_level, main_file_level, experiments_level))

    try:
        fh = logging.FileHandler("logs/experiments.log", mode=mode, encoding = "utf-8")    
    except FileNotFoundError:
        fh = logging.FileHandler("quantumneat/logs/experiments.log", mode=mode, encoding = "utf-8")
    fh.setLevel(experiments_level)
    fh.setFormatter(formatter)
    experiments_logger.addHandler(fh)

def default_logger(debugging = False):
    if debugging:
        setup_logger(main_file_level=logging.DEBUG, quantumneat_level=logging.DEBUG, experiments_level=logging.DEBUG)
    else:
        setup_logger