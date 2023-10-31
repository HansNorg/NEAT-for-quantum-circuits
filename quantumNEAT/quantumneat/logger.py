import logging
    
def setup_logger(filename = "main", test_level = logging.DEBUG, file_level = logging.INFO, console_level = logging.ERROR, name = "quantumNEAT", mode ="a", test_mode="a"):
    print("setup_logger")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(min(file_level, console_level))
    test_logger = logging.getLogger(f"test_{name}")
    test_logger.setLevel(test_level)
    
    sh = logging.StreamHandler()
    sh.setLevel(console_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    test_logger.addHandler(sh)

    try:
        fh = logging.FileHandler("logs/"+filename+".log", mode=mode, encoding = "utf-8")    
    except FileNotFoundError:
        fh = logging.FileHandler("quantumneat/logs/"+filename+".log", mode=mode, encoding = "utf-8")
    fh.setLevel(file_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    try:
        fh = logging.FileHandler("logs/"+filename+"_test.log", mode=test_mode, encoding = "utf-8")    
    except FileNotFoundError:
        fh = logging.FileHandler("quantumneat/logs/"+filename+"_test.log", mode=test_mode, encoding = "utf-8")
    fh.setLevel(test_level)
    fh.setFormatter(formatter)
    test_logger.addHandler(fh)