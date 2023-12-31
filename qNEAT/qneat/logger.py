import logging

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    
class QNEATLogger(metaclass=Singleton):
    
    def __init__(self, filename, file_level = logging.INFO, console_level = logging.ERROR, name = "qNEAT", mode ="a"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(min(file_level, console_level))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler("logs/"+filename+".log", mode=mode, encoding = "utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setLevel(console_level)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

def main():
    QNEATLogger("default_logger")