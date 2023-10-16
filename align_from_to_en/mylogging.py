import logging

logger = logging.getLogger()
sh = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.setLevel(logging.INFO)


def add_filehandler(file):
    global logger
    fh = logging.FileHandler(file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
def get_logger(name):
    global logger
    return logger.getChild(name)