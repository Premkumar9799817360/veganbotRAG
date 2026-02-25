import logging 
import os 
from logging.handlers import RotatingFileHandler 

LOG_DIR = 'logs'
LOG_FILE = 'app.log'

def setup_logger(name: str="veganbot"):

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    logger  = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:

        file_handler = RotatingFileHandler(
            os.path.join(LOG_DIR,LOG_FILE),
            maxBytes = 5*1024*1024,  # 5MB SIZE
            backupCount=10
        )

        console_hander = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        file_handler.setFormatter(formatter)

        console_hander.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_hander)

    
    return logger