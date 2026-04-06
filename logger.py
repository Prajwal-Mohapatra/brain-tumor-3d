import logging
import sys
import os
from config import config

def get_logger(name):
    """
    Creates a logger that outputs to both the console and a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers multiple times if get_logger is called repeatedly
    if not logger.handlers:
        # 1. File Handler (Detailed log with timestamps)
        # Ensure logs directory exists (config.__init__ handles this, but safety first)
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        
        file_handler = logging.FileHandler(config.EXEC_LOG_FILE)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 2. Stream Handler (Console - cleaner output)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_formatter = logging.Formatter('%(message)s') 
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)
        
    return logger