import logging

# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(levelname)s: %(message)s')

def init_logger(logfile, log_level=logging.INFO):
    '''
    Initializes the logger.
        Parameters:
            logfile (str):                  Path to the output logfile.
            log_level (enum):               Logger log level. Default is DEBUG.
    '''
    # disables matplotlib logging to avoid spam
    logging.getLogger('matplotlib').disabled = True
    logging.getLogger('matplotlib.font_manager').disabled = True

    logger = logging.getLogger('hea')

    # Set log format time - log level : msg
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

    # Attach a file logger
    file_handler = logging.FileHandler(filename=logfile, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Attach a console logger
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Set log level
    logger.setLevel(log_level)

    return logger