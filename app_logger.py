"""Given the config file, we can create a logger instance, which will be used 
across the whole project.
"""

import os
import logging
import logging.config
import logging.handlers

import yaml

def setup_logging(
    default_config_path='config/logging.yaml',
    default_level=logging.INFO,
    env_config='CAMELYON16_LOG'
):
    """Setup logging configuration

    """
    config_path = default_config_path
    value = os.getenv(env_config, None)
    if value:
        config_path = value
    if os.path.exists(config_path):
        with open(config_path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
		

def get_logger(
	module_name
):
    """Use this function to get the logger instance

    """
    setup_logging()
    logger = logging.getLogger(module_name)
    return logger
