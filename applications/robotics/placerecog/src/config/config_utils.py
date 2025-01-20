import configparser
from os.path import isfile, join
# from config import ROOT_DIR
ROOT_DIR = './'

def get_config(config_file, logger):
    config = configparser.ConfigParser()
    configFile = join(ROOT_DIR, config_file)
    logger.info(f'The path of the configure file is  {configFile}')
    assert isfile(configFile)
    config.read(configFile)
    return config

def update_config(config, updates):
    for k in updates:
        config[k] = updates[k]
    return config

def get_bool_from_config(config, key, default_value = False):
    if key in config:
        return config[key] == 'True'
    else:
        return default_value
