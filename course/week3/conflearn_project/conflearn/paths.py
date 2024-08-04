# Do not edit this file
from os.path import realpath, dirname, join

# helpful paths we might want to access!
CUR_DIR = realpath(dirname(__file__))
ROOT_DIR = realpath(join(CUR_DIR, '..'))
DATA_DIR = realpath(join(ROOT_DIR, 'data'))
LOG_DIR = realpath(join(ROOT_DIR, 'logs'))
CONFIG_DIR = realpath(join(ROOT_DIR, 'configs'))