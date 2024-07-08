from os.path import realpath, dirname, join

# Helpful paths we might want to access! 
# Nothing you need to do here.
SRC_DIR = realpath(dirname(__file__))
ROOT_DIR = realpath(join(SRC_DIR, '..'))
ARTIFACT_DIR = realpath(join(ROOT_DIR, 'artifacts'))
DATA_DIR = realpath(join(ROOT_DIR, 'data'))
LOG_DIR = realpath(join(ARTIFACT_DIR, 'log'))
SCRIPT_DIR = realpath(join(ROOT_DIR, 'scripts'))
CONFIG_DIR = realpath(join(ROOT_DIR, 'configs'))
