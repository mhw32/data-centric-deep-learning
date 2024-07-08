from os.path import realpath, dirname, join

ROOT_DIR = realpath(join(dirname(__file__), '..'))
SCRIPT_DIR = realpath(join(ROOT_DIR, 'scripts'))
DATA_DIR = realpath(join(ROOT_DIR, 'data'))
CONFIG_DIR = realpath(join(ROOT_DIR, 'configs'))
CHECKPOINT_DIR = realpath(join(ROOT_DIR, 'checkpoints'))
LOG_DIR = realpath(join(ROOT_DIR, 'logs'))
IMAGE_DIR = realpath(join(ROOT_DIR, 'images'))