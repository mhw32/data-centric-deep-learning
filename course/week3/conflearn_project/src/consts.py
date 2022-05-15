from os.path import realpath, dirname, join

# helpful paths we might want to access!
SRC_DIR = realpath(dirname(__file__))
ROOT_DIR = realpath(join(SRC_DIR, '..'))
DATA_DIR = realpath(join(ROOT_DIR, 'data'))
