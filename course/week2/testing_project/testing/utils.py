import json
from dotmap import DotMap


def to_json(x, filepath):
  with open(filepath, 'w') as fp:
    json.dump(x, fp)


def from_json(filepath):
  with open(filepath, 'r') as fp:
    data = json.load(fp)
  return data


def load_config(config_path):
  return DotMap(from_json(config_path))
