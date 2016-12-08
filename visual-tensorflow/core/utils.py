# Utilities
# TODO: make this into a class?

import os
import yaml


def read_yaml(path):
    """Return parsed yaml"""
    with open(path, 'r') as f:
        try:
            return yaml.load(f)
        except yaml.YAMLError as e:
            print e