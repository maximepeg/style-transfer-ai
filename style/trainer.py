import torch
import numpy

import torch.nn as nn

class Trainer:
    def __init__(self, config):
        self.epoch = config["epoch"]
