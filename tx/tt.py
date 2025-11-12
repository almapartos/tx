import numpy as np
import mx

class TensorTrain:
    """Create a tensor train object.

    : param length:
        The length (number of cores) of the train.
    : param maxBondDimension:
        The maximum allowed bond dimension.
    : param target:
        The target function.

    """

    # TODO: implement cores

    def __init__(self, length=None, maxBondDimension=None, target=None):
        self.length = length
        self.maxBondDimension = maxBondDimension
        self.target = target

    def build(self):
        # TODO: build the train from mx.py
        pass



