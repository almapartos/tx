import numpy as np

class MCI:
    """ Matrix cross interpolation. Construct the core based on the target. """

    # TODO: Make the routine.

    def __init__(self, maxBondDimension=None, target=None):
        self.maxBondDimension = maxBondDimension # maybe get this directly from the tt instead
        self.target = target