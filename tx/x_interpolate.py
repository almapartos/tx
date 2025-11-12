import numpy as np
import tt

class CrossInterpolate:
    """ Create the tensor cross interpolation of the target.

        : param target:
            The function to be approximated.

        """

    def __init__(self, target=None):
        self.ci = tt.TensorTrain(target=target)
        self.ci.build()

