import numpy as np
import warnings


class TensorTrain:
    """Create a tensor train object.

    : param length:
        The length (number of cores) of the train.
    : param local_dim:
        The local dimension of each mode.
    : param max_bond_dim:
        The maximum allowed bond dimension.
    : param target:
        The target function.

    """

    TRIVIAL = np.ones(shape=(1,))

    def __init__(
        self,
        length=None,
        local_dim=None
    ):
        self.length = length
        self.local_dim = local_dim
        self.initialize_train()

    def initialize_train(self):
        """Generate a tensor train of appropriate dimensions with trivial entries."""
        self.cores = [np.ones(shape=(1, self.local_dim, 1))] * self.length

    def contract_cores(self, left_core, right_core):
        product = np.einsum("ijk, klm -> ijlm", left_core, right_core, optimize=True)  # cores must have compatible dimensions, i.e. left_core: (alpha_1, d, alpha_2) and right_core: (alpha_2, d, alpha_3) -> (alpha_1, d, d, alpha_3)
        return product

    def contract_all(self):
        """Contract the internal bonds connecting all TT-cores, from left to right, to return the original tensor. Warning: This is very much not recommended for larger dimensions."""
        left = None
        for i in range(self.length + 1):
            left = self.TRIVIAL if i == 0 else left
            right = self.cores[i] if i < self.length else self.TRIVIAL
            if i < self.length:
                left = np.einsum("...i, ijk -> ...jk", left, right, optimize=True)
            else:
                left = np.einsum("...i, i -> ...", left, right, optimize=True)
        return left

    def evaluate(self, index_config):
        """Evaluate the (scalar) tensor element corresponding to the input index configuration, param : index_config, i.e. f(sigma)."""
        if len(index_config) != self.length:
            raise TypeError(f"Number of indices, {len(index_config)}, does not match the length of the tensor train ({self.length}).")
        if any(x > self.local_dim for x in index_config):
            raise TypeError(
                f"The values in the configuration array must not exceed the local dimension, {self.local_dim}.")
        left = None
        for i, index in enumerate(index_config):
            left = self.TRIVIAL if i == 0 else left
            right = self.cores[i]
            left = np.einsum("i, ij -> j", left, right[:, index, :], optimize=True)
        return left.item()

    def sum_all(self):
        """Sum over all configurations."""
        left = None
        for i in range(self.length):
            left = self.TRIVIAL if i == 0 else left
            right = self.cores[i]
            left = np.einsum("i, ijk -> k", left, right, optimize=True)
        return left.item()

    def set_core(self, new_core, index):
        """Set the core param :: new_core at site index param :: index."""
        if index > self.length - 1:
            raise TypeError(f"The tensor train is too short to contain an index {index}!")
        if new_core.shape[1] != self.local_dim:
            warnings.warn(f"The new core has local dimension {new_core.shape[1]}, which is different from the native (uniform) dimension {self.local_dim}. This may affect operations.")
        self.cores[index] = new_core

tr = TensorTrain(5, 4)
print(tr.sum_all())
print(tr.contract_all().shape)
print(tr.evaluate([1, 0, 0, 0, 0]))

