from .module import Module

import numpy

from numba import jit


class Softmax(Module):
    def __init__(
        self,
        dim: int = None,
    ) -> None:
        super().__init__()

        if isinstance(dim, (int, type(None))):
            if dim is None:
                self.dim: int = -1
            else:
                self.dim = dim
        else:
            raise TypeError(f'Expected `dim` type is `int` but got `{type(dim).__name__}`')

    @jit
    def forward(self, input: numpy.ndarray) -> numpy.ndarray:
        input = input - numpy.max(input, axis=self.dim, keepdims=True)
        return numpy.exp(input) / numpy.sum(numpy.exp(input), axis=self.dim, keepdims=True)
