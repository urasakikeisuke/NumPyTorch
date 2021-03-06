from .module import Module

import numpy

from numba import jit


class Dropout(Module):
    def __init__(
        self,
        p: float = 0.5
    ) -> None:
        super().__init__()

        if isinstance(p, float):
            self.p: float = p
        else:
            raise TypeError(f'Expected `p` type is `float` but got `{type(p).__name__}`')

        self.mask: numpy.ndarray = None

    @jit
    def forward(self, input: numpy.ndarray, training: bool) -> numpy.ndarray:
        if training:
            self.mask = numpy.random.rand(*input.shape) > self.p
            return input * self.mask
        else:
            return input * (1. - self.p)

    @jit
    def backward(self, dout: numpy.ndarray) -> numpy.ndarray:
        return dout * self.mask