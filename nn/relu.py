from .module import Module

import numpy

class ReLU(Module):
    def __init__(
        self,
        inplace: bool = True,
    ) -> None:
        super().__init__()

        self.inplace: bool = inplace
        self.mask: numpy.ndarray = None

    def forward(self, input: numpy.ndarray) -> numpy.ndarray:
        self.mask = (input <= 0)

        if self.inplace:
            output: numpy.ndarray = input
        else:
            output = input.copy()
        output[self.mask] = 0

        return output

    def backward(self, dout: numpy.ndarray) -> numpy.ndarray:
        dout[self.mask] = 0
        dx: numpy.ndarray = dout

        return dx