import math

from typing import Optional, Tuple, Union

import numpy

from numba import jit
from .unfold import Unfold
from .fold import Fold
from .module import Module


class Pooling(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = (2, 2),
        padding: Union[int, Tuple[int, int]] = (0, 0),
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            raise TypeError(f'Expected `kernel_size` type is `int` or `tuple` but got `{type(kernel_size).__name__}`')

        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.stride = stride
        else:
            raise TypeError(f'Expected `stride` type is `int` or `tuple` but got `{type(stride).__name__}`')

        if isinstance(padding, int):
            self.padding = (padding, padding)
        elif isinstance(padding, tuple):
            self.padding = padding
        else:
            raise TypeError(f'Expected `padding` type is `int` or `tuple` but got `{type(padding).__name__}`')

        self.unfold: Unfold = Unfold(self.kernel_size, self.stride, self.padding)
        self.fold: Fold = Fold(self.kernel_size, self.stride, self.padding)

        self.input: Optional[numpy.ndarray] = None
        self.arg_maxed: Optional[numpy.ndarray] = None
    
    @jit
    def forward(self, input: numpy.ndarray) -> numpy.ndarray:
        self.input = input

        N, C, H, W = self.input_shape = input.shape

        output_h: int = math.floor(
            (H + 2 * self.padding[0] - 1 * (self.kernel_size[0] - 1) - 1) / self.stride[0]
            + 1
        )
        output_w: int = math.floor(
            (W + 2 * self.padding[1] - 1 * (self.kernel_size[1] - 1) - 1) / self.stride[1]
            + 1
        )

        unfolded: numpy.ndarray = self.unfold(input)
        unfolded = unfolded.reshape(-1, self.kernel_size[0] * self.kernel_size[1])

        self.arg_maxed = numpy.argmax(unfolded, axis=1)

        output = numpy.max(unfolded, axis=1)
        output = output.reshape(N, output_h, output_w, C).transpose(0, 3, 1, 2)

        return output

    @jit
    def backward(self, dout: numpy.ndarray) -> numpy.ndarray:
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size: float = self.kernel_size[0] * self.kernel_size[1]
        dmax: numpy.ndarray = numpy.zeros((dout.size, pool_size))
        dmax[numpy.arange(self.arg_maxed.size), self.arg_maxed.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        d_unfolded: numpy.ndarray = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = self.fold(d_unfolded, self.input_shape)
        
        return dx
