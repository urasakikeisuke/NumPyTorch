"""unfold.py"""
import math
from typing import Tuple, Union

import numpy
from .module import Module

from numba import jit


class Unfold(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int]],
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
    
    @jit
    def forward(self, input: numpy.ndarray) -> numpy.ndarray:
        N, C, H, W = input.shape

        output_h: int = math.floor(
            (H + 2 * self.padding[0] - 1 * (self.kernel_size[0] - 1) - 1) / self.stride[0]
            + 1
        )
        output_w: int = math.floor(
            (W + 2 * self.padding[1] - 1 * (self.kernel_size[1] - 1) - 1) / self.stride[1]
            + 1
        )

        input = numpy.pad(input, [(0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])], 'constant')

        unfolded: numpy.ndarray = numpy.zeros((N, C, self.kernel_size[0], self.kernel_size[1], output_h, output_w))

        for y in range(self.kernel_size[0]):
            y_max = y + self.stride[0] * output_h
            for x in range(self.kernel_size[1]):
                x_max = x + self.stride[1] * output_w
                unfolded[:, :, y, x, :, :] = input[:, :, y:y_max:self.stride[0], x:x_max:self.stride[1]]

        unfolded = unfolded.transpose(0, 4, 5, 1, 2, 3).reshape(N * output_h * output_w, -1)

        return unfolded
