"""fold.py"""
import math

from typing import Tuple, Union

import numpy
from .module import Module


class Fold(Module):
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

    def forward(self, input: numpy.ndarray, input_shape: Tuple[int, ...]) -> numpy.ndarray:
        N, C, H, W = input_shape

        output_h: int = math.floor(
            (H + 2 * self.padding[0] - 1 * (self.kernel_size[0] - 1) - 1) / self.stride[0]
            + 1
        )
        output_w: int = math.floor(
            (W + 2 * self.padding[1] - 1 * (self.kernel_size[1] - 1) - 1) / self.stride[1]
            + 1
        )

        input = input.reshape(N, output_h, output_w, C, self.kernel_size[0], self.kernel_size[1]).transpose(0, 3, 4, 5, 1, 2)

        output: numpy.ndarray = numpy.zeros((N, C, H + 2 * self.padding[0] + self.stride[0] - 1, W + 2 * self.padding[1] + self.stride[1] - 1))

        for y in range(self.kernel_size[0]):
            y_max = y + self.stride[0] * output_h
            for x in range(self.kernel_size[1]):
                x_max = x + self.stride[1] * output_w
                output[:, :, y:y_max:self.stride[0], x:x_max:self.stride[1]] += input[:, :, y, x, :, :]

        return output[:, :, self.padding[0]:H + self.padding[0], self.padding[1]:W + self.padding[1]]