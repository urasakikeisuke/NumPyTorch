import math

from typing import Optional, Tuple, Union

import numpy

# from numba import jit
from .unfold import Unfold
from .fold import Fold
from .module import Module


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]],
        padding: Union[int, Tuple[int]],
        weight_init: str = 'kaiming',
        weight_init_gain: float = numpy.sqrt(2)
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

        # if weight_init not in ['kaiming', 'xavier']:
        if weight_init not in ['kaiming']:
            raise ValueError(f'Expected weight init method is `kaiming` but got {weight_init}')

        self.in_channels: int = in_channels
        self.out_channels: int = out_channels

        if weight_init == 'kaiming':
            weight_init_gain = numpy.sqrt(2. / (self.in_channels * self.kernel_size[0] * self.kernel_size[1]))
            self.weight: numpy.ndarray = weight_init_gain * numpy.random.randn(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
            self.bias: numpy.ndarray = numpy.zeros(self.out_channels)

        self.unfold: Unfold = Unfold(self.kernel_size, self.stride, self.padding)
        self.fold: Fold = Fold(self.kernel_size, self.stride, self.padding)

        self.input_shape: Optional[Tuple[int, ...]] = None

        self.input: Optional[numpy.ndarray] = None
        self.unfolded: Optional[numpy.ndarray] = None
        self.unfolded_weight: Optional[numpy.ndarray] = None

        self.dW = None
        self.db = None
    
    # @jit
    def forward(self, input: numpy.ndarray) -> numpy.ndarray:
        N, _, H, W = self.input_shape = input.shape

        output_h: int = math.floor(
            (H + 2 * self.padding[0] - 1 * (self.kernel_size[0] - 1) - 1) / self.stride[0]
            + 1
        )
        output_w: int = math.floor(
            (W + 2 * self.padding[1] - 1 * (self.kernel_size[1] - 1) - 1) / self.stride[1]
            + 1
        )

        unfolded = self.unfold(input)
        unfolded_weight = self.weight.reshape(self.out_channels, -1).T

        output: numpy.ndarray = numpy.dot(unfolded, unfolded_weight) + self.bias
        output = output.reshape(N, output_h, output_w, -1).transpose(0, 3, 1, 2)

        self.input = input
        self.unfolded = unfolded
        self.unfolded_weight = unfolded_weight

        return output

    def backward(self, dout: numpy.ndarray) -> numpy.ndarray:
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        self.db: numpy.ndarray = numpy.sum(dout, axis=0)
        self.dW: numpy.ndarray = numpy.dot(self.unfolded.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        d_unfolded = numpy.dot(dout, self.unfolded_weight.T)
        dx = self.fold(d_unfolded, self.input_shape)

        return dx
