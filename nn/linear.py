from .module import Module

import numpy


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(in_features, int):
            self.in_features: float = in_features
        else:
            raise TypeError(f'Expected `in_features` type is `int` but got `{type(in_features).__name__}`')

        if isinstance(out_features, int):
            self.out_features: float = out_features
        else:
            raise TypeError(f'Expected `out_features` type is `int` but got `{type(out_features).__name__}`')

        self.weight: numpy.ndarray = numpy.random.uniform(-numpy.sqrt(self.in_features), numpy.sqrt(self.in_features), (in_features, out_features))

        if bias:
            self.bias: numpy.ndarray = numpy.random.uniform(-numpy.sqrt(self.in_features), numpy.sqrt(self.in_features), out_features)
        else:
            self.bias = numpy.zeros(out_features)

        self.input: numpy.ndarray = None
        self.input_shape: numpy.ndarray = None

        self.dW: numpy.ndarray = None
        self.db: numpy.ndarray = None
    
    def forward(self, input: numpy.ndarray) -> numpy.ndarray:
        self.input_shape = input.shape
        input = input.reshape(input.shape[0], -1)
        self.input = input

        output: numpy.ndarray = numpy.dot(self.input, self.weight) + self.bias

        return output

    def backward(self, dout: numpy.ndarray) -> numpy.ndarray:
        dx: numpy.ndarray = numpy.dot(dout, self.weight.T)
        self.dW = numpy.dot(self.input.T, dout)
        self.db = numpy.sum(dout, axis=0)
        
        dx = dx.reshape(*self.input_shape)
        return dx
