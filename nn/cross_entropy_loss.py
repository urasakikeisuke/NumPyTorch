from .module import Module
from .softmax import Softmax

import numpy

from numba import jit


class CrossEntropyLoss(Module):
    def __init__(self) -> None:
        super().__init__()

        self.loss: numpy.ndarray = None
        self.softmaxed: numpy.ndarray = None
        self.target: numpy.ndarray = None

        self.softmax = Softmax()

    @jit
    def forward(self, input: numpy.ndarray, target: numpy.ndarray) -> numpy.ndarray:
        self.target = target
        self.softmaxed: numpy.ndarray = self.softmax(input)

        if self.softmaxed.ndim == 1:
            self.target = self.target.reshape(1, self.target.size)
            self.softmaxed = self.softmaxed.reshape(1, self.softmaxed.size)

        if self.target.size == self.softmaxed.size:
            self.target = self.target.argmax(axis=1)

        N: int = self.softmaxed.shape[0]
        eps: float = 1e-7
        self.loss = -numpy.sum(numpy.log(self.softmaxed[numpy.arange(N), self.target] + eps)) / N

        return self.loss
    
    @jit
    def backward(self, dout: numpy.ndarray = 1) -> numpy.ndarray:
        N: int = self.target.shape[0]

        if self.target.size == self.softmaxed.size:
            dx: numpy.ndarray = (self.softmaxed - self.target) / N
        else:
            dx = self.softmaxed.copy()
            dx[numpy.arange(N), self.target] -= 1
            dx = dx / N
        
        return dx