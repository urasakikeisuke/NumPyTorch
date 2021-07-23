from typing import Any, Optional, Tuple, Union
from .module import Module

import numpy


class BatchNorm2d(Module):
    def __init__(
        self,
        num_features: int,
        momentum: float = 0.9,
        running_mean: Optional[numpy.ndarray] = None,
        running_var: Optional[numpy.ndarray] = None,
    ) -> None:
        super().__init__()
        self.num_features: int = num_features
        self.gamma: Optional[numpy.ndarray] = None
        self.beta: Optional[numpy.ndarray] = None
        self.momentum: Union[Any, numpy.number[Any]] = momentum
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.cnt: int = 0

        self.running_mean: Optional[numpy.ndarray] = running_mean
        self.running_var: Optional[numpy.ndarray] = running_var  
        
        self.batch_size: Optional[int] = None
        self.xc: Optional[numpy.ndarray] = None
        self.std: Optional[numpy.ndarray] = None
        self.dgamma: Optional[numpy.ndarray] = None
        self.dbeta: Optional[Union[numpy.number[Any], numpy.ndarray]] = None

    def forward(self, input: numpy.ndarray, train_flg: bool = True) -> numpy.ndarray:
        self.input_shape = input.shape
        if input.ndim != 2:
            N, C, H, W = self.input_shape
            input = input.reshape(N, -1)

        if self.cnt == 0:
            self.gamma = numpy.ones(self.num_features * H * W)
            self.beta = numpy.zeros(self.num_features * H * W)
        self.cnt += 1

        out: numpy.ndarray = self.__forward(input, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, input: numpy.ndarray, train_flg: bool) -> numpy.ndarray:
        if self.running_mean is None:
            N, D = input.shape
            self.running_mean = numpy.zeros(D)
            self.running_var = numpy.zeros(D)
                        
        if train_flg:
            mu = input.mean(axis=0)
            xc = input - mu
            var = numpy.mean(xc**2, axis=0)
            std = numpy.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = input.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = input - self.running_mean
            xn = xc / ((numpy.sqrt(self.running_var + 10e-7)))
            
        out: numpy.ndarray = self.gamma * xn + self.beta
        return out

    def backward(self, dout: numpy.ndarray) -> numpy.ndarray:
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx: numpy.ndarray = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout: numpy.ndarray) -> numpy.ndarray:
        dbeta = dout.sum(axis=0)
        dgamma = numpy.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -numpy.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = numpy.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx  