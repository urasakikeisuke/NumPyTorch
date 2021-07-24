import numpytorch
from typing import Dict, Optional, Tuple
import numpy


class Adam(object):
    def __init__(
        self,
        net,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8
    ):
        self.net = net

        self.lr: float = lr

        self.beta1: float = betas[0]
        self.beta2: float = betas[1]

        self.eps: float = eps

        self.cnt: int = 0

        self.m: Optional[Dict[str, numpy.ndarray]] = None
        self.v: Optional[Dict[str, numpy.ndarray]] = None

        self.params: Dict[str, numpy.ndarray] = {}
        self.grads: Dict[str, numpy.ndarray] = {}

        self.compat_table: Dict[str, str] = {
            'weight': 'dW',
            'bias': 'db',
            'gamma': 'dgamma',
            'beta': 'dbeta',
        }
        
    def step(self) -> None:
        self._get_params()
        self._get_grads()

        if self.m is None:
            self.m, self.v = {}, {}
            for key, value in self.params.items():
                self.m[key] = numpy.zeros_like(value)
                self.v[key] = numpy.zeros_like(value)
        
        self.cnt += 1
        lr_t  = self.lr * numpy.sqrt(1.0 - self.beta2 ** self.cnt) / (1.0 - self.beta1 ** self.cnt)

        for key_type in self.params.keys():
            key, type = key_type.split('-')
            self.m[key_type] += (1 - self.beta1) * (self.grads[f'{key}-{self.compat_table[type]}'] - self.m[key_type])
            self.v[key_type] += (1 - self.beta2) * (self.grads[f'{key}-{self.compat_table[type]}'] ** 2 - self.v[key_type])
            
            self.params[key_type] -= lr_t * self.m[key_type] / (numpy.sqrt(self.v[key_type]) + self.eps)

        self._set_params()
        self._set_grads()

    def zero_grad(
        self,
        set_to_none: bool = False,
    ) -> None:
        self._get_grads()

        for key, value in self.grads.items():
            if set_to_none:
                self.grads[key] = None
            else:
                zeros: numpy.ndarray = numpy.zeros_like(value)
                self.grads[key] = zeros

        self._set_grads()

    def _get_params(self) -> None:
        _attr = vars(self.net)
        for key, child in _attr.items():
            if hasattr(child, 'weight'):
                self.params[f'{key}-weight'] = child.weight
            if hasattr(child, 'bias'):
                self.params[f'{key}-bias'] = child.bias
            if hasattr(child, 'gamma'):
                self.params[f'{key}-gamma'] = child.gamma
            if hasattr(child, 'beta'):
                self.params[f'{key}-beta'] = child.beta

    def _set_params(self) -> None:
        for key, value in self.params.items():
            name, type = key.split('-')
            if type == 'weight':
                _obj = getattr(self.net, f'{name}')
                setattr(_obj, 'weight', value)
            if type == 'bias':
                _obj = getattr(self.net, f'{name}')
                setattr(_obj, 'bias', value)
            if type == 'gamma':
                _obj = getattr(self.net, f'{name}')
                setattr(_obj, 'gamma', value)
            if type == 'beta':
                _obj = getattr(self.net, f'{name}')
                setattr(_obj, 'beta', value)

    def _get_grads(self) -> None:
        _attr = vars(self.net)
        for key, child in _attr.items():
            if hasattr(child, 'dW'):
                self.grads[f'{key}-dW'] = child.dW
            if hasattr(child, 'db'):
                self.grads[f'{key}-db'] = child.db
            if hasattr(child, 'dgamma'):
                self.grads[f'{key}-dgamma'] = child.dgamma
            if hasattr(child, 'dbeta'):
                self.grads[f'{key}-dbeta'] = child.dbeta

    def _set_grads(self) -> None:
        for key, value in self.grads.items():
            name, type = key.split('-')
            if type == 'dW':
                _obj = getattr(self.net, f'{name}')
                setattr(_obj, 'dW', value)
            if type == 'db':
                _obj = getattr(self.net, f'{name}')
                setattr(_obj, 'db', value)
            if type == 'dgamma':
                _obj = getattr(self.net, f'{name}')
                setattr(_obj, 'dgamma', value)
            if type == 'dbeta':
                _obj = getattr(self.net, f'{name}')
                setattr(_obj, 'dbeta', value)
