from typing import Any

class Module(object):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)