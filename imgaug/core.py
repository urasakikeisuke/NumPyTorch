from random import shuffle, random, uniform, randrange, seed
from typing import Any, List, Tuple, Union

import numpy


class RandomGenerator(object):
    def __init__(
        self,
        range: Tuple[Union[int, float], Union[int, float]],
        seed_: float = None
    ) -> None:
        super().__init__()

        if seed_ is not None:
            seed(seed_)

        if isinstance(range, tuple):
            if len(range) != 2:
                self.range: Tuple[Union[int, float], Union[int, float]] = range
            else:
                raise ValueError(f'Expected `range` length is 2 but got {len(range)}')
        else:
            raise TypeError(f'Expected `range` type is `tuple`but got {type(range).__name__}')
    
    def __call__(self):
        if isinstance(self.range[0], int) and isinstance(self.range[1], int):
            return randrange(self.range[0], self.range[1])
        else:
            return uniform(self.range[0], self.range[1])


class SomeOf(object):
    def __init__(
        self,
        p: float,
        children: List[Any],
        random_order: bool = False,
        input_order: str = 'CHW',
    ) -> None:
        super().__init__()

        if p is None:
            self.p: float = 1.

        if isinstance(p, float):
            self.p = p

        if random_order:
            shuffle(children)
        
        self.children: List[Any] = children

        self.input_order: str = input_order

    def __call__(self, input: numpy.ndarray) -> numpy.ndarray:
        if self.input_order == 'CHW':
            input = input.transpose([1, 2, 0])
        elif self.input_order == 'HWC':
            input = input
        elif self.input_order == 'HW':
            input = input.reshape([input.shape[0], input.shape[1], 1])
        else:
            raise ValueError(f'Expected `input_order` is `CHW` or `CHW` or `HW` but got {self.input_order}')

        for augmentor in self.children:
            if self.p > random():
                input = augmentor.exec(input)

        if self.input_order == 'CHW':
            output = input.reshape([1, input.shape[0], input.shape[1]])
        elif self.input_order == 'HWC':
            output = input.reshape([input.shape[0], input.shape[1], 1])
        elif self.input_order == 'HW':
            output = input

        return output


if __name__ == '__main__':
    raise NotImplementedError