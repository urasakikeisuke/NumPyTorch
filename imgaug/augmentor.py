from typing import Union
import numpy
import cv2

from .utils import _get_affine_matrix
from .core import RandomGenerator


class Identity(object):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def exec(self, input: numpy.ndarray) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()

        transform: numpy.ndarray = cv2.getAffineTransform(sam, sam)

        return cv2.warpAffine(input, transform, (input.shape[1], input.shape[0]))


class ShiftHorizontally(object):
    def __init__(
        self,
        shift: Union[int, float, RandomGenerator]
    ) -> None:
        super().__init__()

        if isinstance(shift, (int, float)):
            self.shift: float = shift
        elif isinstance(shift, RandomGenerator):
            self.shift = shift
        else:
            raise TypeError(f'Expected `shift` type is `int` or `float` but got {type(shift).__name__}')

    def exec(self, input: numpy.ndarray) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()
        dam: numpy.ndarray = sam.copy()
        self.shift = self.shift() if isinstance(self.shift, RandomGenerator) else self.shift
        dam[:, 0] += self.shift

        transform: numpy.ndarray = cv2.getAffineTransform(sam, dam)

        return cv2.warpAffine(input, transform, (input.shape[1], input.shape[0]))


class ShiftVertically(object):
    def __init__(
        self,
        shift: Union[int, float, RandomGenerator]
    ) -> None:
        super().__init__()

        if isinstance(shift, (int, float)):
            self.shift: float = shift
        elif isinstance(shift, RandomGenerator):
            self.shift = shift
        else:
            raise TypeError(f'Expected `shift` type is `int` or `float` but got {type(shift).__name__}')

    def exec(self, input: numpy.ndarray) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()
        dam: numpy.ndarray = sam.copy()
        self.shift = self.shift() if isinstance(self.shift, RandomGenerator) else self.shift
        dam[:, 1] += self.shift

        transform: numpy.ndarray = cv2.getAffineTransform(sam, dam)

        return cv2.warpAffine(input, transform, (input.shape[1], input.shape[0]))


class ShearHorizontally(object):
    def __init__(
        self,
        shear: Union[int, float, RandomGenerator]
    ) -> None:
        super().__init__()

        if isinstance(shear, (int, float)):
            self.shear: float = shear
        elif isinstance(shear, RandomGenerator):
            self.shear = shear
        else:
            raise TypeError(f'Expected `shear` type is `int` or `float` but got {type(shear).__name__}')

    def exec(self, input: numpy.ndarray) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()
        dam: numpy.ndarray = sam.copy()
        self.shear = self.shear() if isinstance(self.shear, RandomGenerator) else self.shear
        dam[:, 0] += (self.shear / input.shape[0] * (input.shape[0] - sam[:, 1])).astype(numpy.float32)

        transform: numpy.ndarray = cv2.getAffineTransform(sam, dam)

        return cv2.warpAffine(input, transform, (input.shape[1], input.shape[0]))


class ShearVertically(object):
    def __init__(
        self,
        shear: Union[int, float, RandomGenerator]
    ) -> None:
        super().__init__()

        if isinstance(shear, (int, float)):
            self.shear: float = shear
        elif isinstance(shear, RandomGenerator):
            self.shear = shear
        else:
            raise TypeError(f'Expected `shear` type is `int` or `float` but got {type(shear).__name__}')

    def exec(self, input: numpy.ndarray) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()
        dam: numpy.ndarray = sam.copy()
        self.shear = self.shear() if isinstance(self.shear, RandomGenerator) else self.shear
        dam[:, 1] += (self.shear / input.shape[1] * (input.shape[1] - sam[:, 0])).astype(numpy.float32)

        transform: numpy.ndarray = cv2.getAffineTransform(sam, dam)

        return cv2.warpAffine(input, transform, (input.shape[1], input.shape[0]))
