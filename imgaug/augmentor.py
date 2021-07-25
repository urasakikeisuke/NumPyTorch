import numpy
import cv2

from utils import _get_affine_matrix


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
        shift: float = None
    ) -> None:
        super().__init__()

        if isinstance(shift, float):
            self.shift: float = shift
        else:
            raise TypeError(f'Expected `shift` type is `float` but got {type(shift).__name__}')

    def exec(self, input: numpy.ndarray) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()
        dam: numpy.ndarray = sam.copy()
        dam[:, 0] += self.shift

        transform: numpy.ndarray = cv2.getAffineTransform(sam, dam)

        return cv2.warpAffine(input, transform, (input.shape[1], input.shape[0]))


class ShiftVertically(object):
    def __init__(
        self,
        shift: float = None
    ) -> None:
        super().__init__()

        if isinstance(shift, float):
            self.shift: float = shift
        else:
            raise TypeError(f'Expected `shift` type is `float` but got {type(shift).__name__}')

    def exec(self, input: numpy.ndarray) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()
        dam: numpy.ndarray = sam.copy()
        dam[:, 1] += self.shift

        transform: numpy.ndarray = cv2.getAffineTransform(sam, dam)

        return cv2.warpAffine(input, transform, (input.shape[1], input.shape[0]))


class Scalling(object):
    def __init__(
        self,
        scale: float = None
    ) -> None:
        super().__init__()

        if isinstance(scale, float):
            self.scale: float = scale
        else:
            raise TypeError(f'Expected `scale` type is `float` but got {type(scale).__name__}')

    def exec(self, input: numpy.ndarray) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()
        dam: numpy.ndarray = sam.copy()
        dam * self.scale

        transform: numpy.ndarray = cv2.getAffineTransform(sam, dam)

        return cv2.warpAffine(input, transform, (input.shape[1] * 2, input.shape[0] * 2))


class ShearHorizontally(object):
    def __init__(
        self,
        shear: float = None
    ) -> None:
        super().__init__()

        if isinstance(shear, float):
            self.shear: float = shear
        else:
            raise TypeError(f'Expected `shear` type is `float` but got {type(shear).__name__}')

    def exec(self, input: numpy.ndarray) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()
        dam: numpy.ndarray = sam.copy()
        dam[:, 0] += (self.shear / input.shape[0] * (input.shape[0] - sam[:, 1])).astype(numpy.float32)

        transform: numpy.ndarray = cv2.getAffineTransform(sam, dam)

        return cv2.warpAffine(input, transform, (input.shape[1], input.shape[0]))


class ShearVertically(object):
    def __init__(
        self,
        shear: float = None
    ) -> None:
        super().__init__()

        if isinstance(shear, float):
            self.shear: float = shear
        else:
            raise TypeError(f'Expected `shear` type is `float` but got {type(shear).__name__}')

    def exec(self, input: numpy.ndarray) -> numpy.ndarray:
        sam: numpy.ndarray = _get_affine_matrix()
        dam: numpy.ndarray = sam.copy()
        dam[:, 1] += (self.shear / input.shape[1] * (input.shape[1] - sam[:, 0])).astype(numpy.float32)

        transform: numpy.ndarray = cv2.getAffineTransform(sam, dam)

        return cv2.warpAffine(input, transform, (input.shape[1], input.shape[0]))
