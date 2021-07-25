import numpy
import cv2


class ShiftHorizontally(object):
    def __init__(
        self,
        shift: float
    ) -> None:
        super().__init__()