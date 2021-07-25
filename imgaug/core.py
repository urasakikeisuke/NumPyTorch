import numpy


def _chw_to_hwc(input: numpy.ndarray) -> numpy.ndarray:
    return input.transpose([1, 2, 0])

