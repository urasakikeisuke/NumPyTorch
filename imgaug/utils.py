import numpy


def _chw_to_hwc(input: numpy.ndarray) -> numpy.ndarray:
    return input.transpose([1, 2, 0])

def _get_affine_matrix() -> numpy.ndarray:
    return numpy.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], numpy.float32)


if __name__ == '__main__':
    raise NotImplementedError