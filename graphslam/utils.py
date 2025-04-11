import numpy as np


def norm_angle(angle: float):
    """
    Taken from https://samialperenakgun.com/blog/2022/05/scale_radian_range_python/
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi
    # return np.arctan2(np.sin(angle), np.cos(angle))
