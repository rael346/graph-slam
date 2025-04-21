from __future__ import annotations
from matplotlib.axes import Axes
import numpy as np
import numpy.typing as npt
import typing
import math


def norm_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class SE2:
    COMPACT_DIM = 3

    def __init__(self, t: npt.NDArray[np.float64], theta: float) -> None:
        self.t = t
        self.theta = norm_angle(theta)

    @classmethod
    def from_mat(cls, matrix: npt.NDArray[np.float64]) -> typing.Self:
        return cls(matrix[0:2, 2], np.atan2(matrix[1, 0], matrix[0, 0]))

    @classmethod
    def from_g2o(cls, line: list[str]) -> typing.Self:
        t = np.array([float(line[2]), float(line[3])], dtype=np.float64)
        theta = float(line[4])
        return cls(t, theta)

    @property
    def R(self) -> npt.NDArray[np.float64]:
        return np.array(
            [
                [math.cos(self.theta), -math.sin(self.theta)],
                [math.sin(self.theta), math.cos(self.theta)],
            ],
            dtype=np.float64,
        )

    @property
    def dR(self) -> npt.NDArray[np.float64]:
        return np.array(
            [
                [-math.sin(self.theta), -math.cos(self.theta)],
                [math.cos(self.theta), -math.sin(self.theta)],
            ],
            dtype=np.float64,
        )

    def to_mat(self) -> npt.NDArray[np.float64]:
        R = self.R
        return np.array(
            [
                [R[0, 0], R[0, 1], self.t[0]],
                [R[1, 0], R[1, 1], self.t[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    def to_compact(self) -> npt.NDArray[np.float64]:
        return np.array([self.t[0], self.t[1], self.theta], dtype=np.float64)

    @staticmethod
    def J_sub_p1(_: SE2, p2: SE2) -> npt.NDArray[np.float64]:
        dt_dt = p2.R.T
        return np.array(
            [
                [dt_dt[0, 0], dt_dt[0, 1], 0],
                [dt_dt[1, 0], dt_dt[1, 1], 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def J_sub_p2(p1: SE2, p2: SE2) -> npt.NDArray[np.float64]:
        dt_dt = -p2.R.T
        dt_dtheta = p2.dR.T @ (p1.t - p2.t)
        return np.array(
            [
                [dt_dt[0, 0], dt_dt[0, 1], dt_dtheta[0]],
                [dt_dt[1, 0], dt_dt[1, 1], dt_dtheta[1]],
                [0, 0, -1],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def J_add_delta_x(p: SE2) -> npt.NDArray[np.float64]:
        return np.array(
            [
                [math.cos(p.theta), -math.sin(p.theta), 0],
                [math.sin(p.theta), math.cos(p.theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    def __add__(self, other: SE2) -> SE2:
        return SE2(self.R @ other.t + self.t, self.theta + other.theta)

    def __sub__(self, other: SE2) -> SE2:
        return SE2(other.R.T @ (self.t - other.t), self.theta - other.theta)

    def update(self, delta: npt.NDArray[np.float64]) -> None:
        self.t += delta[0:2]
        self.theta = norm_angle(self.theta + delta[2])

    def __str__(self) -> str:
        return f"SE2(t=[{self.t[0]}, {self.t[1]}], theta={self.theta})"

    def plot(self, ax: Axes):
        ax.plot(self.t[0], self.t[1], color="darkorange", marker="o", markersize=3)
