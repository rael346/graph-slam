from __future__ import annotations
from typing import Self
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray

from graphslam.utils import norm_angle


class SE2:
    def __init__(self, t: NDArray[np.float64], theta: float):
        self.t = t
        self.theta = norm_angle(theta)

    @classmethod
    def from_mat(cls, matrix: NDArray[np.float64]) -> Self:
        return cls(matrix[0:2, 2], np.atan2(matrix[1, 0], matrix[0, 0]))

    @classmethod
    def from_g2o(cls, line: list[str]) -> Self:
        t = np.array([float(s) for s in line[:2]], dtype=np.float64)
        theta = float(line[2])
        return cls(t, theta)

    @property
    def R(self) -> NDArray[np.float64]:
        return np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ],
            dtype=np.float64,
        )

    @property
    def delta_R(self) -> NDArray[np.float64]:
        return np.array(
            [
                [-np.sin(self.theta), -np.cos(self.theta)],
                [np.cos(self.theta), -np.sin(self.theta)],
            ],
            dtype=np.float64,
        )

    def to_mat(self) -> NDArray[np.float64]:
        R = self.R
        return np.array(
            [
                [R[0, 0], R[0, 1], self.t[0]],
                [R[1, 0], R[1, 1], self.t[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    def to_compact(self) -> NDArray[np.float64]:
        return np.array([self.t[0], self.t[1], self.theta], dtype=np.float64)

    @staticmethod
    def J_sub_p1(_: SE2, p2: SE2) -> NDArray[np.float64]:
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
    def J_sub_p2(p1: SE2, p2: SE2) -> NDArray[np.float64]:
        dt_dt = -p2.R.T
        dt_dtheta = p2.delta_R.T @ (p1.t - p2.t)
        return np.array(
            [
                [dt_dt[0, 0], dt_dt[0, 1], dt_dtheta[0]],
                [dt_dt[1, 0], dt_dt[1, 1], dt_dtheta[1]],
                [0, 0, -1],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def J_add_delta_x(p: SE2) -> NDArray[np.float64]:
        return np.array(
            [
                [np.cos(p.theta), -np.sin(p.theta), 0],
                [np.sin(p.theta), np.cos(p.theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    def plot(self, ax: Axes):
        ax.plot(self.t[0], self.t[1], color="darkorange", marker="o", markersize=3)

    def __add__(self, other: SE2) -> SE2:
        return SE2(self.R @ other.t + self.t, self.theta + other.theta)

    def __sub__(self, other: SE2) -> SE2:
        return SE2(other.R.T @ (self.t - other.t), self.theta - other.theta)

    def __str__(self) -> str:
        return f"PoseSE2(t=[{self.t[0]}, {self.t[1]}], theta={self.theta})"
