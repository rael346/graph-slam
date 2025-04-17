from typing import Self
from matplotlib.axes import Axes
from numpy.typing import NDArray
from typing import cast
import numpy as np

from graphslam.se2 import SE2


class Edge:
    def __init__(
        self,
        pose_ids: tuple[int, int],
        info: NDArray[np.float64],
        z: SE2,
        poses: list[SE2],
    ) -> None:
        self.pose_ids = pose_ids
        self.info = info
        self.z = z
        self.poses = poses

    @classmethod
    def from_g2o(cls, line: list[str], poses: list[SE2]) -> Self:
        pose_ids = (int(line[0]), int(line[1]))
        z = SE2.from_g2o(line[2:5])
        i11, i12, i13, i22, i23, i33 = [float(s) for s in line[5:]]
        info = np.array(
            [
                [i11, i12, i13],
                [i12, i22, i23],
                [i13, i23, i33],
            ],
            np.float64,
        )
        return cls(pose_ids, info, z, poses)

    @property
    def e(self) -> SE2:
        p_i = self.poses[self.pose_ids[0]]
        p_j = self.poses[self.pose_ids[1]]
        return (p_j - p_i) - self.z

    def chi2(self) -> float:
        # the type for np.matmul couldn't infer the
        # single return value case so this is just to make the inference a bit better
        e_compact = self.e.to_compact()
        val = cast(np.float64, e_compact.T @ self.info @ e_compact)
        return val

    def jacobians(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        p_i = self.poses[self.pose_ids[0]]
        p_j = self.poses[self.pose_ids[1]]
        J_error_wrt_pred = SE2.J_sub_p1(p_j - p_i, self.z)

        A_ij = J_error_wrt_pred @ SE2.J_sub_p2(p_j, p_i)
        B_ij = J_error_wrt_pred @ SE2.J_sub_p1(p_j, p_i)
        return (A_ij, B_ij)

    def gradient(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        A_ij, B_ij = self.jacobians()
        e_compact = self.e.to_compact()
        return (
            A_ij.T @ self.info @ e_compact,
            B_ij.T @ self.info @ e_compact,
        )

    def hessian(self) -> list[tuple[int, int, NDArray[np.float64]]]:
        A_ij, B_ij = self.jacobians()
        i, j = self.pose_ids
        i = i * SE2.COMPACT_DIM
        j = j * SE2.COMPACT_DIM

        H_ii = A_ij.T @ self.info @ A_ij
        H_ij = A_ij.T @ self.info @ B_ij
        H_jj = B_ij.T @ self.info @ B_ij

        return [(i, i, H_ii), (i, j, H_ij), (j, i, H_ij.T), (j, j, H_jj)]

    def plot(self, ax: Axes):
        out_id, in_id = self.pose_ids
        out_pos, in_pos = self.poses[out_id].t, self.poses[in_id].t
        ax.plot([out_pos[0], in_pos[0]], [out_pos[1], in_pos[1]], color="dodgerblue")
