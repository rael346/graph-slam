import numpy as np
import numpy.typing as npt
import typing

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from graphslam.edge_odometry import Edge
from graphslam.se2 import SE2


class Graph:
    def __init__(self, poses: list[SE2], edges: list[Edge]) -> None:
        self.poses = poses
        self.edges = edges

    @classmethod
    def from_g2o(cls, path: str) -> typing.Self:
        poses: list[SE2] = []
        edges: list[Edge] = []
        with open(path, "r") as file:
            for line in file.readlines():
                line = line.split()
                section = line[0]
                if section == "VERTEX_SE2":
                    pose = SE2.from_g2o(line[2:])
                    poses.append(pose)

                if section == "EDGE_SE2":
                    edge = Edge.from_g2o(line[1:], poses)
                    edges.append(edge)
        return cls(poses, edges)

    def calc_b(self) -> npt.NDArray[np.float64]:
        len_b = len(self.poses) * SE2.COMPACT_DIM

        b = np.zeros(len_b, dtype=np.float64)
        for edge in self.edges:
            i, j = edge.pose_ids
            b_i, b_j = edge.calc_b()
            if i != 0:
                b[i * SE2.COMPACT_DIM : (i + 1) * SE2.COMPACT_DIM] += b_i

            if j != 0:
                b[j * SE2.COMPACT_DIM : (j + 1) * SE2.COMPACT_DIM] += b_j

        return b

    def calc_H(self) -> lil_matrix:
        # Incremently load the Hessian contributions to a dict.
        # This is way faster than adding to the sparse matrix directly
        # Note: storing in a dict then assign to a lil_matrix is way
        # faster than using a dok_matrix (by about 4 times) since
        # H will need to be converted to CSR/CSC format later for spsolve()
        # and dok_matrix is very inefficient in this conversion operation
        H_dict: dict[tuple[int, int], npt.NDArray[np.float64]] = {}
        for edge in self.edges:
            for r, c, contrib in edge.calc_H():
                H_dict[(r, c)] = contrib + H_dict.get(
                    (r, c), np.zeros((SE2.COMPACT_DIM, SE2.COMPACT_DIM))
                )

        len_b = len(self.poses) * SE2.COMPACT_DIM
        H = lil_matrix((len_b, len_b), dtype=np.float64)

        # fixed the first node
        H[0 : SE2.COMPACT_DIM, 0 : SE2.COMPACT_DIM] = np.eye(SE2.COMPACT_DIM)

        for (r, c), contrib in H_dict.items():
            if r == 0 and c == 0:
                continue
            H[r : r + SE2.COMPACT_DIM, c : c + SE2.COMPACT_DIM] = contrib

        return H

    def calc_chi2(self) -> float:
        return np.sum([edge.calc_chi2() for edge in self.edges])

    def optimize(self, max_iter=20):
        prev_chi2 = float("inf")

        for optimize_i in range(max_iter):
            curr_chi2 = self.calc_chi2()
            print(f"Iter {optimize_i} | chi2 val {curr_chi2}")
            if np.isclose(prev_chi2, curr_chi2):
                break
            prev_chi2 = curr_chi2

            H, b = self.calc_H(), self.calc_b()
            dx = spsolve(H.tocsr(), -b)

            for i in range(len(self.poses)):
                self.poses[i].update(
                    dx[i * SE2.COMPACT_DIM : (i + 1) * SE2.COMPACT_DIM]
                )

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        for e in self.edges:
            e.plot(ax)

        for p in self.poses:
            p.plot(ax)
        plt.show()
