import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

import imageio
from PIL import Image
from graphslam.edge_odometry import Edge
from graphslam.se2 import SE2


class Graph:
    def __init__(self, path: str) -> None:
        poses, edges = self.from_g2o(path)
        self.poses = poses
        self.edges = edges

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.num_iter = -1

    def from_g2o(self, path: str) -> tuple[list[SE2], list[Edge]]:
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
        return (poses, edges)

    def calc_b(self) -> NDArray[np.float64]:
        len_b = len(self.poses) * SE2.COMPACT_DIM

        b = np.zeros(len_b, dtype=np.float64)
        for edge in self.edges:
            i, j = edge.pose_ids
            b_i_contrib, b_j_contrib = edge.gradient()
            if i != 0:
                b[i * 3 : (i + 1) * 3] += b_i_contrib

            if j != 0:
                b[j * 3 : (j + 1) * 3] += b_j_contrib

        return b

    def calc_H(self) -> lil_matrix:
        len_b = len(self.poses) * SE2.COMPACT_DIM
        H_dict: dict[tuple[int, int], NDArray[np.float64]] = {}

        for edge in self.edges:
            for (r, c), contrib in edge.hessian():
                if r <= c:
                    H_dict[(r, c)] = contrib + H_dict.get((r, c), np.zeros((3, 3)))
                else:
                    H_dict[(c, r)] = contrib.T + H_dict.get((c, r), np.zeros((3, 3)))

        H = lil_matrix((len_b, len_b), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3, 3)

        for (r, c), contrib in H_dict.items():
            if r == 0 or c == 0:
                continue
            H[r : r + 3, c : c + 3] = contrib

            if r != c:
                H[c : c + 3, r : r + 3] = contrib.T

        return H

    def calc_chi2(self) -> float:
        return np.sum([edge.chi2() for edge in self.edges])

    def optimize(self, max_iter=20):
        curr_chi2 = self.calc_chi2()

        num_iter = 0
        for optimize_i in range(max_iter):
            print(f"Iter {optimize_i} | chi2 val {curr_chi2}")
            num_iter += 1

            H, b = self.calc_H(), self.calc_b()
            dx = spsolve(H.tocsr(), -b)

            for i in range(len(self.poses)):
                self.poses[i] += SE2(dx[i * 3 : i * 3 + 2], dx[i * 3 + 2])

            new_chi2 = self.calc_chi2()

            self.ax.clear()
            self.ax.set_axis_off()
            self.plot()
            self.fig.savefig(f"./results/iter_{optimize_i}.png")

            if np.isclose(curr_chi2, new_chi2):
                break

            curr_chi2 = new_chi2
        self.num_iter = num_iter

    def plot(self):
        for e in self.edges:
            e.plot(self.ax)

        for p in self.poses:
            p.plot(self.ax)

    def show(self):
        plt.show()

    def gif(self):
        frames = []

        for i in range(self.num_iter):
            frame = Image.open(f"results/iter_{i}.png")
            frame = np.asarray(frame)
            frames.append(np.array(frame))
        imageio.mimsave("results/graph.gif", frames, fps=2)
