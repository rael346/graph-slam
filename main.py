from slam.graph import Graph


def main():
    g = Graph.from_g2o("./dataset/input_M3500b_g2o.g2o")

    g.plot()
    g.optimize(40)
    g.plot()


if __name__ == "__main__":
    main()
