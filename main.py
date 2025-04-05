from graphslam.graph import Graph


def main():
    g = Graph("./dataset/input_INTEL_g2o.g2o")

    g.optimize()
    g.plot()


if __name__ == "__main__":
    main()
