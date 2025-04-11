from graphslam.graph import Graph


def main():
    g = Graph("./dataset/input_M3500_g2o.g2o")

    g.optimize()
    g.gif()
    g.show()


if __name__ == "__main__":
    main()
