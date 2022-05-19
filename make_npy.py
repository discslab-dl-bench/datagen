import numpy as np
import pathlib

if __name__ == "__main__":

    pathlib.Path("./data").mkdir(parents=True, exist_ok=True)

    for i in range(210):
        casename = "case_" + f"{i:05}"  # Pad case number with zeros to be 5 digits long

        dim = np.random.randint(35, 75)
        m = np.random.rand(dim, dim, dim)

        for xy in ["x", "y"]:
            with open(f"./data/{casename}_{xy}.npy", "wb") as f:
                np.save(f, m)
