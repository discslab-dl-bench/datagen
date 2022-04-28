import numpy as np

if __name__ == "__main__":

    for i in range(300):
        casename = "case_" + f"{i:05}"  # Pad case number with zeros to be 5 digits long

        dim = np.random.randint(35, 75)
        m = np.random.rand(dim, dim, dim)
        with open(f"./data/{casename}.npy", "wb") as f:
            np.save(f, m)
