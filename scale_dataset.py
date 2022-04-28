import os.path
import argparse
import pathlib
import shutil
import numpy as np


def get_original_dataset_size(path):
    total_size = 0
    for filename in os.listdir(path):
        case_num = int(filename.split("_")[1])   # expects files with name case_xxxxx_x.npy
        if case_num >= 300:     # Original dataset has 300 cases
            return total_size
        fullpath = os.path.join(path, filename)
        filesize = os.path.getsize(fullpath)
        total_size += filesize

    return total_size


def get_files(path):
    files = []
    for filename in os.listdir(path):
        files.append(os.path.join(path, filename))
    return files


def get_dataset_size(path):
    total_size = 0
    for filename in os.listdir(path):
        fullpath = os.path.join(path, filename)
        filesize = os.path.getsize(fullpath)
        total_size += filesize
        # print(f"{filename}: {filesize} B")

    return total_size


def reset_to_original_ds(path, max_num=300):
    """
        Deletes all cases not part of the original dataset.
        In the case of image segmentation that's all cases > 300.
    """
    for filename in os.listdir(path):
        case_num = int(filename.split("_")[1])   # expects files with name case_xxxxx.npy
        if case_num >= max_num:     # Original dataset has 300 cases
            fullpath = os.path.join(path, filename)
            os.remove(fullpath)



def scale_dataset(path, desired_size):

    original_size = get_original_dataset_size(path)

    if (desired_size < original_size):
        print(f"Can't scale dataset to below it's original size of {original_size} B")
        exit(-1)

    reset_to_original_ds(path)

    casenum_to_cpy = 0
    newcase_counter = 300
    
    while get_dataset_size(path) < desired_size:
        case_to_cpy = "case_" + f"{casenum_to_cpy:05}"
        newcase_prefix = "case_" + f"{newcase_counter:05}" 

        pair_size = 0
        # Copy both x and y for the case
        for xy in ["_x", "_y"]:
            file_to_cpy = os.path.join(path, case_to_cpy + xy + ".npy")
            newcase = os.path.join(path, newcase_prefix + xy + ".npy")
            pair_size += os.path.getsize(file_to_cpy)
            shutil.copy(file_to_cpy, newcase)

        print(f"Copied {case_to_cpy} to {newcase_prefix} ({pair_size} B)")
        casenum_to_cpy = (casenum_to_cpy + 1) % 300
        newcase_counter += 1

    num_cases_created = newcase_counter - 300
    print(f"Created {num_cases_created} new cases to reach total size of {get_dataset_size(path)} B")


def validate_dataset(path):
    """
        Make sure numpy can open all the cases after scaling up
    """
    for filename in os.listdir(path):
        fullpath = os.path.join(path, filename)
        try:
            test = np.load(fullpath)
        except Exception as e:
            print(f"Error opening {fullpath}")
            print(e)
    print("Opened all cases successfully!")


def scale_ds_and_validate(path, size):
    if not os.path.isdir(path):
        print("dataset_path must be a directory!")
        return
    scale_dataset(path, size)
    validate_dataset(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Increase the size of the image segmentation dataset to the given size (B). Uses a naive copying mechanism."
    )
    parser.add_argument(
        "dataset_path", type=pathlib.Path, help="Path to the data containing directory."
    )
    parser.add_argument("final_size", type=float, help="Final size of the dataset to reach.")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        print("dataset_path must be a directory!")
        exit(-1)

    scale_dataset(args.dataset_path, args.final_size)
    validate_dataset(args.dataset_path)
