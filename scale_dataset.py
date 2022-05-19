import os.path
import argparse
import pathlib
import shutil
import numpy as np
import time

# Hard coded number of cases in the kits19 dataset used by the MLPerf Image Segmentation Reference Implementation
NUM_CASES = 210

# Hard coded list of evaluation cases 
EVAL_CASES = [
    "00000", "00003", "00005", "00006", "00012", "00024", "00034", "00041", "00044", "00049", 
    "00052", "00056", "00061", "00065", "00066", "00070", "00076", "00078", "00080", "00084", 
    "00086", "00087", "00092", "00111", "00112", "00125", "00128", "00138", "00157", "00160", 
    "00161", "00162", "00169", "00171", "00176", "00185", "00187", "00189", "00198", "00203", 
    "00206", "00207",
]

SEED = 1234

def get_original_dataset_size(path):
    total_size = 0
    for filename in os.listdir(path):
        case_num = int(filename.split("_")[1])   # expects files with name case_xxxxx_x.npy
        if case_num >= NUM_CASES:     
            return total_size
        fullpath = os.path.join(path, filename)
        filesize = os.path.getsize(fullpath)
        total_size += filesize
    return total_size


def get_dataset_size(path):
    total_size = 0
    for filename in os.listdir(path):
        fullpath = os.path.join(path, filename)
        filesize = os.path.getsize(fullpath)
        total_size += filesize
    return total_size


def reset_to_original_ds(path):
    """
        Deletes all cases not part of the original dataset.
        In the case of image segmentation that's all cases > NUM_CASES.
    """
    for filename in os.listdir(path):
        case_num = int(filename.split("_")[1])   # expects files with name case_xxxxx.npy
        if case_num >= NUM_CASES:    
            fullpath = os.path.join(path, filename)
            os.remove(fullpath)

    cases = os.listdir(path)

    # If there are less than NUM_CASES file, we had previsouly reduced the dataset size
    # by moving cases to another directory. Move these cases back.
    print(f"Num casefiles present: {len(cases)}")
    if len(cases) < 2 * NUM_CASES:
        move_path = pathlib.Path(path).parent / "moved_cases"
        print(f"We have previously lowered dataset size by moving cases in {move_path}")

        if not os.path.isdir(move_path):
            print(f"ERROR: Can't find the moved cases in '{move_path}'. Aborting.")
            exit(-1)
        
        for moved_case in os.listdir(move_path):
            moved_case = pathlib.Path(move_path) / moved_case
            shutil.move(moved_case, path)
            print(f"Restored {moved_case}")
        
        print(f"Restored all original cases to {path}\n")
        assert len(os.listdir(path)) == 2 * NUM_CASES   # for x and y parts
        



def scale_dataset(path, desired_size):

    reset_to_original_ds(path)

    original_size = get_original_dataset_size(path)

    if (desired_size < original_size):
        print(f"Reducing dataset size")
        move_path = os.path.join(os.path.dirname(path), "moved_cases")
        pathlib.Path(move_path).mkdir(parents=True, exist_ok=True)

        print(f"Extra test cases will be moved to {move_path}")

        rng = np.random.default_rng(seed=SEED)
        dataset_size = get_dataset_size(path)

        # As we reduce the number of cases, we want to keep a similar ratio
        # of evaluation vs training cases. Since we're removing uniformly
        # the ratio should be stable?

        while dataset_size > desired_size:
            
            casenum_to_mv = rng.integers(low=0, high=NUM_CASES)
            casenum_to_mv = f"{casenum_to_mv:05}"
            case_to_mv = "case_" + casenum_to_mv
            file_to_mv = os.path.join(path, f"{case_to_mv}_x.npy")

            # Check if case already moved
            if not os.path.isfile(file_to_mv):
                continue
            
            # Move both the x and y files for the given case
            shutil.move(file_to_mv, move_path)
            file_to_mv = os.path.join(path, f"{case_to_mv}_y.npy")
            shutil.move(file_to_mv, move_path)

            dataset_size = get_dataset_size(path)
            print(f"Moved {case_to_mv}, current size: {dataset_size}")

    else:
        casenum_to_cpy = 0
        newcase_counter = NUM_CASES
        
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
            casenum_to_cpy = (casenum_to_cpy + 1) % NUM_CASES
            newcase_counter += 1

        num_cases_created = newcase_counter - NUM_CASES
        print(f"Created {num_cases_created} new cases to reach total size of {get_dataset_size(path)} B")


def validate_dataset(path):
    """
        Ensure numpy can open all the cases after scaling up
        and that all eval cases are still present.
    """

    for filename in os.listdir(path):
        fullpath = os.path.join(path, filename)
        try:
            test = np.load(fullpath)
        except Exception as e:
            print(f"Error opening {fullpath}")
            print(e)
    print("Opened all cases successfully!")

    num_eval_cases = 0
    for case in EVAL_CASES:
        casefile = os.path.join(path, f"case_{case}_x.npy")
        if os.path.isfile(casefile):
            num_eval_cases += 1
    num_cases = len(os.listdir(path))
    print(f"Final evaluation to training case ratio is {num_eval_cases / num_cases} vs {len(EVAL_CASES)/NUM_CASES} orginally.")


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
        print("ERROR: dataset_path must be a directory!")
        exit(-1)

    if args.final_size < 0:
        print("ERROR: Final size cannot be negative!")
        exit(-1)

    scale_dataset(args.dataset_path, args.final_size)
    validate_dataset(args.dataset_path)

    final_size = get_dataset_size(args.dataset_path)
    print(f"Final dataset size: {final_size} vs requested {args.final_size}")