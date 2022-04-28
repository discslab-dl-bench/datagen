import argparse
import pathlib
import os.path
import json
import psutil
import scale_dataset

LAUNCH_SCRIPT = ""
DATASET_PATH = ""
SYSTEM_MEMORY = psutil.virtual_memory()[0]

def main(experiments):
    for experiment, settings in experiments.items():

        # If we want to scale the dataset size
        if settings['scale_data']:
            if not settings['scale_mem']:
                print(f"{experiment} needs data scaling to reach {settings['data_to_mem_ratio']}x of system memory")
                desired_size = settings['data_to_mem_ratio'] * SYSTEM_MEMORY
            else:
                memory_limit = settings['target_mem_size']
                print(f"{experiment} needs data scaling to reach {settings['data_to_mem_ratio']}x of allowed process memory ({memory_limit} B)")
                desired_size = settings['data_to_mem_ratio'] * memory_limit

            scale_dataset.scale_ds_and_validate(DATASET_PATH, desired_size)
        # Else if we only want to limit the process memory
        elif settings['scale_mem']:
            memory_limit = settings['target_mem_size']

            pass
        # Base case, simply run the baseline experiment
        else:
            pass
        
        # Run the experiment by running the LAUNCH_SCRIPT

        # Wait for its completion then collect the traces and archive them, possibly scp them somewhere to free space


            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Schedule and run the experiments"
    )
    parser.add_argument(
        "experiments_file", type=pathlib.Path, help="Path to the experiments.json file"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.experiments_file):
        print("Invalid experiments file")
        exit(-1)
    
    try:
        experiments = json.load(open(args.experiments_file))
    except Exception as e:
        print("Experiments file is not valid JSON")
        print(e)
        exit(-1)

    main(experiments)
