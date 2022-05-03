import sys
import json
import psutil
import pathlib
import os.path
import argparse
import scale_dataset
import subprocess as sp

SYSTEM_MEMORY = psutil.virtual_memory()[0]

def main(experiments, launch_script, dataset_path, output_dir):
    for experiment, settings in experiments.items():

        
        num_gpus = str(settings['num_gpus'])

        # If we want to scale the dataset size
        if settings['scale_data']:
            if not settings['scale_mem']:
                print(f"{experiment} needs data scaling to reach {settings['data_to_mem_ratio']}x of system memory")
                desired_size = settings['data_to_mem_ratio'] * SYSTEM_MEMORY
            else:
                memory_limit = settings['target_mem_size']
                print(f"{experiment} needs data scaling to reach {settings['data_to_mem_ratio']}x of allowed process memory ({memory_limit} B)")
                desired_size = settings['data_to_mem_ratio'] * memory_limit

            scale_dataset.scale_ds_and_validate(dataset_path, desired_size)

        # Else if we only want to limit the process memory
        elif settings['scale_mem']:
            memory_limit = settings['target_mem_size']
            pass

        # Base case, simply run the baseline experiment
        else:
            print("Baseline settings")
            if scale_dataset.get_original_dataset_size(dataset_path) != scale_dataset.get_dataset_size(dataset_path):
                print("Reverting to original dataset size")
                scale_dataset.reset_to_original_ds(dataset_path)
        
        # Run the experiment by running the LAUNCH_SCRIPT
        print(f"Launching {experiment}: {settings}")

        pathlib.Path("logs/").mkdir(parents=True, exist_ok=True)
        
        with open(f'logs/{experiment}.log', 'wb') as f: 
            process = sp.Popen([launch_script, output_dir, num_gpus, experiment], stdout=sp.PIPE, stderr=sp.STDOUT)
            for c in iter(lambda: process.stdout.read(1), b''): 
                sys.stdout.buffer.write(c)
                f.write(c)


            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Schedule and run the experiments"
    )
    parser.add_argument(
        "launch_script", type=pathlib.Path, help="Path to the training launch script"
    )
    parser.add_argument(
        "dataset_path", type=pathlib.Path, help="Path to the dataset"
    )
    parser.add_argument(
        "experiments_file", type=pathlib.Path, help="Path to the experiments.json file"
    )
    parser.add_argument(
        "output_dir", type=pathlib.Path, help="Directory where traces will be stored"
    )
    args = parser.parse_args()


    if not os.path.isfile(args.launch_script):
        print("Invalid launch script")
        exit(-1)
    if not os.path.isdir(args.dataset_path):
        print("Invalid path to dataset (must be a directory)")
        exit(-1)
    if not os.path.isfile(args.experiments_file):
        print("Invalid experiments file")
        exit(-1)
    if not os.path.isdir(args.output_dir):
        print("Invalid output directory")
        exit(-1)
    
    try:
        experiments = json.load(open(args.experiments_file))
    except Exception as e:
        print("Experiments file is not valid JSON")
        print(e)
        exit(-1)


    main(experiments, args.launch_script, args.dataset_path, args.output_dir)
