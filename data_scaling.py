import os, random
import argparse
import pathlib
import shutil
from unicodedata import category
from click import pass_obj
import numpy as np
import tensorflow as tf
import random
import collections


def get_dataset_size(path):
    total_size = 0
    for filename in os.listdir(path):
        fullpath = os.path.join(path, filename)
        filesize = os.path.getsize(fullpath)
        total_size += filesize
    return total_size

def scale_dataset(input_path, output_path, desired_size, workload, aug=False):
    if workload == "imseg":
        scale_imseg(input_path, output_path, desired_size, aug=False)
    elif workload == "bert":
        scale_bert(input_path, output_path, desired_size, aug=False)
    elif workload == "dlrm":
        scale_dlrm(input_path, output_path, desired_size, aug=False)


def scale_imseg(input_path, output_path, desired_size, aug=False):
    NUM_CASES = 210
    casenum_to_cpy = 0
    newcase_counter = 0
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    while get_dataset_size(output_path) < desired_size:
        case_to_cpy = "case_" + f"{casenum_to_cpy:05}"
        newcase_prefix = "case_" + f"{newcase_counter:05}" 

        pair_size = 0
        # Copy both x and y for the case
        for xy in ["_x", "_y"]:
            file_to_cpy = os.path.join(input_path, case_to_cpy + xy + ".npy")
            newcase = os.path.join(output_path, newcase_prefix + xy + ".npy")
            pair_size += os.path.getsize(file_to_cpy)
            shutil.copy(file_to_cpy, newcase)

        print(f"Copied {case_to_cpy} to {newcase_prefix} ({pair_size} B)")
        casenum_to_cpy = (casenum_to_cpy + 1) % NUM_CASES
        newcase_counter += 1


def scale_bert(input_path, output_path, desired_size, aug=False):
    NUM_CASES = 500
    casenum_to_cpy = 0
    newcase_counter = 0
        
    while get_dataset_size(output_path) < desired_size:
        case_to_cpy = f"part-{casenum_to_cpy:05}-of-00500"
        newcase_name = f"part-{newcase_counter:05}-of-00500"
        file_to_cpy = os.path.join(input_path, case_to_cpy)
        newcase = os.path.join(output_path, newcase_name)
        shutil.copy(file_to_cpy, newcase)
        casenum_to_cpy = (casenum_to_cpy + 1) % NUM_CASES
        newcase_counter += 1


def scale_dlrm(input_path, output_path, desired_size, aug=False):
    f_input = open(input_path, "r")

    while os.path.getsize(output_path) < desired_size: 

        for i in range(100):
            line = f_input.readline()

            if line == "": # reopen the file to read from the start
                f_input.close()
                f_input.open()

            f_output = open(output_path, "w")
            f_output.write(line)
            f_output.close()
    
    f_input.close()



