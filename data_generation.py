import os
import time
import random
import pathlib
import argparse
import collections
import numpy as np
import tensorflow as tf

DLRM_CATEGORY_RANGES = [10000000,    38985,    17278,     7418,    20242,        3,
           7120,     1543,       63,  9999978,  2642264,   397262,
             10,     2208,    11931,      155,        4,      976,
             14, 10000000,  9832963, 10000000,   573162,    12969,
            108,       36]


def generate_dataset(output_path, desired_size, workload):
    if workload == "imseg":
        generate_data_imseg(output_path, desired_size)
    elif workload == "bert":
        generate_data_bert(output_path, desired_size)
    elif workload == "dlrm":
        generate_data_dlrm_bin(output_path, desired_size)


def generate_data_imseg(output_path, desired_size):
    # size range
    # [  1 471 444 444]
    # [  1 128 186 186]
    newcase_counter = 0
    total_size = 0
    while total_size < desired_size: 
        size1 = random.randint(128, 471)
        size2 = random.randint(186, 444)
        img = np.random.uniform(low=-2.340702, high=2.639792, size=(1, size1, size2, size2))
        mask = np.random.randint(0, 2, size=(1, size1, size2, size2))
        img = img.astype(np.float32)
        mask = mask.astype(np.uint8)
        fnx = f"{output_path}/case_{newcase_counter:05}_x.npy"
        fny = f"{output_path}/case_{newcase_counter:05}_y.npy"
        np.save(fnx, img)
        np.save(fny, mask)
        newcase_counter += 1
        total_size += os.path.getsize(fnx)
        total_size += os.path.getsize(fny)


def generate_data_bert(output_path, desired_size):
    newcase_counter = 27
    total_size = 0
    while total_size < desired_size: 
        t1 = time.time()
        output_file = f"{output_path}/part-{newcase_counter:05}-of-00500"
        # num_instances = int(195754 +  (260461 - 195754) * random.random()) # from counting # of lines in each part-00xxxx-of-00500
        num_instances = 260461
        writer = tf.io.TFRecordWriter(output_file)
        tf_examples = [create_instance() for _ in range(num_instances)]
        for tf_example in tf_examples:
            writer.write(tf_example.SerializeToString()) 
        # for i in range(num_instances):
        #     tf_example = create_instance()
        #     writer.write(tf_example.SerializeToString()) 
        writer.close()
        newcase_counter += 1
        total_size += os.path.getsize(output_file)
        t2 = time.time()
        print(f"time: {t2 - t1} s")
    

def generate_data_dlrm_bin(output_path, desired_size): 
    total_size = 0
    while total_size < desired_size: 
        fn = f"{output_path}/terabyte_processed_train.bin"
        print(f'Writing DLRM train file {fn}')
        with open(fn, 'ab') as output_file:
            num_instance = 6548660
            X_int = np.random.randint(2557264, size = (num_instance, 13))
            X_cat = np.random.randint(0, DLRM_CATEGORY_RANGES, size = (num_instance, 26))
            y = np.random.randint(2, size=num_instance)
            np_data = np.concatenate([y.reshape(-1, 1), X_int, X_cat], axis=1)
            np_data = np_data.astype(np.int32)
            output_file.write(np_data.tobytes())
        total_size += os.path.getsize(fn)
        print("\tcurrent size", total_size)

    # Make a dummy evaluation file, as the workload checks for its presence
    fn = f"{output_path}/terabyte_processed_test.bin"
    print(f'Writing DLRM eval file {fn}')

    with open(fn, 'ab') as output_file:
        num_instance = 6548660
        X_int = np.random.randint(2557264, size = (num_instance, 13))
        X_cat = np.random.randint(0, DLRM_CATEGORY_RANGES, size = (num_instance, 26))
        y = np.random.randint(2, size=num_instance)
        np_data = np.concatenate([y.reshape(-1, 1), X_int, X_cat], axis=1)
        np_data = np_data.astype(np.int32)
        output_file.write(np_data.tobytes())

    # Save the count metadata needed by the workload
    with open(f"{output_path}/day_fea_count.npz", "wb") as outfile:
        np.savez(outfile, counts=DLRM_CATEGORY_RANGES)



def generate_data_dlrm_text(output_path, desired_size):
    while total_size < desired_size: 
        f_output = open(f"{output_path}/train.txt", "a")
        for i in range(100):
            label = [str(random.randint(0, 1))]
            numerical = [str(random.randint(0, 1000)) for _ in range(13)]
            categorical = ['%08x' % random.randrange(16**8) for _ in range(26)]
            text = label + numerical + categorical
            line = " ".join(text) + "\n"
            f_output.write(line)
        f_output.close()
        total_size = os.path.getsize(f_output)
    

def generate_data_dlrm_npz(output_path, desired_size):
    newcase_counter = 0
    total_size = 0
    while total_size < desired_size: 
        num_instance = 6548660
        X_int = np.random.randint(2557264, size = (num_instance, 13))
        X_cat = np.random.randint(8831335, size = (num_instance, 26))
        y = np.random.randint(2, size=num_instance)
        fn = f'{output_path}/day_{newcase_counter}_reordered.npz'
        np.savez(fn, X_int=X_int, X_cat=X_cat, y=y)
        newcase_counter += 1
        total_size += os.path.getsize(fn)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature

def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_instance():
    max_seq_length=512
    max_predictions_per_seq=76  

    # length of the sentence
    id_length = int(max_seq_length * random.random())
    mask_length = int(0.15 * id_length)
    seg_length = int(id_length * random.random())
 
    # randomly generate data
    input_ids = [int(30522 * random.random()) for _ in range(id_length)] 
    input_mask = [1 for _ in range(id_length)] 
    segment_ids = [0 for _ in range(seg_length)] + [1 for _ in range(id_length - seg_length)]
    masked_lm_positions = random.choices(list(range(id_length)), k=mask_length)
    masked_lm_ids = [input_ids[i] for i in masked_lm_positions]
    masked_lm_weights = [1.0 for _ in range(mask_length)]
    next_sentence_label = random.randint(0,1)

    # padding
    input_ids += [0 for _ in range(max_seq_length - id_length)]
    input_mask += [0 for _ in range(max_seq_length - id_length)]
    segment_ids += [0 for _ in range(max_seq_length - id_length)]
    masked_lm_positions += [0 for _ in range(max_predictions_per_seq - mask_length)]
    masked_lm_ids += [0 for _ in range(max_predictions_per_seq - mask_length)]
    masked_lm_weights += [0.0 for _ in range(max_predictions_per_seq - mask_length)]

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-path", required=True, type=pathlib.Path)
    parser.add_argument("-s", "--size", required=True, type=float)
    parser.add_argument("-w", "--workload", dest="workload", type=str, required=True, choices=['imseg', 'bert', 'dlrm'])
    args = parser.parse_args()

    if args.size < 0:
        print("ERROR: Desired size cannot be negative!")
        exit(-1)

    generate_dataset(args.output_path, args.size, args.workload)
