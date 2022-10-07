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

def generate_dataset(output_path, desired_size, workload, format):
    if workload == "imseg":
        generate_data_imseg(output_path, desired_size)
    elif workload == "bert":
        generate_data_bert(output_path, desired_size)
    elif workload == "dlrm":
        if format == "raw":
            generate_data_dlrm_raw(output_path, desired_size)
        if format == "npz":
            generate_data_dlrm_npz(output_path, desired_size)
        if format == "bin":
            generate_data_dlrm_bin(output_path, desired_size)


def generate_data_imseg(output_path, desired_size):
    # fix size
    newcase_counter = 0
    while os.path.getsize(output_path) < desired_size: 
        img = np.random.uniform(low=-2.340702, high=2.639792, size=(1, 190, 392, 392) )
        mask = np.random.randint(0, 2, size=(1, 190, 392, 392) )
        np.save(f"{output_path}/case_{newcase_counter:05}_x.npy", img)
        np.save(f"{output_path}/case_{newcase_counter:05}_y.npy", mask)
        newcase_counter += 1


def generate_data_bert(output_path, desired_size):
    newcase_counter = 0
    while os.path.getsize(output_path) < desired_size: 
        output_file = f"part-{newcase_counter:05}-of-00500"
        num_instances = random.randint(195754, 260461) # from counting # of lines in each part-00xxxx-of-00500
        writer = tf.io.TFRecordWriter(output_file)

        for i in range(num_instances):
            tf_example = create_instance()
            writer.write(tf_example.SerializeToString())
        
        writer.close()
        newcase_counter += 1


def generate_data_dlrm_raw(output_path, desired_size):
    while os.path.getsize(output_path) < desired_size: 
        label = [str(random.randint(0, 1))]
        numerical = [str(random.randint(0, 1000)) for _ in range(13)]
        categorical = ['%08x' % random.randrange(16**8) for _ in range(26)]
        text = label + numerical + categorical
        line = " ".join(text) + "\n"
        f_output = open(output_path, "w")
        f_output.write(line)
        f_output.close()


def generate_data_dlrm_npz(output_path, desired_size):
    newcase_counter = 0
    while os.path.getsize(output_path) < desired_size: 
        num_instance = 6548660
        X_int = np.random.randint(2557264, size = (num_instance, 13))
        X_cat = np.random.randint(8831335, size = (num_instance, 26))
        y = np.random.randint(2, size=num_instance)
        np.savez(f'{output_path}/day_{newcase_counter}_reordered.npz', X_int=X_int, X_cat=X_cat, y=y)
        newcase_counter += 1

def generate_data_dlrm_bin(output_path, desired_size):
    newcase_counter = 0
    while os.path.getsize(output_path) < desired_size: 
        with open(f"{output_path}/preprocessed_{newcase_counter}.bin", 'wb') as output_file:
            num_instance = 10
            X_int = np.random.randint(2557264, size = (num_instance, 13))
            X_cat = np.random.randint(8831335, size = (num_instance, 26))
            y = np.random.randint(2, size=num_instance)
            np_data = np.concatenate([y.reshape(-1, 1), X_int, X_cat], axis=1)
            np_data = np_data.astype(np.int32)
            output_file.write(np_data.tobytes())

        # np.savez(f'{output_path}/day_{newcase_counter}_reordered.npz', X_int=X_int, X_cat=X_cat, y=y)
        newcase_counter += 1


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
    id_length = random.randint(0, max_seq_length)
    mask_length = int(0.15 * id_length)
    seg_length = random.randint(0, id_length)
 
    # randomly generate data
    input_ids = [random.randint(0, 30522) for _ in range(id_length)] 
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
    output_path = "./generation"
    desired_size = "100M"
    workload = "dlrm"
    format = "bin"
    generate_dataset(output_path, desired_size, workload, format)
