# Data scaling and generation

`data_scaling.py`: dataset scaling code

`data_generation.py`: dataset generation code

`scaling.sh`: examples to run `data_scaling.py`

`generation.sh`: examples to run `data_generation.py` 

## Data scaling

To scale the dataset of image segmetation, run

```shell

python3 data_scaling.py \
    --input_path <path to the input files> \
    --output_path <path to store the output files> \
    --desired_size <the desired size of the dataset (B)> \
    --workload imseg

```

To scale the dataset of bert, run

```shell

python3 data_scaling.py \
    --input_path <path to the input files> \
    --output_path <path to store the output files> \
    --desired_size <the desired size of the dataset (B)> \
    --workload bert

```

To scale the dataset of drlm, run

```shell

python3 data_scaling.py \
    --input_path <path to the input files> \
    --output_path <path to store the output files> \
    --desired_size <the desired size of the dataset (B)> \
    --workload dlrm

```




## Data generation

### Image segmentation

The size of the generated image is randomly determined as follows:

`size1:` the number of images stacked, a random integer between 128 and 471

`size2:` the shape of a single image, a random integer between 186 and 444

To generate data for image segmentation workload, run 

```shell

python3 data_generation.py \
    --output_path <path to store the output files> \
    --desired_size <the desired size of the dataset (B)> \
    --workload imseg

```

### Bert

The number of instances stored in each TFrecord is a random integer between 195754 and 260461.

To generate data for bert workload, run 

```shell

python3 data_generation.py \
    --output_path <path to store the output files> \
    --desired_size <the desired size of the dataset (B)> \
    --workload bert

```

### DLRM

There are three data formats for dlrm training set, `txt`, `npz`, and  `bin`. 

To generate raw text data for dlrm workload, run

```shell

python3 data_generation.py \
    --output_path <path to store the output files> \
    --desired_size <the desired size of the dataset (B)> \
    --workload dlrm \
    --data_format text

```

To generate npz data for dlrm workload, run

```shell

python3 data_generation.py \
    --output_path <path to store the output files> \
    --desired_size <the desired size of the dataset (B)> \
    --workload dlrm \
    --data_format npz

```

To generate binary data for dlrm workload, run

```shell

python3 data_generation.py \
    --output_path <path to store the output files> \
    --desired_size <the desired size of the dataset (B)> \
    --workload dlrm \
    --data_format bin

