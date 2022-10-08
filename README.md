#Data scaling and generation

`data_scaling.py`: dataset scaling code

`data_generation.py`: dataset generation code

##Data generation

To generate data for image segmentation workload, run 

```shell

python3 data_generation.py \
    --output_path <path to store the output files> \
    --desired_size <the desired size of the dataset (B)> \
    --workload imseg

```
