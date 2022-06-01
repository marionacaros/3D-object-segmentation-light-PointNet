# Object Segmentation of Cluttered Airborne LiDAR Point Clouds
3D segmentation on LiDAR data with Deep Learning

![plot](./doc/framework.png)

### Installation
The code has been tested with Python 3.7, [Pytorch](https://pytorch.org/) v1.8, CUDA 11.6  on Ubuntu 20.04. \n
You may also need to install pdal library to transform HAS data into HAG.
```
pip install pdal
```

### Usage
Execute code from main directory

#### Preprocessing
```
python data_proc/1_get_windows.py --LAS_files_path path/LAS_files/here
```

```
bash compute_pdal_bash.sh  # get HeighAboveGround
```
or execute the following code for all .LAS files
```
pdal translate $input_file $output_file hag_nn --writers.las.extra_dims="HeightAboveGround=float32"
```

```
python data_proc/2_preprocessing.py 
```
#### Object Segmentation

### Results
![plot](./doc/segmen_results.png)

### License
Our code is released under MIT License (see LICENSE file for details).
