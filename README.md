# Object Segmentation of Cluttered Airborne LiDAR Point Clouds
3D segmentation on LiDAR data with Deep Learning

![plot](./doc/framework.png)

## Installation
The code has been tested with Python 3.7, [Pytorch](https://pytorch.org/) v1.8, CUDA 11.6  on Ubuntu 20.04. <br />
You may also need to install ```pdal``` library to transform height above sea (HAS) data into height above ground (HAG).<br />
```
pip install pdal
```

## Preprocessing pipeline
![plot](./doc/processing.png)


## Usage
Execute the following commands from the main directory.

### Preprocessing

First, execute:
```
python data_proc/1_get_windows.py --LAS_files_path path/LAS_files/here --sel_class $selected_class --min_p 20
```
This function splits our dataset into windows of a fixed size with and without our target object. <br />
First x,y,z of points labeled as our target object are obtained. <br />
Then, objects are segmented and the center of each object is stored, objects with less than ```min_p``` points are discarded. <br />
Finally, two versions of the same point cloud window are stored. A first one with a tower and a second one without tower. <br />
Point cloud cubes not containing the target object are stored as well.  <br />

Then, use PDAL library to get HAG data by executing the following code for all .LAS files: <br />
```
pdal translate $input_file $output_file hag_nn --writers.las.extra_dims="HeightAboveGround=float32"
```

Finally, run:
```
python data_proc/2_preprocessing.py 
```
This function first removes ground and points above 100 meters and then stores a sampled version using the constrained sampling explained in the paper. <br />


### Object Segmentation

To train models use:<br />
```
python pointNet/train_classification.py  /dades/LIDAR/towers_detection/datasets  --batch_size 32 --epochs 100 --learning_rate 0.001 --weighing_method EFS --number_of_points 2048 --number_of_workers 4 --sampled True
```

```
python pointNet/train_segmentation.py /dades/LIDAR/towers_detection/datasets  --batch_size 32 --epochs 50 --learning_rate 0.001 --weighing_method EFS --number_of_points 2048 --number_of_workers 4
```
To test models use:<br />
```
python pointNet/test_classification.py /dades/LIDAR/towers_detection/datasets pointNet/results/ --weighing_method EFS --number_of_points 2048 --number_of_workers 0 --model_checkpoint $checkpoint_path
```
```
python pointNet/test_segmentation.py /dades/LIDAR/towers_detection/datasets pointNet/results/ --number_of_points 2048 --number_of_workers 0 --model_checkpoint
```
## Results
![plot](./doc/segmen_results.png)

## License
Our code is released under MIT License (see LICENSE file for details).
