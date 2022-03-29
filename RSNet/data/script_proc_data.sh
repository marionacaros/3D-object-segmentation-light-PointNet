#!/bin/bash
#python collect_indoor3d_data.py --raw_data_dir /dades/Stanford3dDataset_v1.2_Aligned_Version

INDOOR3D_DATA_DIR=stanford_indoor3d
# define num points per block
#python gen_indoor3d_h5.py --indoor3d_data_dir $INDOOR3D_DATA_DIR --split train
python gen_indoor3d_h5.py --indoor3d_data_dir $INDOOR3D_DATA_DIR --split test --stride 1.0