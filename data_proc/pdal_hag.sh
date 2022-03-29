#!/bin/bash
#for p in /home/m.caros/work/objectDetection/datasets/RIBERA/w_towers_40x40/*.las; do
#  pdal translate $p $p hag_nn --writers.las.extra_dims="HeightAboveGround=float32"
#  echo "$p"
#done
#for p in /home/m.caros/work/objectDetection/datasets/RIBERA/w_no_towers_40x40/*.las; do
#  pdal translate $p $p hag_nn --writers.las.extra_dims="HeightAboveGround=float32"
#  echo "$p"
#done
#for p in /home/m.caros/work/objectDetection/datasets/CAT3/w_towers_40x40/*.las; do
#  pdal translate $p $p hag_nn --writers.las.extra_dims="HeightAboveGround=float32"
#  echo "$p"
#done
for p in /home/m.caros/work/objectDetection/datasets/CAT3/w_no_towers_40x40/*.las; do
  pdal translate $p $p hag_nn --writers.las.extra_dims="HeightAboveGround=float32"
  echo "$p"
done

# Rename files
#for f in /home/m.caros/work/objectDetection/datasets/CAT3/w_no_towers_40x40/*pkl.las; do
#  mv -- "$f" "${f%.pkl.las}.las"
#done
# save HAG in z
#for p in /dades/LIDAR/towers_detection/datasets/pc_towers_40x40/las_files/*HAG.las; do
#  pdal translate $p $p hag_nn ferry --filters.ferry.dimensions="HeightAboveGround=Z"
#  echo "$p"
#done