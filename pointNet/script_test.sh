#python test.py /dades/LIDAR/towers_detection/datasets classification results/ --weighing_method EFS \
#--number_of_points 1000 --number_of_workers 0 \
#--model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/best_checkpoint_02-16-14:03EFS.pth
#python test.py /dades/LIDAR/towers_detection/datasets classification results/ --weighing_method ISNS \
#--number_of_points 1000 --number_of_workers 0 \
#--model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/best_checkpoint_02-16-13:39ISNS.pth
#python test.py /dades/LIDAR/towers_detection/datasets classification results/ --weighing_method INS \
#--number_of_points 1000 --number_of_workers 0 \
#--model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/best_checkpoint_02-16-13:44INS.pth
# ------------------------------------------- 2000p ---------------------------
#python test.py /dades/LIDAR/towers_detection/datasets classification results/ --weighing_method EFS \
#--number_of_points 2000 --number_of_workers 0 \
#--model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/best_checkpoint_02-16-14:03EFS.pth
#python test.py /dades/LIDAR/towers_detection/datasets classification results/ --weighing_method INS \
#--number_of_points 2000 --number_of_workers 0 \
#--model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/best_checkpoint_02-16-13:44INS.pth
# ----------------------------- 2000p coords change of dataset towers 20x20 -------------------------------------
#python test.py /dades/LIDAR/towers_detection/datasets classification results/ --weighing_method EFS \
#--number_of_points 2000 --number_of_workers 0 \
#--model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/2000p/best_checkpoint_03-05-10:24EFS.pth
#python test.py /dades/LIDAR/towers_detection/datasets classification results/ --weighing_method ISNS \
#--number_of_points 2000 --number_of_workers 0 \
#--model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/2000p/best_checkpoint_03-05-10:26ISNS.pth
# ----------------------------------------- 2000p coords + intensity ------------------------------------------
# modify model to include intensity
#python test.py /dades/LIDAR/towers_detection/datasets classification results/ --weighing_method EFS \
#--number_of_points 2000 --number_of_workers 0 \
#--model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/2000p/best_checkpoint_03-06-19:12EFS.pth
#python test.py /dades/LIDAR/towers_detection/datasets classification results/ --weighing_method ISNS \
#--number_of_points 2000 --number_of_workers 0 \
#--model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/2000p/best_checkpoint_03-06-19:35ISNS.pth
# ----------------------------------------- 2000p coords + intensity + RGB ------------------------------------------
# modify model to include intensity and RGB
#python test.py /dades/LIDAR/towers_detection/datasets classification results/ --weighing_method EFS \
#--number_of_points 2000 --number_of_workers 0 \
#--model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/2000p/best_checkpoint_03-07-23:17EFS.pth
#python test.py /dades/LIDAR/towers_detection/datasets classification results/ --weighing_method ISNS \
#--number_of_points 2000 --number_of_workers 0 \
#--model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/2000p/best_checkpoint_03-07-23:17ISNS.pth

python test.py /dades/LIDAR/towers_detection/datasets classification results/ --weighing_method sklearn \
--number_of_points 2000 --number_of_workers 0 \
--model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/best_checkpoint_03-18-11:52sklearn0.999.pth

# -----new dataset best models-----

#python test.py /dades/LIDAR/towers_detection/datasets classification results/ --weighing_method EFS \
#--number_of_points 2000 --number_of_workers 0 \
#--model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/best_checkpoint_03-18-11:53EFS0.999.pth
#
#python test.py /dades/LIDAR/towers_detection/datasets classification results/ --weighing_method EFS \
#--number_of_points 2000 --number_of_workers 0 \
#--model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/best_checkpoint_03-20-19:39EFS0.9.pth
#
#python test.py /dades/LIDAR/towers_detection/datasets classification results/ --weighing_method EFS \
#--number_of_points 2000 --number_of_workers 0 \
#--model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/best_checkpoint_03-21-13:22EFS0.99.pth