import logging
import time
from progressbar import progressbar
import shutil
import os
import glob
import pickle
import numpy as np
import random

main_path = '/dades/LIDAR/towers_detection/datasets/train/towers_2000/*.pkl'
list_f = glob.glob(main_path)
random.shuffle(list_f)
print(len(list_f))
#
# for i, file in enumerate(list_f):
#     if i <= len(list_f)/2:
#         shutil.move(file, '/dades/LIDAR/towers_detection/datasets/val/towers_2000/.')
#     # print(file)

main_path = '/dades/LIDAR/towers_detection/datasets/train/landscape_2000/*.pkl'
list_f = glob.glob(main_path)
random.shuffle(list_f)
print(len(list_f))
#
# for i, file in enumerate(list_f):
#     if i <= len(list_f)/2:
#         shutil.move(file, '/dades/LIDAR/towers_detection/datasets/val/landscape_2000/.')




    # with open(file, 'rb') as f:
    #     pc = pickle.load(f).astype(np.float32)
    #     if pc.shape == (2000, 11):
    #         print(file)
    #         os.remove(file)


# train_files = glob.glob(os.path.join(main_path, 'train/**/*.pkl'))
# with open('train_files.txt', 'w') as f:
#     for line in train_files:
#         file=line.split('/')[-1]
#         f.write(file)
#         f.write('\n')
#
# val_files = glob.glob(os.path.join(main_path, 'val/**/*.pkl'))
# with open('val_files.txt', 'w') as f:
#     for line in val_files:
#         file = line.split('/')[-1]
#         f.write(file)
#         f.write('\n')
# test_files = glob.glob(os.path.join(main_path, 'test/**/*.pkl'))
# with open('test_files.txt', 'w') as f:
#     for line in test_files:
#         file = line.split('/')[-1]
#         f.write(file)
#         f.write('\n')

# # create list of point clouds in train/val/tes set
    # train_files = open('train_files.txt', 'w')
    # train_no_towers = 0
    # train_towers = 0
    # val_no_towers = 0
    # val_towers = 0
    # test_no_towers = 0
    # test_towers = 0
    # for data in progressbar(train_dataloader):
    #     points, target, file_name = data
    #     train_files.write("%s\n" % file_name)
    #     target = target.cpu().detach().numpy()
    #     if target[0] == 1:
    #         train_no_towers += 1
    #     else:
    #         train_towers += 1
    # train_files.close()
    # val_files = open('val_files.txt', 'w')
    # print('val_files')
    # for data in progressbar(val_dataloader):
    #     points, target, file_name = data
    #     val_files.write("%s\n" % file_name)
    #     target = target.cpu().detach().numpy()
    #     if target[0] == 1:
    #         val_no_towers += 1
    #     else:
    #         val_towers += 1
    # val_files.close()
    # test_files = open('test_files.txt', 'w')
    # for data in progressbar(test_dataloader):
    #     points, target, file_name = data
    #     test_files.write("%s\n" % file_name)
    #     target = target.cpu().detach().numpy()
    #     if target[0] == 1:
    #         test_no_towers += 1
    #     else:
    #         test_towers += 1
    # test_files.close()
    # print('train_no_towers:, ', train_no_towers)
    # print('train_towers:, ', train_towers)
    # print('val_no_towers:, ', val_no_towers)
    # print('val_towers:, ', val_towers)
    # print('test_no_towers:, ', test_no_towers)
    # print('test_towers:, ', test_towers)


# dir='test'
# path='/home/m.caros/work/objectDetection/split_dataset/' + dir + '_files.txt'
# dataset_file = open(path, 'r')
# counter_files_no_exist=0
#
# with open(path, 'r') as fd:
#     files = dataset_file.read().split('\n')
#     for row in progressbar(files):
#         if 'tower' in row:
#             origin_path = '/dades/LIDAR/towers_detection/datasets/pc_towers_40x40/sampled_2000/' + row
#             dest_path = '/dades/LIDAR/towers_detection/datasets/' + dir + '/towers_2000/'
#         elif 'pc' in row:
#             origin_path = '/dades/LIDAR/towers_detection/datasets/pc_no_towers_40x40/sampled_2000/' + row
#             dest_path = '/dades/LIDAR/towers_detection/datasets/' + dir + '/landscape_2000/'
#
#         if not os.path.isdir('/dades/LIDAR/towers_detection/datasets/' + dir):
#             os.mkdir('/dades/LIDAR/towers_detection/datasets/' + dir)
#         if not os.path.isdir(dest_path):
#             os.mkdir(dest_path)
#         try:
#             shutil.move(origin_path, dest_path)
#         except Exception as e:
#             print(e)
#             counter_files_no_exist += 1
# print('counter_files_no_exist: ', counter_files_no_exist)

# dir='train'
# path = glob.glob('/dades/LIDAR/towers_detection/datasets/pc_no_towers_40x40/sampled_2000/*.pkl')
# for i, file in enumerate(path):
#     if i < 0:
#         exit()
#     else:
#         dest_path='/dades/LIDAR/towers_detection/datasets/' + dir + '/landscape_2000/'
#         shutil.move(file, dest_path)




