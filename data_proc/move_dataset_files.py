import logging
import time
from progressbar import progressbar
import shutil
import os
import glob
import pickle
import numpy as np
import random
import json
#
# # ---- move files to dir ----
# path =  '/dades/LIDAR/towers_detection/datasets/*/sampled/*pkl'
# l_files =  glob.glob(path)
# dest_path = '/dades/LIDAR/towers_detection/datasets/pc_towers_40x40/sampled_2048/'
#
# for file in progressbar(l_files):
#     shutil.move(file, dest_path)
#
# path =  '/dades/LIDAR/towers_detection/datasets/*/filtered/*pkl'
# l_files =  glob.glob(path)
# dest_path = '/dades/LIDAR/towers_detection/datasets/pc_towers_40x40/data_no_ground/'
#
# for file in progressbar(l_files):
#     shutil.move(file, dest_path)
#
# # --------------------------------- DATASET BLOCKS PARTITION  -----------------------------------------------
# main_path = '/dades/LIDAR/towers_detection/datasets/pc_towers_40x40/sampled_2048/*pkl'
# list_f = glob.glob(main_path)
# random.shuffle(list_f)
# print(len(list_f))
# print(list_f[0])
#
# rib_blocks = {'train':[], 'test':[], 'val':[]}
# cat3_blocks = {'train':[], 'test':[], 'val':[]}
# bdn_blocks = {'train':[], 'test':[], 'val':[]}
#
# l_tower_files = []
# l_landscape_files = []
# i=0
# for file in progressbar(list_f):
#
#     # tower_CAT3_504678_w11.pkl
#     # tower_RIBERA_pt438656_w18.pkl
#     # dataset = file.split('_')[-3]
#     if 'tower20p_' in file:
#         i+=1
#         l_tower_files.append(file.split('_')[-2])
#     else:
#         l_landscape_files.append(file.split('_')[-2])
#
# l_tower_files = set(l_tower_files)
# l_landscape_files = list(set(l_landscape_files) - l_tower_files)
#
# print(f'num total towers: {i}')
#
# # towers
# for i, fileName in enumerate(l_tower_files):
#
#     # RIBERA
#     if 'pt' in fileName:
#         if fileName not in rib_blocks['test'] and len(rib_blocks['test'])<3:
#             rib_blocks['test'].append(fileName)
#         elif len(rib_blocks['val'])<3:
#             rib_blocks['val'].append(fileName)
#         else:
#             rib_blocks['train'].append(fileName)
#     # BDN
#     elif 'c' in fileName:
#         if fileName not in bdn_blocks['test'] and len(bdn_blocks['test'])<1:
#             bdn_blocks['test'].append(fileName)
#         elif fileName not in bdn_blocks['val'] and len(bdn_blocks['val'])<1:
#             bdn_blocks['val'].append(fileName)
#         else:
#             bdn_blocks['train'].append(fileName)
#     # CAT3
#     else:
#         if fileName not in cat3_blocks['test'] and len(cat3_blocks['test'])<8:
#             cat3_blocks['test'].append(fileName)
#         elif  len(cat3_blocks['val'])<8:
#             cat3_blocks['val'].append(fileName)
#         else:
#             cat3_blocks['train'].append(fileName)
#
# print('len landscape files: ',len(l_landscape_files))
# k=0
# # no towers files
# # discarded= l_landscape_files[-150:]
# for fileName in l_landscape_files:
#     k+=1
#     # RIBERA
#     if 'pt' in fileName:
#         rib_blocks['train'].append(fileName)
#     # BDN
#     elif 'c' in fileName:
#         bdn_blocks['train'].append(fileName)
#     # CAT3
#     else:
#         cat3_blocks['train'].append(fileName)
#
# print('saved landscape files: ',k)
# print('---CAT3---')
# print(cat3_blocks['test'])
# print('---Ribera---')
# print(rib_blocks['test'])
# print('---BDN---')
# print(bdn_blocks['test'])
#
# with open('dicts/dataset_blocks_partition_CAT3_reduced' + '.json', 'w') as f:
#     json.dump(cat3_blocks, f)
# with open('dicts/dataset_blocks_partition_RIBERA_reduced' + '.json', 'w') as f:
#     json.dump(rib_blocks, f)
# with open('dicts/dataset_blocks_partition_BDN_reduced' + '.json', 'w') as f:
#     json.dump(bdn_blocks, f)
# # with open('dicts/dataset_blocks_discarded' + '.json', 'w') as f:
# #     json.dump(discarded, f)
# #
# # #  ---------------------------- move files in partitions ---------------------------------------------------------------
# # #
#
# path = '/home/m.caros/work/objectDetection/dicts'
# with open(path + '/dataset_blocks_partition_CAT3_reduced.json', 'r') as f:
#     cat3_blocks = json.load(f)
# with open(path + '/dataset_blocks_partition_RIBERA_reduced.json', 'r') as f:
#     rib_blocks = json.load(f)
# with open(path + '/dataset_blocks_partition_BDN_reduced.json', 'r') as f:
#     bdn_blocks = json.load(f)
# # with open(path + '/dataset_blocks_discarded.json', 'r') as f:
# #     discarded = json.load(f)
#
#
# main_path = '/dades/LIDAR/towers_detection/datasets/pc_towers_40x40/data_no_ground/*pkl'
# dir='filtered'
# list_f = glob.glob(main_path)
# print(len(list_f))
# print(list_f[0])
#
# for i, file in enumerate(list_f):
#     # tower_CAT3_504678_w11.pkl
#     # tower_RIBERA_pt438656_w18.pkl
#     dataset = file.split('_')[-3]
#     fileName = file.split('_')[-2]
#
#     if fileName in cat3_blocks['test'] or fileName in rib_blocks['test'] or fileName in bdn_blocks['test']:
#         shutil.move(file, '/dades/LIDAR/towers_detection/datasets/test/'+dir)
#     elif fileName in cat3_blocks['val'] or fileName in rib_blocks['val'] or fileName in bdn_blocks['val']:
#         shutil.move(file, '/dades/LIDAR/towers_detection/datasets/val/'+dir)
#     else:
#         shutil.move(file, '/dades/LIDAR/towers_detection/datasets/train/'+dir)
#
#
# main_path = '/dades/LIDAR/towers_detection/datasets/pc_towers_40x40/sampled_2048/*pkl'
# dir='sampled'
# list_f = glob.glob(main_path)
# print(len(list_f))
# print(list_f[0])
#
# for i, file in enumerate(list_f):
#     # tower_CAT3_504678_w11.pkl
#     # tower_RIBERA_pt438656_w18.pkl
#     dataset = file.split('_')[-3]
#     fileName = file.split('_')[-2]
#
#     if fileName in cat3_blocks['test'] or fileName in rib_blocks['test'] or fileName in bdn_blocks['test']:
#         shutil.move(file, '/dades/LIDAR/towers_detection/datasets/test/'+dir)
#     elif fileName in cat3_blocks['val'] or fileName in rib_blocks['val'] or fileName in bdn_blocks['val']:
#         shutil.move(file, '/dades/LIDAR/towers_detection/datasets/val/'+dir)
#     else:
#         shutil.move(file, '/dades/LIDAR/towers_detection/datasets/train/'+dir)
#
# # # discard landscape files
# # for dir in ['filtered', 'sampled']:
# #
# #     main_path = '/dades/LIDAR/towers_detection/datasets/train/'+dir+'/*pkl'
# #     list_f = glob.glob(main_path)
# #     print(len(list_f))
# #     print(list_f[0])
# #
# #     for i, file in enumerate(list_f):
# #         # tower_CAT3_504678_w11.pkl
# #         # tower_RIBERA_pt438656_w18.pkl
# #         dataset = file.split('_')[-3]
# #         fileName = file.split('_')[-2]
# #
# #         if fileName in discarded:
# #             shutil.move(file, '/dades/LIDAR/towers_detection/datasets/discarded/'+dir+'/')
#

# # ------------------------------------ create textfile with names of files --------------------------------


for partition in ['val']:
    f1 = glob.glob(
        os.path.join('/dades/LIDAR/towers_detection/datasets/', partition, 'sampled', 'tower20p_moved_1*.pkl'))
    f2 = glob.glob(
        os.path.join('/dades/LIDAR/towers_detection/datasets/', partition, 'sampled', 'tower20p_moved_2*.pkl'))
    files = f1 + f2
    print(partition)
    print(f'Length files: {len(files)}')
    i_t = 0
    i_l = 0

    with open(partition + '_moved_towers_files.txt', 'w') as f:
        for line in files:
            if 'tower20p_' in line:
                file = line.split('/')[-1]
                f.write(file)
                f.write('\n')
                i_t+=1

    # with open(partition + '_landscape_files.txt', 'w') as f:
    #     for line in files:
    #         if not 'tower20p_' in line:
    #             file = line.split('/')[-1]
    #             f.write(file)
    #             f.write('\n')
    #             i_l+=1

    print(f'Length towers files: {i_t}')
    print(f'Length landscape files: {i_l}')

for partition in ['train']:
    print(partition)

    print(f'Length files: {len(files)}')
    i_t = 0
    i_l = 0
    with open('RGBN_' + partition + '_moved_towers_files.txt', 'w') as f:
        for line in files:
            if not 'BDN' in line:
                if 'tower20p_' in line:
                    file = line.split('/')[-1]
                    f.write(file)
                    f.write('\n')
                    i_t+=1
    #
    # with open('RGBN_' + partition + '_landscape_files.txt', 'w') as f:
    #     for line in files:
    #         if not 'BDN' in line:
    #             if not 'tower20p_' in line:
    #                 file = line.split('/')[-1]
    #                 f.write(file)
    #                 f.write('\n')
    #                 i_l+=1
    print(f'Length towers files: {i_t}')
    print(f'Length landscape files: {i_l}')


# old......
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








