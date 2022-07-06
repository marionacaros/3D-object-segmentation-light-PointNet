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

# --------------------------------- DATASET BLOCKS PARTITION  -----------------------------------------------
main_path = '/dades/LIDAR/towers_detection/datasets/pc_towers_40x40/sampled_2048/*pkl'
list_f = glob.glob(main_path)
random.shuffle(list_f)
print(len(list_f))
print(list_f[0])

rib_blocks = {'train': [], 'test': [], 'val': []}
cat3_blocks = {'train': [], 'test': [], 'val': []}
bdn_blocks = {'train': [], 'test': [], 'val': []}

l_tower_files = []
l_landscape_files = []
i = 0

for file in progressbar(list_f):

    # tower_CAT3_504678_w11.pkl
    # tower_RIBERA_pt438656_w18.pkl
    # dataset = file.split('_')[-3]
    if 'tower20p_' in file:
        i += 1
        l_tower_files.append(file.split('_')[-2])
    else:
        l_landscape_files.append(file.split('_')[-2])

l_tower_files = set(l_tower_files)
l_landscape_files = list(set(l_landscape_files) - l_tower_files)

print(f'num total towers: {i}')

# towers
for i, fileName in enumerate(l_tower_files):

    # RIBERA
    if 'pt' in fileName:
        if fileName not in rib_blocks['test'] and len(rib_blocks['test']) < 3:
            rib_blocks['test'].append(fileName)
        elif len(rib_blocks['val']) < 3:
            rib_blocks['val'].append(fileName)
        else:
            rib_blocks['train'].append(fileName)
    # BDN
    elif 'c' in fileName:
        if fileName not in bdn_blocks['test'] and len(bdn_blocks['test']) < 1:
            bdn_blocks['test'].append(fileName)
        elif fileName not in bdn_blocks['val'] and len(bdn_blocks['val']) < 1:
            bdn_blocks['val'].append(fileName)
        else:
            bdn_blocks['train'].append(fileName)
    # CAT3
    else:
        if fileName not in cat3_blocks['test'] and len(cat3_blocks['test']) < 8:
            cat3_blocks['test'].append(fileName)
        elif len(cat3_blocks['val']) < 8:
            cat3_blocks['val'].append(fileName)
        else:
            cat3_blocks['train'].append(fileName)

print('len landscape files: ', len(l_landscape_files))
k = 0

# no towers files
for fileName in l_landscape_files:
    k += 1
    # RIBERA
    if 'pt' in fileName:
        rib_blocks['train'].append(fileName)
    # BDN
    elif 'c' in fileName:
        bdn_blocks['train'].append(fileName)
    # CAT3
    else:
        cat3_blocks['train'].append(fileName)

print('saved landscape files: ', k)
print('---CAT3---')
print(cat3_blocks['test'])
print('---Ribera---')
print(rib_blocks['test'])
print('---BDN---')
print(bdn_blocks['test'])

with open('dicts/dataset_blocks_partition_CAT3_reduced' + '.json', 'w') as f:
    json.dump(cat3_blocks, f)
with open('dicts/dataset_blocks_partition_RIBERA_reduced' + '.json', 'w') as f:
    json.dump(rib_blocks, f)
with open('dicts/dataset_blocks_partition_BDN_reduced' + '.json', 'w') as f:
    json.dump(bdn_blocks, f)

# ------------------------------------ create textfile with names of files --------------------------------

path = '/home/m.caros/work/objectDetection/dicts'
with open(path + '/dataset_blocks_partition_CAT3_reduced.json', 'r') as f:
    cat3_blocks = json.load(f)
with open(path + '/dataset_blocks_partition_RIBERA_reduced.json', 'r') as f:
    rib_blocks = json.load(f)
with open(path + '/dataset_blocks_partition_BDN_reduced.json', 'r') as f:
    bdn_blocks = json.load(f)

# set variables
RGBN = True
directory = 'sampled_2048'
name = '_files'

if RGBN:
    if not os.path.exists('RGBN'):
        os.makedirs('RGBN')

files = glob.glob(os.path.join('/dades/LIDAR/towers_detection/datasets/pc_towers_40x40', directory, '*.pkl'))
print(files[0])
print(directory)
print(f'Length all files: {len(files)}')
print(f'RGBN: {RGBN}')
i_t = 0
i_l = 0
dict_len = {}
dict_blockNames = {}
ctrain_pc = 0
cval_pc = 0
ctest_pc = 0
ctrain_tower = 0
cval_tower = 0
ctest_tower = 0

file_object = open('RGBN/train' + name + '.txt', 'w')
file_object = open('RGBN/val' + name + '.txt', 'w')
file_object = open('RGBN/test' + name + '.txt', 'w')

for file in progressbar(files):

    blockName = file.split('_')[-2]  # block ie pt440650
    file = file.split('/')[-1]

    # train
    if blockName in cat3_blocks['train'] or blockName in rib_blocks['train'] or blockName in bdn_blocks['train']:
        out_name = 'train'
        if RGBN:
            out_name = 'RGBN/train'
            if 'BDN' in file:
                continue
            if 'pc_' in file:
                ctrain_pc += 1
                # if ctrain_pc > 12000:
                #     continue
            else:
                ctrain_tower += 1
    # val
    elif blockName in cat3_blocks['val'] or blockName in rib_blocks['val'] or blockName in bdn_blocks['val']:
        out_name = 'val'
        if RGBN:
            out_name = 'RGBN/val'
            if 'BDN' in file:
                continue
            if 'pc_' in file:
                cval_pc += 1
                # if cval_pc > 2000:
                #     continue
            else:
                cval_tower += 1
    # test
    elif blockName in cat3_blocks['test'] or blockName in rib_blocks['test'] or blockName in bdn_blocks['test']:
        out_name = 'test'
        if RGBN:
            out_name = 'RGBN/test'
            if 'BDN' in file:
                continue
            if 'pc_' in file:
                ctest_pc += 1
                # if ctest_pc > 2000:
                #     continue
            else:
                ctest_tower += 1
    else:
        continue

    file_object = open(out_name + name + '.txt', 'a')
    file_object.write(file)
    file_object.write('\n')
    file_object.close()

print(f'RGBN - Length train landscape files: {ctrain_pc}')
print(f'RGBN - Length val landscape files: {cval_pc}')
print(f'RGBN - Length test landscape files: {ctest_pc}')
print(f'RGBN - Length train towers files: {ctrain_tower}')
print(f'RGBN - Length val towers files: {cval_tower}')
print(f'RGBN - Length test towers files: {ctest_tower}')
