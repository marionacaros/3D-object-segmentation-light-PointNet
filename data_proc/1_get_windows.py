import argparse

import laspy
from utils import *
import logging
import time
from progressbar import progressbar
from alive_progress import alive_bar
import json
import random
import hashlib
import pickle

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def split_dataset_windows(DATASET_NAME, LAS_PATH, SEL_CLASS, min_p=10, w_size=[40, 40], data_augm=3):
    global save_path
    start_time = time.time()
    logging.info(f"Dataset: {DATASET_NAME}")
    W_SIZE = w_size

    # # ------------------------------------------------- 1 --------------------------------------------------------
    # Get LAS blocks containing towers and store x,y,z in dict
    logging.info('----------------- 1 -----------------')
    logging.info(f"Get point clouds of class {SEL_CLASS}")
    # Get x,y,z of  points labeled as our target object (selClass)
    block_points_towers = get_pointCloud_selClass(LAS_PATH, selClass=SEL_CLASS)

    # store dictionary of towers points
    if not os.path.exists('./dicts'):
        os.makedirs('dicts')
    with open('dicts/dict_points_other_towers_' + DATASET_NAME + '.pkl', 'wb') as f:
        pickle.dump(block_points_towers, f)
    Load dictionary
    with open('dicts/dict_points_towers_' + DATASET_NAME + '.pkl', 'rb') as f:
        block_points_towers = pickle.load(f)

    # ----------------------------------------------- 2 ----------------------------------------------------------
    # Sliding Window for tower segmentation
    logging.info('----------------- 2 -----------------')
    dic_pc_towers, dic_center_towers = object_segmentation(block_points_towers,
                                                           min_points=min_p,
                                                           windowSize=[40, 40],
                                                           stepSize_x=20,
                                                           stepSize_y=40,
                                                           show_prints=False)
    with open('dicts/dict_segmented_other_towers_w20p' + str(min_p) + DATASET_NAME + '.pkl', 'wb') as f:
        pickle.dump(dic_pc_towers, f)
    with open('dicts/loc/dict_center_other_towers_w50p' + str(min_p) + DATASET_NAME + '.json', 'w') as f:
        json.dump(dic_center_towers, f)

    # Load dictionaries
    with open('dicts/loc/dict_center_towers_w20p' + str(min_p) + DATASET_NAME + '.json', 'r') as f:
        dic_center_towers = json.load(f)


    # ------------------------------------------------ 3 ---------------------------------------------------------
    # Loop over LAS files point clouds to get towers with context and store as LAS file
    logging.info('----------------- 3 -----------------')
    get_context(dic_center_towers, w_size=W_SIZE, path=LAS_PATH, dataset=DATASET_NAME, min_p=min_p, data_augm=data_augm,
                name='tower')

    # -------------------------------------------------- 4 -------------------------------------------------------
    # Store all points != selClass as LAS
    logging.info('----------------- 4 -----------------')
    get_points_without_object(SEL_CLASS, w_size=W_SIZE, path=LAS_PATH, center_t=dic_center_towers, dataset=DATASET_NAME)

    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
    # ------------------------------------------------------------------------------------------------------------

    
def get_pointCloud_selClass(path, selClass=15):
    """
    Get x,y,z of  points labeled as our target object (selClass)
    :param path:
    :param selClass: selected class
    :return: block_pc_towers -> points labeled as towers
    :return: all_pc -> all points and classes of blocks containing towers
    """
    block_pc_towers = {}
    files = glob.glob(os.path.join(path, '*.las'))
    logging.info("Loading LAS files")
    # with alive_bar(len(files), bar='bubbles', spinner='notes2') as bar:
    for f in progressbar(files):
        data_f = laspy.read(f)
        classes = set(data_f.classification)
        if selClass in classes:
            fileName = f.split('/')[-1].split('.')[0]
            data_f.points = data_f.points[np.where(data_f.classification == selClass)]
            # Save only coords of point clouds
            block_pc_towers[fileName] = np.vstack((data_f.x, data_f.y, data_f.z))

    return block_pc_towers


def object_segmentation(block_pc, min_points=0, windowSize=[40, 40], stepSize_x=20, stepSize_y=40, show_prints=False):
    logging.info("Towers segmentation")
    dic_pc_towers = {}
    dic_center_towers = {}
    count_t = 0

    for pc_key in progressbar(block_pc, redirect_stdout=True):
        # print('key: ', pc_key)
        towers, center_w = sliding_window_coords(point_cloud=block_pc[pc_key],
                                                 stepSize_x=stepSize_x,
                                                 stepSize_y=stepSize_y,
                                                 windowSize=windowSize,
                                                 min_points=min_points,
                                                 show_prints=show_prints)
        if towers:
            dic_pc_towers[pc_key] = towers
            dic_center_towers[pc_key] = center_w
            count_t += len(towers.keys())
            # print('Number of towers found: ', len(towers.keys()))
    print('Total number of towers: ', count_t)
    return dic_pc_towers, dic_center_towers


def read_las_files(path):
    """

    :param path: path containing LAS files
    :return: dict with [x,y,z,class]
    """
    dict_pc = {}
    files = glob.glob(os.path.join(path, '*.las'))
    with alive_bar(len(files), bar='bubbles', spinner='notes2') as bar:
        for f in files:
            fileName = f.split('/')[-1].split('.')[0]
            las_pc = laspy.read(f)
            dict_pc[fileName] = np.vstack((las_pc.x, las_pc.y, las_pc.z, las_pc.classification))
            bar()

    return dict_pc


def get_context(dic_center_obj, w_size=[40, 40], path='', dataset='', min_p=10, data_augm=0,
                name='tower'):
    """
    Get cubes of size w_size by using the center of the towers (stored in dic_center_towers)
    The minimum amount of points per tower is defined by min_p
    If data_augm > 0 data augmentation of towers with rotation and translation is generated

    :param dic_center_obj: dictionary with location of target object
    :param w_size: window size [x,y]
    :param path: directory of .las files
    :param dataset: name of dataset
    :param min_p: minimum amount of points per object to be stored
    :param data_augm: amount of augmented objects
    :param name: name of object to be stored
    """
    logging.info("Getting context of towers")
    logging.info('Loading LAS files')

    dirName = 'w_' + name + 's_' + str(w_size[0]) + 'x' + str(w_size[1])
    count = 0
    files = glob.glob(os.path.join(path, '*.las'))
    with alive_bar(len(files), bar='bubbles', spinner='notes2') as bar:
        for f in files:
            fileName = f.split('/')[-1].split('.')[0]
            bar()
            if fileName in dic_center_obj:
                las_pc = laspy.read(f)
                if dataset == 'CAT3' or dataset == 'RIBERA':
                    nir = las_pc.nir
                    red = las_pc.red
                    green = las_pc.green
                    blue = las_pc.blue
                elif dataset == 'BDN':
                    nir = np.zeros(len(las_pc))
                    red = np.zeros(len(las_pc))
                    green = np.zeros(len(las_pc))
                    blue = np.zeros(len(las_pc))

                coords_pc_class = np.vstack((las_pc.x, las_pc.y, las_pc.z, las_pc.classification,
                                             las_pc.intensity,
                                             red, green, blue,
                                             nir))
                dict_w_c = dic_center_obj[fileName]

                # Data Augmentation
                for ix in range(data_augm):
                    for w in dict_w_c:
                        # First iteration is stored without rotation or translation
                        if ix != 0 and data_augm > 0:
                            # get probabilities of variation for data augmentation
                            p_xpos = random.randint(0, 10)
                            p_xneg = random.randint(0, 10)
                            p_ypos = random.randint(0, 10)
                            p_yneg = random.randint(0, 10)
                            # move center of window
                            x = dict_w_c[w][0] + (p_xpos - p_xneg)
                            y = dict_w_c[w][1] + (p_ypos - p_yneg)
                            # rotate points
                            pc = coords_pc_class.copy()
                            angle = random.randrange(360)
                            pc[0], pc[1] = rotatePoint(angle, coords_pc_class[0], coords_pc_class[1])
                            x, y = rotatePoint(angle, x, y)

                        else:
                            pc = coords_pc_class
                            x = dict_w_c[w][0]
                            y = dict_w_c[w][1]

                        bool_w_x = np.logical_and(pc[0] < (x + w_size[0] / 2),
                                                  pc[0] > (x - w_size[0] / 2))
                        bool_w_y = np.logical_and(pc[1] < (y + w_size[1] / 2),
                                                  pc[1] > (y - w_size[1] / 2))
                        bool_w = np.logical_and(bool_w_x, bool_w_y)
                        if not any(bool_w):
                            logging.error(f'Error: No points in window {w}!')
                        else:
                            count += 1
                            # store las file
                            path_las_dir = os.path.join(save_path, dirName + '_' + str(min_p) + 'p')
                            new_file_name = name + '_v' + str(ix) + '_' + DATASET_NAME + '_' + fileName + '_w' + str(w)
                            store_las_file_from_pc(pc[:, bool_w], new_file_name, path_las_dir, dataset)

    print('Total amount of window point clouds with towers:', count)


def store_las_file_from_pc(pc, fileName, path_las_dir, dataset):
    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.4")  # we need this format for processing with PDAL
    # header.add_extra_dim(laspy.ExtraBytesParams(name="nir_extra", type=np.int32))
    header.offsets = np.array([0, 0, 0])  # np.min(pc, axis=0)
    header.scales = np.array([1, 1, 1])

    # 2. Create a Las
    las = laspy.LasData(header)
    las.x = pc[0].astype(np.int32)
    las.y = pc[1].astype(np.int32)
    las.z = pc[2].astype(np.int32)
    p_class = pc[3].astype(np.int8)
    las.intensity = pc[4].astype(np.int16)
    las.red = pc[5].astype(np.int16)
    las.green = pc[6].astype(np.int16)
    las.blue = pc[7].astype(np.int16)
    # las.return_number = pc[5].astype(np.int8)
    # las.number_of_returns = pc[6].astype(np.int8)

    # Classification unsigned char 1 byte (max is 31)
    p_class[p_class == 135] = 30
    p_class[p_class == 106] = 31
    las.classification = p_class

    if not os.path.exists(path_las_dir):
        os.makedirs(path_las_dir)
    las.write(os.path.join(path_las_dir, fileName + ".las"))

    if dataset != 'BDN':  # BDN data do not have NIR
        # Store NIR with hash ID
        nir = {}
        for i in range(pc.shape[1]):
            mystring = str(int(pc[0, i])) + '_' + str(int(pc[1, i])) + '_' + str(int(pc[2, i]))
            hash_object = hashlib.md5(mystring.encode())
            nir[hash_object.hexdigest()] = int(pc[8, i])

        with open(os.path.join(path_las_dir, fileName + '_NIR.pkl'), 'wb') as f:
            pickle.dump(nir, f)


def get_points_without_object(selClass, w_size=[70, 70], path='', center_t={}, dataset=''):
    """
    Get background, i.e. point cloud cubes without towers
    Points labeled as selClass are removed

    :param selClass: target class
    :param w_size:
    :param path:
    :param center_t:
    :param dataset:
    """

    logging.info("Get point cloud without towers")
    logging.info('Loading LAS files')
    c_filter = 0
    c_no_t = 0
    files = glob.glob(os.path.join(path, '*.las'))
    dir_name = 'w_no_towers_' + str(w_size[0]) + 'x' + str(w_size[1])

    if not os.path.exists(os.path.join(save_path, dir_name)):
        os.makedirs(os.path.join(save_path, dir_name))

    with alive_bar(len(files), bar='filling', spinner='waves') as bar:
        for f in files:
            name_f = f.split('/')[-1].split('.')[0]
            las_pc = laspy.read(f)
            if dataset == 'CAT3' or dataset == 'RIBERA':
                nir = las_pc.nir
                red = las_pc.red
                green = las_pc.green
                blue = las_pc.blue
            elif dataset == 'BDN':
                nir = np.zeros(len(las_pc))
                red = np.zeros(len(las_pc))
                green = np.zeros(len(las_pc))
                blue = np.zeros(len(las_pc))

            points = np.vstack((las_pc.x, las_pc.y, las_pc.z, las_pc.classification,
                                las_pc.intensity,
                                red, green, blue,
                                nir))
            # las_pc.return_number,
            # las_pc.number_of_returns,

            # get blocks that contained towers
            if selClass in set(points[3]):
                # get all points != class 14 and 15 (cables and towers)
                # block_pc = points[:, np.logical_and(points[3] != [14], points[3] != [15])]
                block_pc = points[points[:, 3] != selClass]
                c_filter += 1
            else:
                # Get LAS files not containing towers
                block_pc = points
                c_no_t += 1
            split_pointCloud(block_pc, f_name=name_f, dir=dir_name, path=save_path, w_size=w_size, c_tow=center_t,
                             dataset=dataset)
            bar()

    print(f'LAS files filtered towers: {c_filter}')
    print(f'LAS files no towers: {c_no_t}')


def split_pointCloud(point_cloud, f_name='', dir='w_no_towers_40x40', path='', w_size=[40, 40], c_tow={}, dataset=''):
    """ Split point cloud into windows of size w_size.

        :param point_cloud is a dict of point clouds blocks of 1km x 1km
        :param w_size is size of window
        :return pc_w
    """
    pc_had_towers = {}
    i_w = 0
    overlap_w = 0
    coords = point_cloud
    x_min, y_min, z_min = coords[0].min(), coords[1].min(), coords[2].min()
    x_max, y_max, z_max = coords[0].max(), coords[1].max(), coords[2].max()

    for y in range(round(y_min), round(y_max), w_size[1]):
        bool_w_y = np.logical_and(coords[1] < (y + w_size[1]), coords[1] > y)

        for x in range(round(x_min), round(x_max), w_size[0]):
            bool_w_x = np.logical_and(coords[0] < (x + w_size[0]), coords[0] > x)
            bool_w = np.logical_and(bool_w_x, bool_w_y)
            i_w += 1

            if any(bool_w):
                # get windows that contained towers
                if c_tow:
                    if f_name in c_tow.keys():
                        arr = np.array(list(c_tow[f_name].values()))
                        arr_x = arr[:, 0]
                        arr_y = arr[:, 1]
                        # condition to avoid doing the loop over towers centers
                        if x + w_size[0] > arr_x.min() and y + w_size[1] > arr_y.min():
                            if x < arr_x.max() and y < arr_y.max():
                                block_c_tow = c_tow[f_name]
                                # loop to check if window contains center
                                for c_i in block_c_tow:
                                    if x + w_size[0] > block_c_tow[c_i][0] and y + w_size[1] > block_c_tow[c_i][1]:
                                        if x < block_c_tow[c_i][0] and y < block_c_tow[c_i][1]:
                                            pc_had_towers[f_name + '_w' + str(i_w)] = coords[:, bool_w]
                                            # had_t_f = os.path.join(path, 'had_towers', 'hadTower_' + DATASET_NAME
                                            #                        + '_' + f_name + '_w' + str(i_w) + '.pkl')
                                            # store window that had tower twice for visualization purposes
                                            # with open(had_t_f, 'wb') as f:
                                            #     pickle.dump(coords[:, bool_w], f)

                if coords[:, bool_w].shape[1] > 0:
                    # store las file
                    path_las_dir = os.path.join(path, dir)
                    file = 'pc_' + DATASET_NAME + '_' + f_name + '_w' + str(i_w)
                    store_las_file_from_pc(coords[:, bool_w], file, path_las_dir, dataset)
                    i_w += 1
                    # Store point cloud of window in pickle
                    # stored_f = os.path.join(path, dir, 'pc_' + DATASET_NAME + '_' + f_name + '_w' + str(i_w) + '.pkl')
                    # with open(stored_f, 'wb') as f:
                    #     pickle.dump(coords[:, bool_w], f)

                # execute this code with probability 0.5%
                p = random.randrange(0, 100)
                if p < 0.5:
                    # store again window shifted 20m with overlap
                    bool_w_x = np.logical_and(coords[0] < (x + w_size[0] / 2), coords[0] > (x - w_size[0] / 2))
                    bool_w = np.logical_and(bool_w_x, bool_w_y)
                    if coords[:, bool_w].shape[1] > 0:
                        stored_f = 'pc_' + DATASET_NAME + '_' + f_name + '_w' + str(i_w) + 'overlap'
                        path_las_dir = os.path.join(path, dir)
                        store_las_file_from_pc(coords[:, bool_w], stored_f, path_las_dir, dataset)
                        # with open(stored_f, 'wb') as f:
                        #     pickle.dump(coords[:, bool_w], f)
                        overlap_w += 1

    print(f'Stored windows of block {f_name}: {i_w}')
    print(f'Overlapped windows of block {f_name}: {overlap_w}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='/dades/LIDAR/towers_detection/LAS_data_windows',
                        help='output folder where processed files are stored')
    parser.add_argument('--min_p', type=int, default=10, help='minimum number of points in object')
    parser.add_argument('--sel_class', type=int, default=15, help='selected class')
    parser.add_argument('--datasets', type=list, default=['CAT3', 'RIBERA', 'BDN'], help='list of datasets names')
    parser.add_argument('--LAS_files_path', type=str)
    parser.add_argument('--w_size', default=[40, 40])
    parser.add_argument('--data_augm', default=5)

    args = parser.parse_args()

    SEL_CLASS = args.sel_class
    # 15 corresponds to power transmission tower
    # 18 corresponds to other towers
    DATASET_NAME = args.dataset_name
    LAS_files_path = args.LAS_files_path

    for DATASET_NAME in args.datasets:
      
        save_path = os.path.join(args.out_path, DATASET_NAME)
        split_dataset_windows(DATASET_NAME, LAS_files_path, SEL_CLASS, args.min_p, args.w_size, args.data_augm)

