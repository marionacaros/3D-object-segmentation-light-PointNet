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


def split_dataset_windows(DATASET_NAME, LAS_PATH, SEL_CLASS, min_p = 20):

    global save_path
    start_time = time.time()
    logging.info(f"Dataset: {DATASET_NAME}")

    # ------------------------------------------------- 1 --------------------------------------------------------
    # Get LAS blocks containing towers
    logging.info('----------------- 1 -----------------')
    logging.info(f"Get point clouds of class {SEL_CLASS}")
    block_points_towers = get_pointCloud_selClass(LAS_PATH, selClass=SEL_CLASS)

    if not os.path.exists('./dicts'):
        os.makedirs('dicts')

    # save dictionary of towers points
    with open('dicts/dict_points_towers_' + DATASET_NAME + '.pkl', 'wb') as f:
        pickle.dump(block_points_towers, f)

    # Load dictionary
    with open('dicts/dict_points_towers_' + DATASET_NAME + '.pkl', 'rb') as f:
        block_points_towers = pickle.load(f)

    # ----------------------------------------------- 2 ----------------------------------------------------------
    # Sliding Window for tower segmentation
    logging.info('----------------- 2 -----------------')
    dic_pc_towers, dic_center_towers = object_segmentation(block_points_towers,
                                                           min_points=min_p,
                                                           windowSize=[20, 20],
                                                           stepSize_x=10,
                                                           stepSize_y=20,
                                                           show_prints=False)
    with open('dicts/dict_segmented_towers_w20p' + str(min_p) + DATASET_NAME + '.pkl', 'wb') as f:
        pickle.dump(dic_pc_towers, f)
    with open('dicts/dict_center_towers_w20p' + str(min_p) + DATASET_NAME + '.json', 'w') as f:
        json.dump(dic_center_towers, f)

    # Load dictionaries
    with open('dicts/dict_segmented_towers_w20p' + str(min_p) + DATASET_NAME + '.pkl', 'rb') as f:
        dic_pc_towers = pickle.load(f)
    with open('dicts/dict_center_towers_w20p' + str(min_p) + DATASET_NAME + '.json', 'r') as f:
        dic_center_towers = json.load(f)

    # ------------------------------------------------ 3 ---------------------------------------------------------
    # Loop over LAS files point clouds to get towers with context
    logging.info('----------------- 3 -----------------')
    get_context(dic_center_towers, w_size=[40, 40], path=LAS_PATH, dataset=DATASET_NAME)

    # # -------------------------------------------------- 4 -------------------------------------------------------
    # Store all points != selClass
    logging.info('----------------- 4 -----------------')
    get_points_without_object(SEL_CLASS, path=LAS_PATH, center_t=dic_center_towers, dataset=DATASET_NAME)

    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
    # ------------------------------------------------------------------------------------------------------------


# def parallel_reading(files_list, selClass, NUM_CPUS):
#     p = multiprocessing.Pool(NUM_CPUS)
#     func = partial(get_pointCloud_selClass,
#                    selClass=selClass)
#     data_outputs=p.imap_unordered(func, files_list, 10)
#     p.close()
#     p.join()
#     return data_outputs


def get_pointCloud_selClass(path, selClass=15):
    """
    Store x,y,z of points classified as a certain class (towers)
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


def get_context(dic_center_towers, w_size=[40, 40], path='', dataset=''):
    logging.info("Getting context of towers")
    logging.info('Loading LAS files')
    count = 0
    files = glob.glob(os.path.join(path, '*.las'))
    with alive_bar(len(files), bar='bubbles', spinner='notes2') as bar:
        for f in files:
            fileName = f.split('/')[-1].split('.')[0]
            bar()
            if fileName in dic_center_towers:
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
                dict_w_c = dic_center_towers[fileName]

                for w in dict_w_c:

                    # get probabilities of variation
                    p_xpos = random.randint(0, 15)
                    p_xneg = random.randint(0, 15)
                    p_ypos = random.randint(0, 15)
                    p_yneg = random.randint(0, 15)

                    x = dict_w_c[w][0] + (p_xpos - p_xneg)
                    y = dict_w_c[w][1] + (p_ypos - p_yneg)

                    # print('x: ', (p_xpos - p_xneg))
                    # print('y: ', p_ypos - p_yneg)

                    bool_w_x = np.logical_and(coords_pc_class[0] < (x + w_size[0] / 2),
                                              coords_pc_class[0] > (x - w_size[0] / 2))
                    bool_w_y = np.logical_and(coords_pc_class[1] < (y + w_size[1] / 2),
                                              coords_pc_class[1] > (y - w_size[1] / 2))
                    bool_w = np.logical_and(bool_w_x, bool_w_y)
                    if not any(bool_w):
                        logging.error(f'Error: No points in window {w}!')
                    else:
                        count += 1
                        # store las file
                        path_las_dir = os.path.join(save_path, 'w_towers_40x40_20p')
                        new_file_name = 'tower20p_moved_3_' + DATASET_NAME + '_' + fileName + '_w' + str(w)
                        store_las_file_from_pc(coords_pc_class[:, bool_w], new_file_name, path_las_dir, dataset)

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

    if dataset != 'BDN': # BDN data do not have NIR
        # Store NIR with hash ID
        nir = {}
        for i in range(pc.shape[1]):
            mystring = str(int(pc[0, i])) + '_' + str(int(pc[1, i])) + '_' + str(int(pc[2, i]))
            hash_object = hashlib.md5(mystring.encode())
            nir[hash_object.hexdigest()] = int(pc[8, i])

        with open(os.path.join(path_las_dir, fileName + '_NIR.pkl'), 'wb') as f:
            pickle.dump(nir, f)


def get_points_without_object(selClass, path='', center_t={}, dataset=''):
    """get windows without towers
       Points labeled as selClass are removed """

    logging.info("Get point cloud without towers")
    logging.info('Loading LAS files')
    c_filter = 0
    c_no_t = 0
    files = glob.glob(os.path.join(path, '*.las'))
    dir_name = 'w_no_towers_40x40'

    if not os.path.exists(os.path.join(save_path,dir_name)):
        os.makedirs(os.path.join(save_path,dir_name))

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
                block_pc = points[:, np.logical_and(points[3] != [14], points[3] != [15])]
                c_filter += 1
            else:
                # Get LAS files not containing towers
                block_pc = points
                c_no_t += 1
            split_pointCloud(block_pc, f_name=name_f, dir=dir_name, path=save_path, w_size=[40, 40], c_tow=center_t,
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
                    bool_w_x = np.logical_and(coords[0] < (x + w_size[0]/2), coords[0] > (x-w_size[0]/2))
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
    parser.add_argument('--output_folder', type=str, default='datasets', help='output folder')
    parser.add_argument('--min_p', type=int, default=20, help='minimum number of points in object')
    parser.add_argument('--sel_class', type=int, default=15, help='selected class')
    parser.add_argument('--dataset_name', type=str, default='CAT3', help='name of dataset')
    parser.add_argument('--LAS_files_path', type=str)

    args = parser.parse_args()

    SEL_CLASS = args.sel_class # 15 corresponds to our target class (power transmission tower)
    DATASET_NAME = args.dataset_name
    LAS_files_path = args.LAS_files_path

    save_path = os.path.join(args.output_folder,DATASET_NAME)
    split_dataset_windows(DATASET_NAME, LAS_files_path, SEL_CLASS, args.min_p)

