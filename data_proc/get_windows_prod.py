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


def split_dataset_windows(dataset, path, save_path):

    start_time = time.time()
    logging.info(f"Dataset: {dataset}")

    # Sliding Window for tower segmentation
    logging.info('----------------- Sliding window -----------------')

    logging.info('Loading LAS files')
    files = glob.glob(os.path.join(path, '*.las'))
    dir_name = 'w_prod_40x40'

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

            sliding_window(points, f_name=name_f, dir=dir_name, path=save_path, w_size=[40, 40],
                           dataset=dataset)
            bar()

    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))


def sliding_window(coords, f_name='', dir='', path='', w_size=[40, 40], dataset=''):
    """ Split point cloud into overlapping windows of size w_size.

        :param coords is a dict of point clouds blocks of 1km x 1km
        :param w_size is size of window
    """
    x_w = 0
    y_w = 0
    x_min, y_min, z_min = coords[0].min(), coords[1].min(), coords[2].min()
    x_max, y_max, z_max = coords[0].max(), coords[1].max(), coords[2].max()

    for y in range(round(y_min), round(y_max), int(w_size[1]/2)):
        bool_w_y = np.logical_and(coords[1] < (y + w_size[1]), coords[1] > y)
        y_w += 1

        for x in range(round(x_min), round(x_max), int(w_size[0]/2)):
            bool_w_x = np.logical_and(coords[0] < (x + w_size[0]), coords[0] > x)
            bool_w = np.logical_and(bool_w_x, bool_w_y)
            x_w += 1

            if any(bool_w):
                if coords[:, bool_w].shape[1] > 0:
                    # store las file
                    path_las_dir = os.path.join(path, dir)
                    file = 'w_' + DATASET_NAME + '_' + f_name + '_y' + str(y_w) + '_x' + str(x_w)
                    store_las_file_from_pc(coords[:, bool_w], file, path_las_dir, dataset)

    # print(f'Stored windows of block {f_name}: {i_w}')
    # print(f'Overlapped windows of block {f_name}: {overlap_w}')


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


if __name__ == '__main__':

    SEL_CLASS = 15
    # DATASET_NAME = 'RIBERA'
    # DATASET_NAME = 'CAT3'
    # DATASET_NAME = 'BDN'

    DATASETS = ['BDN', 'RIBERA', 'CAT3']

    for DATASET_NAME in DATASETS:

        # paths
        if DATASET_NAME == 'BDN':
            LAS_files_path = '/mnt/Lidar_K/PROJECTES/0025310000_VOLTA_MachineLearning_Badalona_FBK_5anys/Lliurament_211203_Mariona/LASCLAS_AMB_FOREST-URBAN/FOREST'
        elif DATASET_NAME == 'CAT3':
            LAS_files_path = '/mnt/Lidar_M/DEMO_Productes_LIDARCAT3/LAS_Filtrat_Offset4m_Z'
        elif DATASET_NAME == 'RIBERA':
            LAS_files_path = '/mnt/Lidar_O/DeepLIDAR/VolVegetacioRibera_ClassTorres-Linies/LAS'

        # save_path = '/home/m.caros/work/objectDetection/datasets/'+DATASET_NAME
        save_path = '/dades/LIDAR/towers_detection/datasets/datasets_prod/'+DATASET_NAME

        split_dataset_windows(DATASET_NAME, LAS_files_path, save_path)

