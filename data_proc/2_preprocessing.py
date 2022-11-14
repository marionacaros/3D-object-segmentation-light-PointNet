import argparse
import hashlib
import logging
import random
from utils import *
import pickle
import laspy
import time

logging.basicConfig(format='--- %(asctime)s %(levelname)-8s --- %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def remove_ground_and_outliers(files_path, out_path, max_z=100.0, max_intensity=5000, n_points=2048,
                               dataset='', TH_1=3, TH_2=8):
    """
    1- Remove certain labeled points (by Terrasolid) to reduce noise and number of points
    2- Add NIR from dictionary
    3- Remove outliers defined as points > max_z and points < 0
    4- Normalize data
    5- Remove terrain points (up to n_points points in point cloud)
    6- Add constrained sampling flag at column 10

    Point labels:
    2 ➔ terreny
    8 ➔ Punts clau del model
    13 ➔ altres punts del terreny
    24 ➔ solapament
    135 (30) ➔ soroll del sensor

    It stores pickle files with preprocessed data
    """
    counters = {
        'total_count': 0,
        'need_ground': 0,
        'keep_ground': 0,
        'count_sample3': 0,
        'count_sample8': 0,
        'sample_all': 0,
        'missed': 0
    }

    out_path = os.path.join(out_path, 'normalized_' + str(n_points))
    logging.info(f'output path: {out_path}')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    files = glob.glob(os.path.join(files_path, '*.las'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        data_f = laspy.read(file)

        # Remove all categories of ground points
        data_f.points = data_f.points[np.where(data_f.classification != 8)]
        data_f.points = data_f.points[np.where(data_f.classification != 13)]
        data_f.points = data_f.points[np.where(data_f.classification != 24)]

        # Remove sensor noise
        data_f.points = data_f.points[np.where(data_f.classification != 30)]
        try:
            # Remove outliers (points above max_z)
            data_f.points = data_f.points[np.where(data_f.HeightAboveGround <= max_z)]
            # Remove points z < 0
            data_f.points = data_f.points[np.where(data_f.HeightAboveGround >= 0)]

            # check file is not empty
            if len(data_f.x) > 0:

                if dataset != 'BDN':
                    # get NIR
                    nir_arr = []
                    with open(file.replace(".las", "") + '_NIR.pkl', 'rb') as f:
                        nir_dict = pickle.load(f)

                    for x, y, z in zip(data_f.x, data_f.y, data_f.z):
                        mystring = str(int(x)) + '_' + str(int(y)) + '_' + str(int(z))
                        hash_object = hashlib.md5(mystring.encode())
                        nir_arr.append(nir_dict[hash_object.hexdigest()])

                    # NDVI
                    nir_arr = np.array(nir_arr)
                    ndvi_arr = (nir_arr - data_f.red) / (nir_arr + data_f.red)  # range [-1, 1]
                else:
                    nir_arr = np.zeros(len(data_f.x))
                    ndvi_arr = np.zeros(len(data_f.x))

                pc = np.vstack((data_f.x, data_f.y, data_f.HeightAboveGround, data_f.classification,
                                data_f.intensity / max_intensity,
                                data_f.red / 65536.0,
                                data_f.green / 65536.0,
                                data_f.blue / 65536.0,
                                nir_arr / 65535.0,
                                ndvi_arr))

                # ----------------------------------------- NORMALIZATION -----------------------------------------
                pc = pc.transpose()
                # normalize axes
                pc[:, 0] = (pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())
                pc[:, 1] = (pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())
                pc[:, 2] = pc[:, 2] / max_z
                # Remove points z < 0
                pc = pc[pc[:, 2] >= 0]

                # make sure intensity and NIR is in range (0,1)
                pc[:, 4] = np.clip(pc[:, 4], 0.0, 1.0)
                pc[:, 8] = np.clip(pc[:, 8], 0.0, 1.0)
                # normalize NDVI
                pc[:, 9] = (pc[:, 9] + 1) / 2
                pc[:, 9] = np.clip(pc[:, 9], 0.0, 1.0)

                # Check if points different from terrain < n_points
                len_pc = pc[pc[:, 3] != 2].shape[0]

                if 20 < len_pc < n_points:
                    # Get indices of ground points
                    labels = pc[:, 3]
                    i_terrain = [i for i in range(len(labels)) if labels[i] == 2.0]
                    # i_terrain = np.where(labels == 2.0, labels)
                    len_needed_p = n_points - len_pc
                    # if we have enough points of ground to cover missed points
                    if len_needed_p < len(i_terrain):
                        counters['keep_ground'] += 1
                        needed_i = random.sample(i_terrain, k=len_needed_p)
                    else:
                        needed_i = i_terrain
                        counters['need_ground'] += 1
                    points_needed_terrain = pc[needed_i, :]
                    # remove terrain points
                    pc = pc[pc[:, 3] != 2, :]
                    # store only needed terrain points
                    pc = np.concatenate((pc, points_needed_terrain), axis=0)

                elif len_pc >= n_points:
                    pc = pc[pc[:, 3] != 2, :]

                # store files with n_points as minimum
                if pc.shape[0] >= n_points:
                    counters['total_count'] += 1

                    # Add constrained sampling flag
                    pc, counters = constrained_sampling(pc, n_points, TH_1/max_z, TH_2/max_z, counters)
                    # store file
                    with open(os.path.join(out_path, fileName) + '.pkl', 'wb') as f:
                        pickle.dump(pc, f)

                else:
                    counters['missed'] += 1

        except Exception as e:
            print(f'Error {e} in file {fileName}')

    print(f'count keep ground: ', counters['keep_ground'])
    print(f'count not enough ground points: ', counters['need_ground'])
    print(f'total_count:', counters['total_count'])
    print(' ----- Constrained Sampling ------')
    print(f'counter sampled below {TH_1} m: ', counters['count_sample3'])
    print(f'counter sampled below {TH_2} m: ', counters['count_sample8'])
    print(f'counter sampled all pc: ', counters['sample_all'])
    print(f'counter total sampled: ', counters['count_sample3'] + counters['count_sample8'] + counters['sample_all'])
    print(f'counter less than n_points: ', counters['missed'])


def constrained_sampling(pc, n_points, TH_1=3.0, TH_2=8.0, counters={}):
    """
    Gradual sampling considering thresholds TH_1 and TH_2. It drops lower points and keeps higher points.
    The goal is to remove noise caused by vegetation.

    :param pc: data to apply constrained sampling
    :param n_points: minimum amount of point per PC
    :param TH_1: first height threshold to sample
    :param TH_2: second height threshold to sample
    :param counters: dictionary with counters for info purposes

    :return:pc_sampled, counters
    """

    # add column of zeros
    pc = np.c_[pc, np.zeros(pc.shape[0])]  # column ix=10
    assert pc.shape[1] == 11

    # if number of points > n_points sampling is needed
    if pc.shape[0] > n_points:
        pc_veg = pc[pc[:, 2] <= TH_1]
        pc_other = pc[pc[:, 2] > TH_1]
        # Number of points above 3m < n_points
        if pc_other.shape[0] < n_points:
            end_veg_p = n_points - pc_other.shape[0]
            counters['count_sample3'] += 1
        else:
            end_veg_p = n_points
        # if num points in vegetation > points to sample
        if pc_veg.shape[0] > end_veg_p:
            # rdm sample points < thresh 1
            sampling_indices = random.sample(range(0, pc_veg.shape[0]), k=end_veg_p)
        else:
            sampling_indices = range(pc_veg.shape[0])
        # pc_veg = pc_veg[sampling_indices, :]
        # sampled indices
        pc_veg[sampling_indices, 10] = 1
        pc_other[:, 10] = 1
        pc_sampled = np.concatenate((pc_other, pc_veg), axis=0)
        # print(f'--> sampled pc shape {pc_sampled.shape}')

        # if we still have > n_points in point cloud
        if pc_other.shape[0] > n_points:
            pc_high_veg = pc[pc[:, 2] <= TH_2]
            pc_other = pc[pc[:, 2] > TH_2]
            pc_other[:, 10] = 1
            # Number of points above 8m < n_points
            if pc_other.shape[0] < n_points:
                end_veg_p = n_points - pc_other.shape[0]
                counters['count_sample8'] += 1
            else:
                end_veg_p = n_points
            # if num points in vegetation > points to sample
            if pc_high_veg.shape[0] > end_veg_p:
                sampling_indices = random.sample(range(0, pc_high_veg.shape[0]), k=end_veg_p)
                # pc_high_veg = pc_high_veg[sampling_indices, :]
                pc_high_veg[sampling_indices, 10] = 1
                pc_sampled = np.concatenate((pc_other, pc_high_veg), axis=0)
            else:
                pc_sampled = pc_other

            # if we still have > n_points in point cloud
            if pc_sampled.shape[0] > n_points:
                # rdm sample all point cloud
                sampling_indices = random.sample(range(0, pc_sampled.shape[0]), k=n_points)
                # pc_sampled = pc_sampled[sampling_indices, :]
                pc_sampled[:, 10] = 0
                pc_sampled[sampling_indices, 10] = 1
                counters['sample_all'] += 1

    else:  # elif pc.shape[0] == n_points:
        pc[:, 10] = 1
        pc_sampled = pc

    return pc_sampled, counters


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='/dades/LIDAR/towers_detection/datasets/pc_towers_40x40_10p',
                        help='output folder where processed files are stored')
    parser.add_argument('--in_path', default='/dades/LIDAR/towers_detection/LAS_data_windows/')
    parser.add_argument('--datasets', type=list, default=['RIBERA'], help='list of datasets names')
    parser.add_argument('--n_points', type=int, default=2048)
    parser.add_argument('--max_z', type=float, default=100.0)

    args = parser.parse_args()
    start_time = time.time()

    for DATASET in args.datasets:
        paths = [args.in_path + DATASET + '/w_towers_40x40_10p',
                 args.in_path + DATASET + '/w_no_towers_40x40']

        for input_path in paths:
            logging.info(f'Input path: {input_path}')

            # IMPORTANT !!!!!!!!!
            # First execute pdal_hag.sh  # to get HeightAboveGround

            # ------ Remove ground, noise and outliers and normalize ------
            logging.info(f"1. Remove points of ground (up to {args.n_points}), noise and outliers, normalize"
                         f"and add constrained sampling flag ")
            remove_ground_and_outliers(input_path, args.out_path, max_z=args.max_z, max_intensity=5000,
                                       n_points=args.n_points, dataset=DATASET)
            print("--- Remove ground and noise time: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
            rm_ground_time = time.time()

    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
