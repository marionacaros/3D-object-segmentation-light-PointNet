import hashlib
import logging
import os
import random

from utils import *
import pickle
import laspy
import time

logging.basicConfig(format='--- %(asctime)s %(levelname)-8s --- %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def get_max(files_path):
    files = glob.glob(os.path.join(files_path, '*.las'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        data_f = laspy.read(file)
        hag = data_f.HeightAboveGround
        if hag.max() > max_z:
            max_z = hag.max()


def remove_ground_and_outliers(files_path, out_path, max_z=100.0, max_intensity=5000, n_points=2048,
                               raw_data=False, dataset=''):
    """
    1- Remove certain labeled points (by Terrasolid) to reduce noise and number of points
    2- Add NIR from dictionary
    3- Remove outliers defined as points > max_z and points < 0
    4- Normalize data
    5- Remove terrain points (up to 2000 points in point cloud)

    Point labels:
    2 ➔ terreny
    8 ➔ Punts clau del model
    13 ➔ altres punts del terreny
    24 ➔ solapament
    135 (30) ➔ soroll del sensor

    It stores pickle files with preprocessed data
    """
    count_mantain_terrain_p = 0
    count_less_2000 = 0
    total_count = 0

    out_path = os.path.join(out_path, 'sampled_'+str(n_points))
    logging.info(f'output path: {out_path}')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    files = glob.glob(os.path.join(files_path, '*.las'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        data_f = laspy.read(file)

        if not raw_data:
            data_f.points = data_f.points[np.where(data_f.classification != 8)]
            data_f.points = data_f.points[np.where(data_f.classification != 13)]
            data_f.points = data_f.points[np.where(data_f.classification != 24)]

        # Remove sensor noise
        data_f.points = data_f.points[np.where(data_f.classification != 30)]

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
            try:
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

                if not raw_data:

                    # Check if points different from terrain < 2000
                    len_pc = pc[pc[:, 3] != 2].shape[0]

                    if 20 < len_pc < n_points:
                        # Get indices of terrain points
                        labels = pc[:, 3]
                        i_terrain = [i for i in range(len(labels)) if labels[i] == 2.0]
                        # i_terrain = np.where(labels == 2.0, labels)
                        len_needed_p = n_points - len_pc
                        if len_needed_p > len(i_terrain):
                            needed_i = i_terrain
                            count_less_2000 += 1
                        else:
                            count_mantain_terrain_p += 1
                            needed_i = random.sample(i_terrain, k=len_needed_p)
                        points_needed_terrain = pc[needed_i, :]
                        # remove terrain points
                        pc = pc[pc[:, 3] != 2, :]
                        # store only needed terrain points
                        pc = np.concatenate((pc, points_needed_terrain), axis=0)

                    elif len_pc >= n_points:
                        pc = pc[pc[:, 3] != 2, :]

                    elif len_pc <= 20:
                        print(f'Point cloud {fileName} not stored. Less than 20 points. Only {len_pc} points')
                        continue

                    if pc.shape[0] >= n_points:
                        if pc[:, 2].max() > max_z:
                            print('Outliers not removed correctly!!')
                        total_count += 1
                        f_path = os.path.join(out_path, fileName)
                        with open(f_path + '.pkl', 'wb') as f:
                            pickle.dump(pc, f)
                    else:
                        print(f'Point cloud {fileName} not stored. Number of points < {n_points}')
                else:
                    # store raw data
                    if pc.shape[0] >= n_points:  # remove windows with less than 1000 points
                        total_count += 1
                        f_path = os.path.join(out_path, fileName)
                        with open(f_path + '.pkl', 'wb') as f:
                            pickle.dump(pc, f)
                    else:
                        print(f'Point cloud {fileName} not stored. Less than {n_points} points. Only {pc.shape[0]} points')

            except Exception as e:
                print(f'Error {e} in file {fileName}')

    print(f'count_mantain_terrain_p: {count_mantain_terrain_p}')
    print(f'count_less {n_points}: {count_less_2000}')
    print(f'total_count: {total_count}')


def sampling(files_path, n_points, TH_1=3.0, TH_2=8.0):
    count_interpolate = 0
    count_sample3 = 0
    count_sample8 = 0
    count_sample_all = 0

    path_sampled = os.path.join(files_path, 'sampled_'+str(n_points))
    logging.info(f'path: {path_sampled}')
    # if not os.path.exists(path_sampled):
    #     os.makedirs(path_sampled)
    for file in progressbar(glob.glob(os.path.join(path_sampled,'*pkl'))):
        fileName = file.split('/')[-1].split('.')[0]
        with open(file, 'rb') as f:
            pc = pickle.load(f).astype(np.float32)
        # add column of zeros
        pc = np.c_[pc, np.zeros(pc.shape[0])] # column ix=10

        # if number of points > n_points sampling is needed
        if pc.shape[0] > n_points:
            pc_veg = pc[pc[:, 2] <= TH_1]
            pc_other = pc[pc[:, 2] > TH_1]
            # Number of points above 3m < n_points
            if pc_other.shape[0] < n_points:
                end_veg_p = n_points - pc_other.shape[0]
                count_sample3 += 1
            else:
                end_veg_p = n_points
            # if num points in vegetation > points to sample
            if pc_veg.shape[0] > end_veg_p:
                # rdm sample points < thresh 1
                sampling_indices = random.sample(range(0,pc_veg.shape[0]), k=end_veg_p)
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
                    count_sample8 += 1
                else:
                    end_veg_p = n_points
                # if num points in vegetation > points to sample
                if pc_high_veg.shape[0] > end_veg_p:
                    sampling_indices = random.sample(range(0,pc_high_veg.shape[0]), k=end_veg_p)
                    # pc_high_veg = pc_high_veg[sampling_indices, :]
                    pc_high_veg[sampling_indices, 10] = 1
                    pc_sampled = np.concatenate((pc_other, pc_high_veg), axis=0)
                else:
                    pc_sampled = pc_other

                # if we still have > n_points in point cloud
                if pc_sampled.shape[0] > n_points:
                    # rdm sample all point cloud
                    sampling_indices = random.sample(range(0,pc_sampled.shape[0]), k=n_points)
                    # pc_sampled = pc_sampled[sampling_indices, :]
                    pc_sampled[:, 10] = 0
                    pc_sampled[sampling_indices, 10] = 1
                    count_sample_all += 1

            with open(os.path.join(path_sampled, fileName) + '.pkl', 'wb') as f:
                pickle.dump(pc_sampled, f)

        elif pc.shape[0] == n_points:
            pc[:,10]=1
            pc_sampled = pc
            with open(os.path.join(path_sampled, fileName) + '.pkl', 'wb') as f:
                pickle.dump(pc_sampled, f)

        ##################################### ADD SYNTHETIC SAMPLES ###########################################
        else:
            # if MIN_POINTS < pc.shape[0] < n_points:
            count_interpolate += 1
            pc[:, 10] = 1
            pc_sampled = pc
            with open(os.path.join(path_sampled, fileName) + '.pkl', 'wb') as f:
                pickle.dump(pc_sampled, f)
            # generate points
            # i = 0
            # new_pc = list(pc)
            # # while points < 1000 keep duplicating
            # while (len(new_pc) < n_points):
            #     if i >= len(pc[:, 0]) - 1:
            #         i = 0
            #     else:
            #         i += 1
            #     # get point in position i and duplicate with noise
            #     p = pc[i, :]
            #     # mu, sigma = [0, 0, 0], [0.1, 0.1, 0.1]
            #     # creating noise with the same dimension as the dataset
            #     # noise = abs(np.random.normal(mu, sigma, [3]))
            #     # new_p = list(p[:3] + noise)
            #     new_p = list(p[:3])
            #     new_p.append(int(p[3]))  # label
            #     new_p.append(p[4])
            #     new_p.append(p[5])
            #     new_p.append(p[6])
            #     new_p.append(p[7])
            #     new_p.append(p[8])
            #     new_pc.append(new_p)
            # new_pc = np.array(new_pc)

            # with open(os.path.join(path_sampled, fileName) + '.pkl', 'wb') as f:
            #     pickle.dump(new_pc, f)
            # else:
            #     if MIN_POINTS > pc.shape[0]:
            #         c_min_points += 1

    print(f'counter sampled below {TH_1} m: {count_sample3}')
    print(f'counter sampled below {TH_2} m: {count_sample8}')
    print(f'counter sampled all: {count_sample_all}')
    print(f'counter total sampled: {count_sample3 + count_sample8 + count_sample_all}')
    print(f'counter less than n_points: {count_interpolate}')
    # print(f'Discarded point clouds because < {MIN_POINTS} points: ', c_min_points)


if __name__ == '__main__':

    N_POINTS = 2048
    MAX_Z = 100.0
    raw_data = False
    logging.info(f'Want raw data: {raw_data}') # if raw data == True code does not remove ground points
    out_path = '/dades/LIDAR/towers_detection/datasets/pc_towers_40x40'


    for DATASET in ['CAT3','RIBERA', 'BDN']:
        paths = ['datasets/' + DATASET + '/w_towers_40x40_10p',
                 'datasets/' + DATASET + '/w_no_towers_40x40']

        start_time = time.time()
        for files_path in paths:
            logging.info(f'Input path: {files_path}')

            # IMPORTANT !!!!!!!!!
            # execute compute_pdal_bash.sh  # to get HeighAboveGround

            # ------ Remove ground, noise and outliers and normalize ------
            logging.info(f"1. Remove points of ground, noise and outliers and normalize ")
            remove_ground_and_outliers(files_path, out_path, max_z=MAX_Z, max_intensity=5000,
                                                        n_points=N_POINTS, raw_data=raw_data, dataset=DATASET)
            print("--- Remove ground and noise time: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
            rm_ground_time = time.time()

    # ------ sampling ------
    logging.info(f"2. Sampling")
    sampling(out_path,  n_points=N_POINTS, TH_1=3.0 / MAX_Z, TH_2=8.0 / MAX_Z)

    print("--- Sample and interpolate time: %s h ---" % (round((time.time() - rm_ground_time) / 3600, 3)))
    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
    
