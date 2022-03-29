import hashlib
import logging
from utils import *
import pickle
import laspy
import time

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
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


def remove_ground_and_outliers(files_path, out_path, max_z=100.0, max_intensity=5000):
    """
    1- Remove certain labeled points (from Terrasolid) to reduce noise and number of points
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

    path_no_ground_dir = os.path.join(out_path, 'data_no_ground')
    print(f'output path: {path_no_ground_dir}')
    if not os.path.exists(path_no_ground_dir):
        os.makedirs(path_no_ground_dir)

    files = glob.glob(os.path.join(files_path, '*.las'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        data_f = laspy.read(file)

        data_f.points = data_f.points[np.where(data_f.classification != 8)]
        data_f.points = data_f.points[np.where(data_f.classification != 13)]
        data_f.points = data_f.points[np.where(data_f.classification != 135)]
        data_f.points = data_f.points[np.where(data_f.classification != 24)]
        # Remove outliers (points above max_z)
        data_f.points = data_f.points[np.where(data_f.HeightAboveGround <= max_z)]
        # Remove points z < 0
        data_f.points = data_f.points[np.where(data_f.HeightAboveGround >= 0)]
        # data_f.points = data_f.points[np.where(data_f.classification != 2)]

        # check file is not empty
        if len(data_f.x) > 0:
            # get NIR
            nir_arr = []
            with open(file.replace(".las", "") + '_NIR.pkl', 'rb') as f:
                nir_dict = pickle.load(f)

            for x,y,z in zip(data_f.x, data_f.y, data_f.z):
                mystring = str(int(x)) + '_' + str(int(y)) + '_' + str(int(z))
                hash_object = hashlib.md5(mystring.encode())
                nir_arr.append(nir_dict[hash_object.hexdigest()])

            # NDVI
            nir_arr = np.array(nir_arr)
            ndvi_arr = (nir_arr - data_f.red)/(nir_arr + data_f.red)  # range [-1, 1]

            try:
                pc = np.vstack((data_f.x, data_f.y, data_f.HeightAboveGround, data_f.classification,
                                    data_f.intensity / max_intensity,
                                    data_f.return_number,
                                    data_f.number_of_returns,
                                    data_f.red / 65536.0,
                                    data_f.green / 65536.0,
                                    data_f.blue / 65536.0,
                                    nir_arr / 65535.0,
                                    ndvi_arr))

                # ----------------------------------------- NORMALIZATION -----------------------------------------
                pc=pc.transpose()
                # normalize axes
                pc[:, 0] = (pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())
                pc[:, 1] = (pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())
                pc[:, 2] = pc[:, 2] / max_z
                # Remove points z < 0
                pc = pc[pc[:, 2] >= 0]

                # make sure intensity is in range (0,1)
                pc[:, 4] = np.clip(pc[:, 4], 0.0, 1.0)
                pc[:, 10] = np.clip(pc[:, 10], 0.0, 1.0)
                # return number not normalized
                # number of returns not normalized

                # Check if points different from terrain < 2000
                len_pc = pc[pc[:, 3] != 2].shape[0]

                if 0 < len_pc < 2000:
                    # Get indices of terrain points
                    labels = pc[:, 3]
                    i_terrain = [i for i in range(len(labels)) if labels[i] == 2.0]
                    # i_terrain = np.where(labels == 2.0, labels)
                    len_needed_p = 2000-len_pc
                    if len_needed_p > len(i_terrain):
                        needed_i = i_terrain
                        count_less_2000 += 1
                    else:
                        count_mantain_terrain_p +=1
                        needed_i = np.random.choice(i_terrain, len_needed_p)
                    points_needed_terrain = pc[needed_i, :]
                    # remove terrain points
                    pc = pc[pc[:, 3] != 2, :]
                    # store only needed terrain points
                    pc = np.concatenate((pc, points_needed_terrain), axis=0)

                else:
                    pc = pc[pc[:, 3] != 2, :]

                if pc.shape[0] > 0:
                    if pc[:, 2].max() > max_z:
                        print('Outliers not removed correctly!!')
                    total_count += 1
                    f_path = os.path.join(path_no_ground_dir, fileName)
                    with open(f_path + '.pkl', 'wb') as f:
                        pickle.dump(pc, f)

            except Exception as e:
                print(f'Error {e} in file {fileName}')

    print(f'count_mantain_terrain_p: {count_mantain_terrain_p}')
    print(f'count_less_2000: {count_less_2000}')
    print(f'total_count: {total_count}')
    return path_no_ground_dir


def sampling(files_path, N_POINTS, TH_1=3.0, TH_2=8.0, MIN_POINTS=100):
    count_interpolate = 0
    count_sample3 = 0
    count_sample8 = 0
    count_sample_all = 0
    c_min_points = 0

    dir_path = os.path.dirname(files_path)
    path_sampled = os.path.join(dir_path, 'sampled_' + str(N_POINTS))
    if not os.path.exists(path_sampled):
        os.makedirs(path_sampled)

    files = glob.glob(os.path.join(files_path, '*.pkl'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        with open(file, 'rb') as f:
            pc = pickle.load(f).astype(np.float32)  # [3890, 12]

        # if number of points > N_POINTS sampling is needed
        if pc.shape[0] > N_POINTS:
            pc_veg = pc[pc[:, 2] <= TH_1]
            pc_other = pc[pc[:, 2] > TH_1]
            # print(f'Number of points below {TH_1*100}m: {pc_veg.shape[0]}')
            # print(f'Number of points above {TH_1*100}m: {pc_other.shape[0]}')

            # Number of points above 3m < 1000
            if pc_other.shape[0] < N_POINTS:
                end_veg_p = N_POINTS - pc_other.shape[0]
                count_sample3 += 1
            else:
                end_veg_p = N_POINTS
            # if num points in vegetation > points to sample
            if pc_veg.shape[0] > end_veg_p:
                # rdm sample points < thresh 1
                sampling_indices = np.random.choice(pc_veg.shape[0], end_veg_p)
            else:
                sampling_indices = range(pc_veg.shape[0])
            pc_veg = pc_veg[sampling_indices, :]
            pc_sampled = np.concatenate((pc_other, pc_veg), axis=0)
            # print(f'--> sampled pc shape {pc_sampled.shape}')

            # if we still have > 1000 in point cloud
            if pc_other.shape[0] > N_POINTS:
                pc_high_veg = pc[pc[:, 2] <= TH_2]
                pc_other = pc[pc[:, 2] > TH_2]
                # print(f'Number of points below {TH_2*100}m: {pc_high_veg.shape[0]}')
                # print(f'Number of points above {TH_2*100}m: {pc[pc[:, 2] > TH_2].shape[0]}')

                # Number of points above 8m < 1000
                if pc_other.shape[0] < N_POINTS:
                    end_veg_p = N_POINTS - pc_other.shape[0]
                    count_sample8 += 1
                else:
                    end_veg_p = N_POINTS
                # if num points in vegetation > points to sample
                if pc_high_veg.shape[0] > end_veg_p:
                    sampling_indices = np.random.choice(pc_high_veg.shape[0], end_veg_p)
                    pc_high_veg = pc_high_veg[sampling_indices, :]
                    pc_sampled = np.concatenate((pc_other, pc_high_veg), axis=0)
                else:
                    pc_sampled = pc_other

                # if we still have > 1000 in point cloud
                if pc_sampled.shape[0] > N_POINTS:
                    # print(f'Number of points above {TH_2}m: {pc[pc[:, 2] > TH_2].shape[0]}')
                    # rdm sample all point cloud
                    sampling_indices = np.random.choice(pc_sampled.shape[0], N_POINTS)
                    pc_sampled = pc_sampled[sampling_indices, :]
                    count_sample_all += 1

            with open(os.path.join(path_sampled, fileName) + '.pkl', 'wb') as f:
                pickle.dump(pc_sampled, f)

        elif pc.shape[0] == N_POINTS:
            pc_sampled = pc
            with open(os.path.join(path_sampled, fileName) + '.pkl', 'wb') as f:
                pickle.dump(pc_sampled, f)

        ##################################### ADD SYNTHETIC SAMPLES ###########################################
        else:
            if MIN_POINTS < pc.shape[0] < N_POINTS:
                count_interpolate += 1
                # generate points
                # i = 0
                # new_pc = list(pc)
                # # while points < 1000 keep duplicating
                # while (len(new_pc) < N_POINTS):
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
            else:
                if MIN_POINTS > pc.shape[0]:
                    c_min_points += 1

    print(f'counter sampled below {TH_1} m: {count_sample3}')
    print(f'counter sampled below {TH_2} m: {count_sample8}')
    print(f'counter sample all: {count_sample_all}')
    print(f'counter total sampled: {count_sample3 + count_sample8 + count_sample_all}')
    print(f'counter added synthetic data: {count_interpolate}')
    print(f'Discarded point clouds because < {MIN_POINTS} points: ', c_min_points)


if __name__ == '__main__':

    N_POINTS = 2000
    MAX_Z = 100.0
    # paths = ['/home/m.caros/work/objectDetection/datasets/data_BDN/pc_towers_40x40/*.las',
    #          '/home/m.caros/work/objectDetection/datasets/data_BDN/pc_no_towers_40x40/*.las']
    paths = ['/home/m.caros/work/objectDetection/datasets/CAT3/w_towers_40x40',
             '/home/m.caros/work/objectDetection/datasets/CAT3/w_no_towers_40x40']
        # ['/home/m.caros/work/objectDetection/datasets/RIBERA/w_towers_40x40',
        #      '/home/m.caros/work/objectDetection/datasets/RIBERA/w_no_towers_40x40']

    start_time = time.time()

    for files_path in paths:
        print(files_path)

        # execute compute_pdal_bash.sh  # transform HAS data into HAG

        if 'w_towers' in files_path:
            out_path = '/dades/LIDAR/towers_detection/datasets/pc_towers_40x40'
        elif 'w_no_towers' in files_path:
            out_path = '/dades/LIDAR/towers_detection/datasets/pc_no_towers_40x40'

        # ------ Remove ground, noise and outliers and normalize ------
        # logging.info(f"1. Remove points of ground, noise and outliers and normalize ")
        # no_ground_path = remove_ground_and_outliers(files_path, out_path, max_z=MAX_Z, max_intensity=5000)
        # print("--- Remove ground and noise time: %s h ---" % (round((time.time() - start_time) / 3600, 3)))

        rm_ground_time = time.time()
        no_ground_path = os.path.join(out_path, 'data_no_ground')
        print(no_ground_path)

        # ------ sampling ------
        logging.info(f"2. Sampling")
        sampling(no_ground_path, N_POINTS=N_POINTS, TH_1=3.0/MAX_Z, TH_2=8.0/MAX_Z)
        print("--- Sample and interpolate time: %s h ---" % (round((time.time() - rm_ground_time) / 3600, 3)))

    print("--- TOTAL TIME: %s h ---" % (round((time.time() - start_time) / 3600, 3)))
