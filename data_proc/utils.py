import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import glob
from progressbar import progressbar
import laspy
import pickle


def plot_class_points(inFile, fileName, selClass, save_plot=False, point_size=40, save_dir='figures/'):
    """Plot point cloud of a specific class"""

    # get class
    selFile = inFile
    selFile.points = inFile.points[np.where(inFile.classification == selClass)]

    # plot
    fig = plt.figure(figsize=[20, 10])
    ax = plt.axes(projection='3d')
    sc = ax.scatter(selFile.x, selFile.y, selFile.z, c=selFile.z, s=point_size, marker='o', cmap="Spectral")
    plt.colorbar(sc)
    plt.title('Points of class %i of file %s' % (selClass, fileName))
    if save_plot:
        directory = save_dir
        name = 'point_cloud_class_' + str(selClass) + '_' + fileName + '.png'
        plt.savefig(directory + name, bbox_inches='tight', dpi=100)
    plt.show()


def plot_2d_class_points(inFile, fileName, selClass, save_plot=False, point_size=40, save_dir='figures/'):
    """Plot point cloud of a specific class"""

    # get class
    selFile = inFile
    selFile.points = inFile.points[np.where(inFile.classification == selClass)]

    # plot
    fig = plt.figure(figsize=[10, 5])
    sc = plt.scatter(selFile.x, selFile.y, c=selFile.z, s=point_size, marker='o', cmap="viridis")
    plt.colorbar(sc)
    plt.title('Points of class %i of file %s' % (selClass, fileName))
    if save_plot:
        directory = save_dir
        name = 'point_cloud_class_' + str(selClass) + '_' + fileName + '.png'
        plt.savefig(directory + name, bbox_inches='tight', dpi=100)
    plt.show()


def plot_3d_coords(coords, fileName='', selClass=[], save_plot=False, point_size=40, save_dir='figures/',
                   c_map="Spectral",
                   show=True, figsize=[20, 10]):
    """Plot of point cloud. Can be filtered by a specific class"""

    # plot
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    sc = ax.scatter(coords[0], coords[1], coords[2], c=coords[2], s=point_size, marker='o', cmap=c_map)
    plt.colorbar(sc)
    plt.title('Point cloud - file %s' % (fileName))
    if save_plot:
        directory = save_dir
        if selClass:
            name = 'point_cloud_class_' + str(selClass) + '_' + fileName + '.png'
        else:
            name = 'point_cloud_' + fileName + '.png'
        plt.savefig(os.path.join(directory, name), bbox_inches='tight', dpi=100)
    if show:
        plt.show()
    else:
        plt.close()


def plot_2d_coords(coords, ax=[], save_plot=False, point_size=40, figsize=[10, 5], save_dir='figures/'):
    if not ax:
        fig = plt.figure(figsize=figsize)
        sc = plt.scatter(coords[0], coords[2], c=coords[1], s=point_size, marker='o', cmap="viridis")
        plt.colorbar(sc)
    else:
        ax.scatter(coords[1], coords[2], c=coords[2], s=point_size, marker='o', cmap="viridis")
        ax.title.set_text('Points=%i' % (len(coords[1])))


def sliding_window_coords(point_cloud, stepSize_x=10, stepSize_y=10, windowSize=[20, 20], min_points=10,
                          show_prints=False):
    """
    Slide a window across the coords of the point cloud to segment objects.

    :param point_cloud:
    :param stepSize_x:
    :param stepSize_y:
    :param windowSize:
    :param min_points:
    :param show_prints:

    :return: (dict towers, dict center_w)

    Example of return:
    For each window we get the center and the points of the tower
    dict center_w = {'0': {0: [2.9919000000227243, 3.0731000006198883]},...}
    dict towers = {'0': {0: array([[4.88606837e+05, 4.88607085e+05, 4.88606880e+05, ...,]])}...}
    """
    i_w = 0
    last_w_i = 0
    towers = {}
    center_w = {}
    point_cloud = np.array(point_cloud)
    x_min, y_min, z_min = point_cloud[0].min(), point_cloud[1].min(), point_cloud[2].min()
    x_max, y_max, z_max = point_cloud[0].max(), point_cloud[1].max(), point_cloud[2].max()

    # if window is larger than actual point cloud it means that in the point cloud there is only one tower
    if windowSize[0] > (x_max - x_min) and windowSize[1] > (y_max - y_min):
        if show_prints:
            print('Window larger than point cloud')
        if point_cloud.shape[1] >= min_points:
            towers[0] = point_cloud
            # get center of window
            center_w[0] = [point_cloud[0].mean(), point_cloud[1].mean()]
            return towers, center_w
        else:
            return None, None
    else:
        for y in range(round(y_min), round(y_max), stepSize_y):
            # check if there are points in this range of y
            bool_w_y = np.logical_and(point_cloud[1] < (y + windowSize[1]), point_cloud[1] > y)
            if not any(bool_w_y):
                continue
            if y + stepSize_y > y_max:
                continue

            for x in range(round(x_min), round(x_max), stepSize_x):
                i_w += 1
                # check points i window
                bool_w_x = np.logical_and(point_cloud[0] < (x + windowSize[0]), point_cloud[0] > x)
                if not any(bool_w_x):
                    continue
                bool_w = np.logical_and(bool_w_x, bool_w_y)
                if not any(bool_w):
                    continue
                # get coords of points in window
                window = point_cloud[:, bool_w]

                if window.shape[1] >= min_points:
                    # if not first item in dict
                    if len(towers) > 0:
                        # if consecutive windows overlap
                        if last_w_i == i_w - 1: # or last_w_i == i_w - 2:
                            # if more points in new window -> store w, otherwise do not store
                            if window.shape[1] > towers[list(towers)[-1]].shape[1]:
                                towers[list(towers)[-1]] = window
                                center_w[list(center_w)[-1]] = [window[0].mean(), window[1].mean()]

                                last_w_i = i_w
                                if show_prints:
                                    print('Overlap window %i key %i --> %s points' % (i_w, list(towers)[-1], str(window.shape)))
                        else:
                            towers[len(towers)] = window
                            center_w[len(center_w)] = [window[0].mean(), window[1].mean()]
                            last_w_i = i_w
                            if show_prints:
                                print('window %i key %i --> %s points' % (i_w, list(towers)[-1], str(window.shape)))

                    else:
                        towers[len(towers)] = window
                        center_w[len(center_w)] = [window[0].mean(), window[1].mean()]
                        last_w_i = i_w
                        if show_prints:
                            print('window %i key %i --> %s points' % (i_w, list(towers)[-1], str(window.shape)))

        return towers, center_w


def remove_outliers(files_path, max_z=100.0):
    dir_path = os.path.dirname(files_path)
    path_norm_dir = os.path.join(dir_path, 'data_without_outliers')
    if not os.path.exists(path_norm_dir):
        os.makedirs(path_norm_dir)

    files = glob.glob(os.path.join(files_path, '*.las'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        data_f = laspy.read(file)

        try:
            # check file is not empty
            if len(data_f.x) > 0:

                points = np.vstack((data_f.x, data_f.y, data_f.HeightAboveGround, data_f.classification,
                                    data_f.intensity,
                                    data_f.return_number,
                                    data_f.red,
                                    data_f.green,
                                    data_f.blue
                                    ))

                # Remove outliers (points above max_z)
                points = points[:, points[2] <= max_z]
                # Remove points z < 0
                points = points[:, points[2] >= 0]

                if points[2].max() > max_z:
                    print('Outliers not removed correctly!!')

                if points.shape[1] > 0:
                    f_path = os.path.join(path_norm_dir, fileName)
                    with open(f_path + '.pkl', 'wb') as f:
                        pickle.dump(points, f)
            else:
                print(f'File {fileName} is empty')
        except Exception as e:
            print(f'Error {e} in file {fileName}')


def normalize_LAS_data(files_path, max_z=100.0):
    dir_path = os.path.dirname(files_path)
    path_norm_dir = os.path.join(dir_path, 'dataset_input_model')
    if not os.path.exists(path_norm_dir):
        os.makedirs(path_norm_dir)

    files = glob.glob(os.path.join(files_path, '*.las'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        data_f = laspy.read(file)

        try:
            # check file is not empty
            if len(data_f.x) > 0:
                # normalize axes
                data_f.x = (data_f.x - data_f.x.min()) / (data_f.x.max() - data_f.x.min())
                data_f.y = (data_f.y - data_f.y.min()) / (data_f.y.max() - data_f.y.min())
                data_f.HeightAboveGround = data_f.HeightAboveGround / max_z

                points = np.vstack((data_f.x, data_f.y, data_f.HeightAboveGround, data_f.classification))

                # Remove outliers (points above max_z)
                points = points[:, points[2] <= 1]
                # Remove points z < 0
                points = points[:, points[2] >= 0]

                if points[2].max() > 1:
                    print('Outliers not removed correctly!!')

                if points.shape[1] > 0:
                    f_path = os.path.join(path_norm_dir, fileName)
                    with open(f_path + '.pkl', 'wb') as f:
                        pickle.dump(points, f)
            else:
                print(f'File {fileName} is empty')
        except Exception as e:
            print(f'Error {e} in file {fileName}')


def normalize_pickle_data(files_path, max_z=100.0, max_intensity=5000, dir_name=''):
    dir_path = os.path.dirname(files_path)
    path_out_dir = os.path.join(dir_path, dir_name)
    if not os.path.exists(path_out_dir):
        os.makedirs(path_out_dir)

    files = glob.glob(os.path.join(files_path, '*.pkl'))
    for file in progressbar(files):
        fileName = file.split('/')[-1].split('.')[0]
        with open(file, 'rb') as f:
            pc = pickle.load(f)
        # print(pc.shape)  # [1000,4]
        # try:
            # check file is not empty
        if pc.shape[0] > 0:
            # normalize axes
            pc[:, 0] = (pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())
            pc[:, 1] = (pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())
            pc[:, 2] = pc[:, 2] / max_z

            # normalize intensity
            pc[:, 4] = pc[:, 4] / max_intensity
            pc[:, 4] = np.clip(pc[:, 4], 0, max_intensity)

            # return number
            # number of returns

            # normalize color
            pc[:, 7] = pc[:, 7] / 65536.0
            pc[:, 8] = pc[:, 8] / 65536.0
            pc[:, 9] = pc[:, 9] / 65536.0

            # todo add nir and ndv

            # Remove outliers (points above max_z)
            pc = pc[pc[:, 2] <= 1]
            # Remove points z < 0
            pc = pc[pc[:, 2] >= 0]

            if pc[:, 2].max() > 1:
                print('Outliers not removed correctly!!')

            if pc.shape[0] > 0:
                f_path = os.path.join(path_out_dir, fileName)
                with open(f_path + '.pkl', 'wb') as f:
                    pickle.dump(pc, f)
        else:
            print(f'File {fileName} is empty')
        # except Exception as e:
        #     print(f'Error {e} in file {fileName}')











