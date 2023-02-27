import json
import os
import csv
import torch.utils.data as data
import torch
import glob
import numpy as np
import pickle


class LidarDataset(data.Dataset):
    NUM_CLASSIFICATION_CLASSES = 2
    POINT_DIMENSION = 2

    def __init__(self, dataset_folder,
                 task='classification',
                 number_of_points=None,
                 number_of_windows=None,
                 files=None,
                 fixed_num_points=True,
                 c_sample=False):
        # 0 -> no tower
        # 1 -> tower
        self.dataset_folder = dataset_folder
        self.task = task
        self.n_points = number_of_points
        self.n_windows = number_of_windows
        self.files = files
        self.fixed_num_points = fixed_num_points
        self.classes_mapping = {}
        self.constrained_sampling = c_sample
        self.paths_files = [os.path.join(self.dataset_folder, f) for f in self.files]
        self._init_mapping()

    def __len__(self):
        return len(self.paths_files)

    def _init_mapping(self):

        for file in self.files:
            if 'pc_' in file:
                self.classes_mapping[file] = 0
            elif 'tower_' in file or 'powerline_' in file:
                self.classes_mapping[file] = 1

        self.len_towers = sum(value == 1 for value in self.classes_mapping.values())
        self.len_landscape = sum(value == 0 for value in self.classes_mapping.values())

    def __getitem__(self, index):
        """
        :param index: index of the file
        :return: pc: [n_points, dims], labels, filename
        """
        filename = self.paths_files[index]
        pc = self.prepare_data(filename,
                               self.n_points,
                               fixed_num_points=self.fixed_num_points,
                               constrained_sample=self.constrained_sampling)
        labels = self.get_labels(pc, self.classes_mapping[self.files[index]], self.task)
        pc = np.concatenate((pc[:, :3], pc[:, 4].unsqueeze(1), pc[:, 5:9], pc[:, 9].unsqueeze(1)), axis=1)
        return pc, labels, filename

    @staticmethod
    def prepare_data(point_file,
                     number_of_points=None,
                     fixed_num_points=True,
                     constrained_sample=False):

        with open(point_file, 'rb') as f:
            pc = pickle.load(f).astype(np.float32)  # [17434, 14]

        # if constrained sampling -> get points labeled for sampling
        if constrained_sample:
            pc = pc[pc[:, 10] == 1]  # should be flag of position 10

        # random sample points if fixed_num_points
        if fixed_num_points and pc.shape[0] > number_of_points:
            sampling_indices = np.random.choice(pc.shape[0], number_of_points)
            pc = pc[sampling_indices, :]

        # duplicate points if needed
        elif fixed_num_points and pc.shape[0] < number_of_points:
            points_needed = number_of_points - pc.shape[0]
            rdm_list = np.random.randint(0, pc.shape[0], points_needed)
            extra_points = pc[rdm_list, :]
            pc = np.concatenate([pc, extra_points], axis=0)

        # normalize axes between -1 and 1
        pc[:, 0] = 2 * ((pc[:, 0] - pc[:, 0].min()) / (pc[:, 0].max() - pc[:, 0].min())) - 1
        pc[:, 1] = 2 * ((pc[:, 1] - pc[:, 1].min()) / (pc[:, 1].max() - pc[:, 1].min())) - 1

        pc = torch.from_numpy(pc)
        return pc

    @staticmethod
    def get_labels(pointcloud,
                   point_cloud_class,
                   task='classification'):
        """
        Get labels for classification or segmentation

        Classification labels:
        0 -> No tower (negative)
        1 -> Tower (positive)

        Segmentation labels:
        0 -> background (other classes we're not interested)
        1 -> tower
        2 -> low vegetation
        3 -> medium vegetation
        4 -> high vegetation

        :param pointcloud: [n_points, dim, seq_len]
        :param point_cloud_class: point cloud category
        :param task: classification or segmentation

        :return labels: points with categories to segment or classify
        """
        if task == 'segmentation':
            segment_labels = pointcloud[:, 3]
            segment_labels[segment_labels == 15] = 100
            segment_labels[segment_labels == 14] = 200
            segment_labels[segment_labels == 3] = 300  # low veg
            segment_labels[segment_labels == 4] = 300  # med veg
            segment_labels[segment_labels == 5] = 400
            segment_labels[segment_labels < 100] = 0
            segment_labels = (segment_labels / 100)

            # binary segmentation
            # segment_labels[segment_labels == 15] = 1
            # segment_labels[segment_labels != 15] = 0

            labels = segment_labels.type(torch.LongTensor)  # [2048, 5]

        elif task == 'classification':
            labels = point_cloud_class  # for training data

        return labels
