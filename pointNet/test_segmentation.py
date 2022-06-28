import argparse
import glob
import pickle
import time
from progressbar import progressbar
from torch.utils.data import random_split
from datasets import LidarDataset
from model.pointnetRNN import *
import logging
from utils import *
# from model.pointnet import *
from model.light_pointnet_IGBVI import *


if torch.cuda.is_available():
    logging.info(f"cuda available")
    device = 'cuda'
else:
    logging.info(f"cuda not available")
    device = 'cpu'

def test(dataset_folder,
         n_points,
         output_folder,
         number_of_workers,
         model_checkpoint):
    start_time = time.time()

    checkpoint = torch.load(model_checkpoint)
    mean_iou_tower = []
    mean_iou_veg = []
    with open('pointNet/data/RGBN/RGBN_test_moved_towers_files.txt', 'r') as f:
        tower_files = f.read().splitlines()
    with open('pointNet/data/RGBN/RGBN_test_landscape_files.txt', 'r') as f:
        landscape_files = f.read().splitlines()

    logging.info(f'Samples with towers: {len(tower_files)}')
    n_lanscape = int(len(landscape_files) * 0.01)

    landscape_files = landscape_files[:n_lanscape]
    logging.info(f'Samples with landscape for segmentation: {n_lanscape}')
    dataset_folder=dataset_folder + '/pc_towers_40x40/sampled_4096'
    test_dataset = LidarDataset(dataset_folder,
                                task='segmentation',
                                number_of_points=n_points,
                                towers_files=tower_files,
                                landscape_files=landscape_files,
                                fixed_num_points=False)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False,
                                                  collate_fn=collate_segmen_padd)


    model = SegmentationPointNet_IGBVI(num_classes=test_dataset.NUM_SEGMENTATION_CLASSES,
                                     point_dimension=test_dataset.POINT_DIMENSION)

    if torch.cuda.is_available():
        logging.info(f"cuda available")
        model.cuda()

    logging.info('Loading checkpoint')
    model.load_state_dict(checkpoint['model'])
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    name = model_checkpoint.split('/')[-1]
    print(name)
    with open(os.path.join(output_folder, 'results-%s.csv' % name), 'w+') as fid:
        fid.write('file_name,positive points,IOU_tower\n')

    for data in test_dataloader:
        points, targets, file_name = data  # [1, 2000, 12], [1, 2000]
        points = points.view(1, -1, 11)  # [batch, n_samples, dims]
        targets = targets.view(1, -1)  # [batch, n_samples]

        points, targets = points.to(device), targets.to(device)
        model = model.eval()

        pc_pred, feature_transform = model(points)  # [batch, n_points, 2] [2, batch, 128]

        probs = torch.exp(pc_pred.cpu().detach())  # [1, points in pc, 2]
        probs = probs.cpu().numpy().reshape(-1, 2) # num of points is variable in each point cloud
        # get max over dim 1
        preds = np.argmax(probs, axis=1)
        targets = targets.reshape(-1).cpu().numpy()

        all_positive = (np.array(targets) == np.ones(len(targets))).sum()  # TP + FN
        all_neg = (np.array(targets) == np.zeros(len(targets))).sum()  # TN + FP
        detected_positive = (np.array(preds) == np.ones(len(targets)))  # boolean with positions of 1s
        detected_negative = (np.array(preds) == np.zeros(len(targets)))  # boolean with positions of 1s

        corrects = np.array(np.array(preds) == np.array(targets))
        tp = np.logical_and(corrects, detected_positive).sum()
        tn = np.logical_and(corrects, detected_negative).sum()
        fp = np.array(detected_positive).sum() - tp
        fn = np.array(detected_negative).sum() - tn

        # summarize scores
        file_name = file_name[0].split('/')[-1]
        print(file_name)
        print('detected_positive: ', np.array(detected_positive).sum())

        iou_tower=0
        # if detected_positive.any() > 0:
        iou_veg = tn / (all_neg + fn)

        if all_positive.sum() > 0:
            iou_tower = tp  /(all_positive + fp )
            print('IOU tower: ', iou_tower)
            print('IOU veg: ', iou_tower)
            mean_iou_tower.append(iou_tower)
        mean_iou_veg.append(iou_veg)
        print('-------------')

        # mean_ptg_corrects.append(ptg_corrects)
        with open(os.path.join(output_folder, 'results-%s.csv' % name), 'a') as fid:
            fid.write('%s,%s,%s\n' % (file_name, all_positive, round(iou_tower,3)))

        # store segmentation results in pickle file for plotting
        points = points.reshape(-1, 11)
        print(points.shape)
        preds = preds[..., np.newaxis]
        print(preds.shape)

        points = np.concatenate((points.cpu().numpy(), preds), axis=1)
        dir_results= 'segmentation_regular'
        with open(os.path.join(output_folder, dir_results,  file_name ), 'wb') as f:
            pickle.dump(points, f)

    mean_iou_tower = np.array(mean_iou_tower).sum()/len(mean_iou_tower)
    mean_iou_veg = np.array(mean_iou_veg).sum()/len(mean_iou_veg)
    print('-------------')
    print('mean_iou_tower: ',mean_iou_tower)
    print('mean_iou_veg: ',mean_iou_veg)
    print('mean_iou: ',(mean_iou_tower + mean_iou_veg) / 2)
    epochs = checkpoint['epoch']
    print(f'Model trained for {epochs} epochs')
    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('output_folder', type=str, help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2000, help='number of points per cloud')
    parser.add_argument('--number_of_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    sys.path.insert(0, '/home/m.caros/work/objectDetection/pointNet')

    test(args.dataset_folder,
         args.number_of_points,
         args.output_folder,
         args.number_of_workers,
         args.model_checkpoint)

# python pointNet/test_segmentation.py /dades/LIDAR/towers_detection/datasets pointNet/results/ --number_of_points 4096 --number_of_workers 0 --model_checkpoint
