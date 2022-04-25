import argparse
import glob
import pickle
import time
from progressbar import progressbar
from torch.utils.data import random_split
from datasets import LidarDataset, BdnDataset
from model.pointnetRNN import *
import logging
from utils import *
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

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
         model_checkpoint,
         use_rnn):
    start_time = time.time()

    checkpoint = torch.load(model_checkpoint)

    with open('pointNet/data/towers_files.txt', 'r') as f:
        tower_files = f.read().splitlines()
    with open('pointNet/data/landscape_files.txt', 'r') as f:
        landscape_files = f.read().splitlines()

    logging.info(f'Samples with towers: {len(tower_files)}')
    n_lanscape = int(len(landscape_files) * 0.0002)
    logging.info(f'Samples with landscape for segmentation: {n_lanscape}')
    # split train (80%) / val (10%) / test (10%)
    t_test = round(len(tower_files) * 0.9)
    l_test = round(n_lanscape * 0.9)

    test_dataset = LidarDataset(dataset_folder,
                                task='segmentation',
                                number_of_points=n_points,
                                towers_files=tower_files[t_test:],
                                landscape_files=landscape_files[-l_test:],
                                fixed_num_points=False)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False,
                                                  collate_fn=collate_fn_padd)

    if use_rnn:
        model = RNNSegmentationPointNet(num_classes=test_dataset.NUM_SEGMENTATION_CLASSES,
                                        hidden_size=256,
                                        point_dimension=test_dataset.POINT_DIMENSION)
    else:
        model = SegmentationPointNet(num_classes=test_dataset.NUM_SEGMENTATION_CLASSES,
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
        fid.write('file_name,positive points,% corrects,F1,AUC\n')

    shape_preds = 0
    for data in test_dataloader:

        points, targets, file_name = data  # [1, 2000, 12], [1, 2000]

        points = points.view(1, -1, 12)  # [batch, n_samples, dims]
        targets = targets.view(1, -1)  # [batch, n_samples]

        points, targets = points.to(device), targets.to(device)
        pc_pred = torch.Tensor().to(device)
        model = model.eval()
        if use_rnn:
            hidden = model.initHidden(points)
        j = 0
        while shape_preds < points.shape[1]:
            end_batch = n_points * (j + 1)
            if end_batch < points.shape[1]:
                in_points = points[:, j * n_points: end_batch, :]
            else:
                points_needed = end_batch - points.shape[1]
                rdm_list = np.random.randint(0, points.shape[1], points_needed)
                in_points = points[:, j * n_points:, :]
                extra_points = points[:, rdm_list, :]
                in_points = torch.cat([in_points, extra_points], dim=1)
                # add duplicated targets
                extra_targets = targets[:, rdm_list]
                targets = torch.cat((targets, extra_targets), dim=1)
                points = torch.cat([points, extra_points], dim=1)

            if use_rnn:
                preds, hidden, feature_transform = model(in_points, hidden)  # [batch, n_points, 2] [2, batch, 128]
            else:
                preds, feature_transform = model(in_points)  # [batch, n_points, 2] [2, batch, 128]

            pc_pred = torch.cat((pc_pred, preds), dim=1)
            shape_preds = pc_pred.shape[1]
            j += 1

        shape_preds = 0

        probs = torch.exp(pc_pred.cpu().detach())  # [1, points in pc, 2]
        probs = probs.cpu().numpy().reshape(-1, 2) # num of points is variable in eacy point cloud
        # get max over dim 1
        preds = np.argmax(probs, axis=1)
        targets = targets.reshape(-1).cpu().numpy()

        # calculate F1 score
        lr_f1 = f1_score(targets, preds)

        # keep probabilities for the positive outcome only
        lr_probs = probs[:, 1]
        lr_precision, lr_recall, thresholds = precision_recall_curve(targets, lr_probs)
        lr_auc = auc(lr_recall, lr_precision)

        all_positive = (np.array(targets) == np.ones(len(targets))).sum()  # TP + FN
        detected_positive = (np.array(preds) == np.ones(len(targets)))  # boolean with positions of 1s
        # tp = np.logical_and(corrects, detected_positive).sum()
        # fp = detected_positive.sum() - tp
        corrects = (np.array(preds) == np.array(targets))

        # summarize scores
        file_name = file_name[0].split('/')[-1]
        print(file_name)
        # ptg_corrects = (corrects.sum() / preds.shape[0]) * 100
        ptg_corrects = (detected_positive.sum() / all_positive)*100
        print(f'Ptg corrects: {ptg_corrects}%')
        print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
        print('Positive points: ', all_positive)
        print('-------------')
        with open(os.path.join(output_folder, 'results-%s.csv' % name), 'a') as fid:
            fid.write('%s,%s,%s,%s,%s\n' % (
                file_name, all_positive, round(ptg_corrects,3), round(lr_f1,3),round(lr_auc,3)))

        # store segmentation results in pickle file for plotting
        points = points.reshape(-1, 12)
        preds = preds[..., np.newaxis]
        points = np.concatenate((points.cpu().numpy(), preds), axis=1)
        if use_rnn:
            dir_results = 'segmentation_rnn'
        else:
            dir_results= 'segmentation_regular'
        with open(os.path.join(output_folder, dir_results,  file_name ), 'wb') as f:
            pickle.dump(points, f)

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
    parser.add_argument('--use_rnn', type=bool, default=False, help='True if want to use RNN model')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    sys.path.insert(0, '/home/m.caros/work/objectDetection/pointNet')

    test(args.dataset_folder,
         args.number_of_points,
         args.output_folder,
         args.number_of_workers,
         args.model_checkpoint,
         args.use_rnn)
