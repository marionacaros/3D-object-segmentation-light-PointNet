import argparse
import glob
import time

import matplotlib.pyplot as plt
from progressbar import progressbar
from torch.utils.data import random_split
from datasets import LidarDataset, BdnDataset
from model.pointnet import ClassificationPointNet, SegmentationPointNet
import logging
import datetime
from utils import *
import json

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc


def test(dataset_folder,
         task,
         number_of_points,
         weighing_method,
         output_folder,
         number_of_workers,
         model_checkpoint):
    start_time = time.time()
    logging.info(f"Weighing method: {weighing_method}")
    BETA = 0.999
    batch_size = 1

    checkpoint = torch.load(model_checkpoint)

    # Tensorboard location and plot names
    # now = datetime.datetime.now()
    # location = 'runs/tower_detec/' + str(number_of_points) + 'points/'
    # if not os.path.isdir(output_folder):
    #     os.mkdir(output_folder)
    # writer_test = SummaryWriter(location + now.strftime("%m-%d-%H:%M_") + weighing_method + '_train')
    # logging.info(f"Tensorboard runs: {writer_test.get_logdir()}")

    # Datasets train / test
    train_dataset = LidarDataset(os.path.join(dataset_folder, 'train'), task=task, number_of_points=number_of_points)
    logging.info(f'Samples for training: {len(train_dataset)}')

    towers_files = glob.glob(os.path.join(dataset_folder, 'test/towers_2000/*.pkl'))
    test_dataset = LidarDataset('', task=task, number_of_points=number_of_points,
                                          files_segmentation=towers_files)
    logging.info(f'Samples for validation: {len(test_dataset)}')

    logging.info(f'Task: {train_dataset.task}')
    logging.info(f'Samples with towers in TRAIN: {train_dataset.len_towers}')
    logging.info(f'Samples without towers in TRAIN: {train_dataset.len_landscape}')
    logging.info(f'Samples with towers in TEST: {test_dataset.len_towers}')
    logging.info(f'Samples without towers in TEST: {test_dataset.len_landscape}')

    # if using BDN dataset to test (only models without intensity or RGB)
    # bdn_dataset = BdnDataset(dataset_folder, task=task, number_of_points=number_of_points)
    # test_dataset = torch.utils.data.ConcatDataset([test_dataset, bdn_dataset])
    # l_test = len(test_dataset)
    # logging.info(f'Samples test dataset + BDN dataset: {l_test}')

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False)
    if task == 'classification':
        model = ClassificationPointNet(num_classes=train_dataset.NUM_CLASSIFICATION_CLASSES,
                                       point_dimension=train_dataset.POINT_DIMENSION,
                                       dataset=train_dataset)
    elif task == 'segmentation':
        model = SegmentationPointNet(num_classes=train_dataset.NUM_SEGMENTATION_CLASSES,
                                     point_dimension=train_dataset.POINT_DIMENSION)
    else:
        raise Exception('Unknown task !')

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
        fid.write('point_cloud,prob[0],prob[1],pred,target\n')
    with open(os.path.join(output_folder, 'wrong_predictions-%s.csv' % name), 'w+') as fid:
        fid.write('file_name,prob[0],prob[1],pred,target\n')
    # files for segmentation
    with open(os.path.join(output_folder, 'files-segmentation-%s.csv' % name), 'w+') as fid:
        fid.write('file_name\n')

    all_preds = []
    all_probs = []
    targets = []
    for data in progressbar(test_dataloader):

        pc, target, file_name = data  # [1, 2000, 9], [1]
        # points = pc[:, :, :3]
        if torch.cuda.is_available():
            pc, target = pc.cuda(), target.cuda()
        model = model.eval()

        log_probs, feature_transform = model(pc)  # [1,2], [1, 64, 64]  epoch=0, target=target, fileName=file_name
        probs = torch.exp(log_probs.cpu().detach())  # [1, 2]
        all_probs.append(probs.numpy().reshape(2))

        # .max(1) takes the max over dimension 1 and returns two values (the max value in each row and the column index
        # at which the max value is found)
        pred = probs.data.max(1)[1]
        all_preds.append(pred.item())

        # if other tower classified as tower we count it as correct
        clases = pc[:, :, 3].view(-1).cpu().numpy().astype('int')
        if 16 in set(clases) and pred.item() == 1:
            targets.append(1)
        else:
            targets.append(target.item())

        if target.item() == 1 or pred.item() == 1:
            with open(os.path.join(output_folder, 'results-positives-%s.csv' % name), 'a') as fid:
                fid.write('%s,%s,%s,%s,%s\n' % (
                file_name[0], probs[0, 0].item(), probs[0, 1].item(), pred.item(), target.item()))

        if pred.item() != target.item():
            # print(f'Wrong prediction in: {file_name}')
            with open(os.path.join(output_folder, 'wrong_predictions-%s.csv' % name), 'a') as fid:
                fid.write('%s,%s,%s,%s,%s\n' % (file_name[0], probs[0,0].item(), probs[0,1].item(), pred.item(), target.item()))
            if target.item() == 0:
                with open(os.path.join(output_folder, 'files-segmentation-%s.csv' % name), 'a') as fid:
                    fid.write('/dades/LIDAR/towers_detection/datasets/test/landscape_2000/%s\n' % (file_name[0]))

    epochs = checkpoint['epoch']
    print(f'Model trained for {epochs} epochs')

    # targets = np.ones(len(targets)) - targets
    # all_preds = np.ones(len(all_preds)) - all_preds

    # calculate F1 score
    lr_f1 = f1_score(targets, all_preds)

    all_probs = np.array(all_probs).transpose()  # [2, len(test data)]
    # keep probabilities for the positive outcome only
    lr_probs = all_probs[1]

    lr_precision, lr_recall, thresholds = precision_recall_curve(targets, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)

    corrects = (np.array(all_preds) == np.array(targets))  # boolean with matched predictions
    detected_positive = (np.array(all_preds) == np.ones(len(all_preds)))  # boolean with positions of 1s
    all_positive = (np.array(targets) == np.ones(len(targets))).sum()  # TP + FN

    tp = np.logical_and(corrects, detected_positive).sum()
    fp = detected_positive.sum() - tp

    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    print('All positive: ', all_positive)
    print('TP: ', tp)
    print('FP: ', fp)

    if task == 'classification':
        print(f'Accuracy: {round(corrects.sum() / len(test_dataset) * 100, 2)} %')
        # Recall - Del total de torres, quin percentatge s'han trobat?
        print(f'Recall: {round(tp / all_positive * 100, 2)} %')
        # Precision - De les que s'han detectat com a torres quin percentatge era realment torre?
        print(f'Precision: {round(tp / detected_positive.sum() * 100, 2)} %')

    data = {"lr_recall": str(list(lr_recall)),
            "lr_precision": str(list(lr_precision))}

    with open('json_files/precision-recall-%s.json' % name, 'w') as f:
        json.dump(data, f)

    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('task', type=str, choices=['classification', 'segmentation'], help='type of task')
    parser.add_argument('output_folder', type=str, help='output folder')
    parser.add_argument('--number_of_points', type=int, default=1000, help='number of points per cloud')
    parser.add_argument('--weighing_method', type=str, default='ISNS', help='sample weighing method')
    parser.add_argument('--number_of_workers', type=int, default=0, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    sys.path.insert(0, '/home/m.caros/work/objectDetection/pointNet')

    test(args.dataset_folder,
         args.task,
         args.number_of_points,
         args.weighing_method,
         args.output_folder,
         args.number_of_workers,
         args.model_checkpoint)
