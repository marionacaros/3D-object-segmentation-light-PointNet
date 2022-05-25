import argparse
import glob
import time

from progressbar import progressbar
from torch.utils.data import random_split
from datasets import LidarDataset
# from model.light_pointnet import ClassificationPointNet
# from model.light_pointnet_IGBVI import ClassificationPointNet_IGBVI
from model.pointnet import *

import logging
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

    checkpoint = torch.load(model_checkpoint)

    # Datasets train / test
    RGBN = True
    if RGBN:
        with open('pointNet/data/RGBN/RGBN_test_moved_towers_files.txt', 'r') as f:
            tower_files = f.read().splitlines()
        with open('pointNet/data/RGBN/RGBN_test_landscape_files.txt', 'r') as f:
            landscape_files = f.read().splitlines()
    else:
        with open('pointNet/data/test_moved_towers_files.txt', 'r') as f:
            tower_files = f.read().splitlines()
        with open('pointNet/data/test_landscape_files.txt', 'r') as f:
            landscape_files = f.read().splitlines()

    path_dataset = os.path.join(dataset_folder, 'pc_towers_40x40', 'sampled_2048')
    logging.info(f'Dataset path: {path_dataset}')

    test_dataset = LidarDataset(dataset_folder=path_dataset,
                                          task=task, number_of_points=number_of_points,
                                          towers_files = tower_files,
                                          landscape_files = landscape_files,
                                          fixed_num_points = True)

    logging.info(f'Samples for validation: {len(test_dataset)}')
    logging.info(f'Samples with towers in TEST: {test_dataset.len_towers}')
    logging.info(f'Samples without towers in TEST: {test_dataset.len_landscape}')


    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False)
    if RGBN:
        model = ClassificationPointNet(num_classes=test_dataset.NUM_CLASSIFICATION_CLASSES,
                                       point_dimension=test_dataset.POINT_DIMENSION,
                                       dataset=test_dataset)
    else:
        model = ClassificationPointNet(num_classes=test_dataset.NUM_CLASSIFICATION_CLASSES,
                                       point_dimension=test_dataset.POINT_DIMENSION,
                                       dataset=test_dataset)

    if torch.cuda.is_available():
        logging.info(f"cuda available")
        model.cuda()

    logging.info('Loading checkpoint')
    model.load_state_dict(checkpoint['model'])
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    name = model_checkpoint.split('/')[-1]
    print(name)
    # with open(os.path.join(output_folder, 'results-%s.csv' % name), 'w+') as fid:
    #     fid.write('point_cloud,prob[0],prob[1],pred,target\n')
    with open(os.path.join(output_folder, 'wrong_predictions-%s.csv' % name), 'w+') as fid:
        fid.write('file_name,prob[0],prob[1],pred,target\n')
    with open(os.path.join(output_folder, 'results-positives-%s.csv' % name), 'w+') as fid:
        fid.write('file_name\n')
    # store file names for segmentation
    # with open(os.path.join(output_folder, 'files-segmentation-%s.csv' % name), 'w+') as fid:
    #     fid.write('file_name\n')

    all_preds = []
    all_probs = []
    targets = []
    targets_real = []

    for data in progressbar(test_dataloader):

        pc, target, file_name = data  # [1, 2000, 9], [1]
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

        targets_real.append(target.item())
        if 16 in set(clases) and pred.item() == 1:
            targets.append(1)
        else:
            targets.append(target.item())

        if target.item() == 1 or pred.item() == 1:
            with open(os.path.join(output_folder, 'results-positives-%s.csv' % name), 'a') as fid:
                fid.write('%s\n' % (file_name[0].split('/')[-1]))

        if pred.item() != target.item():
            # print(f'Wrong prediction in: {file_name}')
            with open(os.path.join(output_folder, 'wrong_predictions-%s.csv' % name), 'a') as fid:
                fid.write('%s,%s,%s,%s,%s\n' % (file_name[0], probs[0,0].item(), probs[0,1].item(), pred.item(), target.item()))
            # if target.item() == 0:
            #     with open(os.path.join(output_folder, 'files-segmentation-%s.csv' % name), 'a') as fid:
            #         fid.write('%s\n' % (file_name[0]))

    epochs = checkpoint['epoch']
    print(f'Model trained for {epochs} epochs')
    print('--------  considering ALL TOWERS as correct -------')

    # --------  considering all type of towers detected as correct -------
    # calculate F1 score
    lr_f1 = f1_score(targets, all_preds)

    # keep probabilities for the positive outcome only
    lr_probs = np.array(all_probs).transpose()[1] # [2, len(test data)]
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

    # --------  considering ONLY TRANSMISSION TOWERS as correct -------

    print('--------  considering ONLY TRANSMISSION TOWERS as correct -------')
    # calculate F1 score
    lr_f1 = f1_score(targets_real, all_preds)

    # keep probabilities for the positive outcome only
    lr_probs = np.array(all_probs).transpose()[1] # [2, len(test data)]
    lr_precision, lr_recall, thresholds = precision_recall_curve(targets_real, lr_probs)
    lr_auc = auc(lr_recall, lr_precision)

    corrects = (np.array(all_preds) == np.array(targets_real))  # boolean with matched predictions
    detected_positive = (np.array(all_preds) == np.ones(len(all_preds)))  # boolean with positions of 1s
    all_positive = (np.array(targets_real) == np.ones(len(targets_real))).sum()  # TP + FN

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
    with open('pointNet/json_files/precision-recall-%s.json' % name, 'w') as f:
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

# python pointNet/test_classification.py /dades/LIDAR/towers_detection/datasets classification pointNet/results/ --weighing_method EFS --number_of_points 2048 --number_of_workers 0 --model_checkpoint /home/m.caros/work/objectDetection/pointNet/checkpoints/checkpoint_05-11-12:540.999.pth