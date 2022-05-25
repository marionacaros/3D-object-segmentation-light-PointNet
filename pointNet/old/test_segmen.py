import argparse
import glob
import time
from progressbar import progressbar
from torch.utils.data import random_split
from datasets import LidarDataset
from model.light_pointnet_IGBVI import SegmentationPointNet_IGBVI
import logging
from utils import *
import json
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc


def test(dataset_folder,
         number_of_points,
         output_folder,
         number_of_workers,
         model_checkpoint):
    start_time = time.time()

    checkpoint = torch.load(model_checkpoint)

    # path = '/home/m.caros/work/objectDetection/pointNet/results/files-segmentation-checkpoint_05-11-12:540.999.pth.csv'
    # df = pd.read_csv(path)
    # detected_positive = list(df['file_name'])

    with open('../data/RGBN/RGBN_test_moved_towers_files.txt', 'r') as f:
        tower_files = f.read().splitlines()
    with open('../data/RGBN/RGBN_test_landscape_files.txt', 'r') as f:
        landscape_files = f.read().splitlines()

    test_dataset = LidarDataset(dataset_folder=os.path.join(dataset_folder, 'test', 'filtered'),
                                          task='segmentation', number_of_points=number_of_points,
                                          towers_files = tower_files,
                                          landscape_files = landscape_files,
                                          fixed_num_points = True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=number_of_workers,
                                                  drop_last=False)

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
    # with open(os.path.join(output_folder, 'results-%s.csv' % name), 'w+') as fid:
    #     fid.write('point_cloud,prob[0],prob[1],pred,targets\n')

    for data in test_dataloader:

        pc, targets, file_name = data  # [1, 2000, 9], [1]
        if torch.cuda.is_available():
            pc, targets = pc.cuda(), targets.cuda()
        model = model.eval()

        log_probs, feature_transform = model(pc)
        probs = torch.exp(log_probs.cpu().detach())  # [1, 2000, 2]
        probs = probs.cpu().numpy().reshape(2048, 2)
        # get max over dim 1
        preds = np.argmax(probs, axis=1)
        targets = targets.reshape(2048).cpu().numpy()

        # pc = pc.reshape(2000, -1)
        # preds = preds[..., np.newaxis]
        # pc = np.concatenate((pc.cpu().numpy(), preds), axis=1)
        # with open('results/segmentation/' + file_name[0] + '_pred.pkl', 'wb') as f:
        #     pickle.dump(pc, f)

        # calculate F1 score
        lr_f1 = f1_score(targets, preds)

        # keep probabilities for the positive outcome only
        lr_probs = probs[:, 1]
        lr_precision, lr_recall, thresholds = precision_recall_curve(targets, lr_probs)
        lr_auc = auc(lr_recall, lr_precision)

        all_positive = (np.array(targets) == np.ones(len(targets))).sum()  # TP + FN
        # detected_positive = (np.array(all_preds) == np.ones(len(all_preds)))  # boolean with positions of 1s
        # tp = np.logical_and(corrects, detected_positive).sum()
        # fp = detected_positive.sum() - tp
        corrects = (np.array(preds) == np.array(targets))

        # summarize scores
        print(file_name[0])
        print(f'Ptg corrects: {(corrects.sum()/2048)*100}%')
        print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
        print('All positive: ', all_positive)
        print('-------------')

        # data = {"lr_recall": str(list(lr_recall)),
        #         "lr_precision": str(list(lr_precision))}

        # with open('json_files/precision-recall-%s.json' % name, 'w') as f:
        #     json.dump(data, f)
    epochs = checkpoint['epoch']
    print(f'Model trained for {epochs} epochs')
    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('output_folder', type=str, help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2048, help='number of points per cloud')
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
