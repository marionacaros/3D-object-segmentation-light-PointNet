import argparse

import torch.optim as optim
import torch.nn.functional as F
import time
from progressbar import progressbar
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import LidarDataset
from model.light_pointnet_IGBVI import SegmentationPointNet_IGBVI
# from model.light_pointnet import SegmentationPointNet
# from model.pointnet import *
import logging
import datetime
from sklearn.metrics import balanced_accuracy_score
import warnings
from utils import *
import glob
from prettytable import PrettyTable

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    logging.info(f"cuda available")
    device = 'cuda'
else:
    logging.info(f"cuda not available")
    device = 'cpu'


def train(
        dataset_folder,
        path_list_files,
        output_folder,
        n_points,
        batch_size,
        epochs,
        learning_rate,
        weighing_method,
        beta,
        number_of_workers,
        model_checkpoint,
        c_sample=False):
    start_time = time.time()
    logging.info(f"Weighing method: {weighing_method}")
    logging.info(f"Constrained sampling: {c_sample}")

    # Tensorboard location and plot names
    now = datetime.datetime.now()
    location = 'pointNet/runs/tower_detec/' + str(n_points) + 'p/'

    # Datasets train / val / test
    with open(os.path.join(path_list_files, 'train_seg_files.txt'), 'r') as f:
        train_files = f.read().splitlines()
    with open(os.path.join(path_list_files, 'val_seg_files.txt'), 'r') as f:
        val_files = f.read().splitlines()

    writer_train = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'seg_train')
    writer_val = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'seg_val')
    logging.info(f"Tensorboard runs: {writer_train.get_logdir()}")

    # Initialize datasets
    train_dataset = LidarDataset(dataset_folder=dataset_folder,
                                 task='segmentation', number_of_points=n_points,
                                 files=train_files,
                                 fixed_num_points=True,
                                 c_sample=c_sample)
    val_dataset = LidarDataset(dataset_folder=dataset_folder,
                               task='segmentation', number_of_points=n_points,
                               files=val_files,
                               fixed_num_points=True,
                               c_sample=c_sample)

    logging.info(f'Towers PC in train: {train_dataset.len_towers}')
    logging.info(f'Landscape PC in train: {train_dataset.len_landscape}')
    logging.info(
        f'Proportion towers/landscape: {round((train_dataset.len_towers / (train_dataset.len_towers + train_dataset.len_landscape)) * 100, 3)}%')
    logging.info(f'Towers PC in val: {val_dataset.len_towers}')
    logging.info(f'Landscape PC in val: {val_dataset.len_landscape}')
    logging.info(
        f'Proportion towers/landscape: {round((val_dataset.len_towers / (val_dataset.len_towers + val_dataset.len_landscape)) * 100, 3)}%')
    logging.info(f'Samples for training: {len(train_dataset)}')
    logging.info(f'Samples for validation: {len(val_dataset)}')
    logging.info(f'Task: {train_dataset.task}')

    # Datalaoders
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=number_of_workers,
                                                   drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=number_of_workers,
                                                 drop_last=True)

    model = SegmentationPointNet_IGBVI(num_classes=train_dataset.NUM_SEGMENTATION_CLASSES,
                                       point_dimension=train_dataset.POINT_DIMENSION)
    model.to(device)

    # print model and parameters
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    # print(table)
    logging.info(f"Total Trainable Params: {total_params}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if model_checkpoint:
        print('Loading checkpoint')
        checkpoint = torch.load(model_checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    best_vloss = 1_000_000.
    epochs_since_improvement = 0

    for epoch in progressbar(range(epochs), redirect_stdout=True):
        epoch_train_loss = []
        epoch_train_acc = []
        all_weights = []
        epoch_val_loss = []
        epoch_val_acc = []
        epoch_train_acc_w = []
        epoch_val_acc_w = []
        detected_positive = []
        detected_negative = []
        targets_pos = []
        targets_neg = []

        if epochs_since_improvement == 10:
            adjust_learning_rate(optimizer, 0.5)
        elif epoch == 10:
            adjust_learning_rate(optimizer, 0.5)

        # --------------------------------------------- train loop ---------------------------------------------
        for data in train_dataloader:
            points, targets, filenames = data  # [7557, batch, dims], [4]

            points = points.view(batch_size, n_points, -1).to(device)  # [batch, n_samples, dims]
            targets = targets.view(batch_size, -1).to(device)  # [batch, n_samples]

            # Pytorch accumulates gradients. We need to clear them out before each instance
            optimizer.zero_grad()
            model = model.train()
            preds, feature_transform = model(points)

            preds = preds.view(-1, train_dataset.NUM_SEGMENTATION_CLASSES)
            targets = targets.view(-1)

            # get weights for imbalanced loss
            points_tower = (np.array(targets.cpu()) == np.ones(len(targets))).sum()
            points_landscape = (np.array(targets.cpu()) == np.zeros(len(targets))).sum()

            if not points_tower:
                points_tower = 100
                points_landscape = 4000

            c_weights = get_weights4class(weighing_method,
                                          n_classes=2,
                                          samples_per_cls=[points_landscape, points_tower],
                                          beta=beta).to(device)
            sample_weights = get_weights4sample(c_weights.cpu(), labels=targets).numpy()

            identity = torch.eye(feature_transform.shape[-1]).to(device)

            regularization_loss = torch.norm(identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1)))
            loss = F.nll_loss(preds, targets, weight=c_weights) + 0.001 * regularization_loss
            epoch_train_loss.append(loss.cpu().item())

            loss.backward()
            optimizer.step()

            preds = preds.data.max(1)[1]
            corrects = preds.eq(targets.data).cpu().sum()
            accuracy = corrects.item() / float(targets.shape[0])
            accuracy_w = balanced_accuracy_score(targets.cpu(), preds.cpu(), sample_weight=sample_weights)
            epoch_train_acc_w.append(accuracy_w)
            epoch_train_acc.append(accuracy)
            all_weights.append(c_weights[1].cpu())

        # --------------------------------------------- val loop ---------------------------------------------

        with torch.no_grad():
            for data in val_dataloader:
                points, targets, filenames = data  # [7557, 4, 12]

                points = points.view(batch_size, n_points, -1).to(device)  # [batch, n_samples, dims]
                targets = targets.view(batch_size, -1).to(device)  # [batch, n_samples]

                optimizer.zero_grad()
                model = model.eval()
                preds, feature_transform = model(points)

                preds = preds.view(-1, train_dataset.NUM_SEGMENTATION_CLASSES)
                targets = targets.view(-1)

                # get weights for imbalanced loss
                points_tower = (np.array(targets.cpu()) == np.ones(len(targets))).sum()
                points_landscape = (np.array(targets.cpu()) == np.zeros(len(targets))).sum()

                # weights
                c_weights = get_weights4class(weighing_method,
                                              n_classes=2,
                                              samples_per_cls=[points_landscape, points_tower],
                                              beta=beta).to(device)
                sample_weights = get_weights4sample(c_weights.cpu(), labels=targets).numpy()

                loss = F.nll_loss(preds, targets, weight=c_weights)
                epoch_val_loss.append(loss.cpu().item())
                preds = preds.data.max(1)[1]
                corrects = preds.eq(targets.data).cpu().sum()

                # sum of targets in batch
                targets_pos.append((np.array(targets.cpu()) == np.ones(len(targets))).sum())
                targets_neg.append((np.array(targets.cpu()) == np.zeros(len(targets))).sum())
                detected_positive.append(
                    (np.array(preds.cpu()) == np.ones(len(preds))).sum())  # bool with positions of 1s
                detected_negative.append(
                    (np.array(preds.cpu()) == np.zeros(len(preds))).sum())  # bool with positions of 0s

                accuracy = corrects.item() / float(targets.shape[0])
                accuracy_w = balanced_accuracy_score(targets.cpu(), preds.cpu(), sample_weight=sample_weights)
                epoch_val_acc_w.append(accuracy_w)
                epoch_val_acc.append(accuracy)
        # ------------------------------------------------------------------------------------------------------
        # Tensorboard
        writer_train.add_scalar('loss', np.mean(epoch_train_loss), epoch)
        writer_val.add_scalar('loss', np.mean(epoch_val_loss), epoch)
        writer_train.add_scalar('mean_detected_positive', np.mean(targets_pos), epoch)
        writer_val.add_scalar('mean_detected_positive', np.mean(detected_positive), epoch)
        writer_train.add_scalar('mean_detected_negative', np.mean(targets_neg), epoch)
        writer_val.add_scalar('mean_detected_negative', np.mean(detected_negative), epoch)
        writer_train.add_scalar('accuracy_weighted', np.mean(epoch_train_acc_w), epoch)
        writer_val.add_scalar('accuracy_weighted', np.mean(epoch_val_acc_w), epoch)
        writer_train.add_scalar('accuracy', np.mean(epoch_train_acc), epoch)
        writer_val.add_scalar('accuracy', np.mean(epoch_val_acc), epoch)
        writer_val.add_scalar('epochs_since_improvement', epochs_since_improvement, epoch)
        writer_val.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer_val.add_scalar('c_weights', np.mean(all_weights), epoch)

        writer_train.flush()
        writer_val.flush()

        if np.mean(epoch_val_loss) < best_vloss:
            # Save checkpoint
            name = now.strftime("%m-%d-%H:%M") + '_seg'
            save_checkpoint(name, epoch, epochs_since_improvement, model, optimizer, accuracy, batch_size,
                            learning_rate, n_points, weighing_method)
            epochs_since_improvement = 0
            best_vloss = np.mean(epoch_val_loss)

        else:
            epochs_since_improvement += 1

    # if output_folder:
    #     if not os.path.isdir(output_folder):
    #         os.mkdir(output_folder)
    # plot_losses(train_loss, test_loss, save_to_file=os.path.join(output_folder, 'loss_plot.png'))
    # plot_accuracies(train_acc, test_acc, save_to_file=os.path.join(output_folder, 'accuracy_plot.png'))
    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('--path_list_files', type=str,
                        default='pointNet/data/train_test_files/RGBN')
    parser.add_argument('--output_folder', type=str, help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2048, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weighing_method', type=str, default='EFS',
                        help='sample weighing method: ISNS or INS or EFS')
    parser.add_argument('--beta', type=float, default=0.999, help='model checkpoint path')
    parser.add_argument('--number_of_workers', type=int, default=4, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')
    parser.add_argument('--c_sample', type=bool, default=False, help='use constrained sampling')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    train(args.dataset_folder,
          args.path_list_files,
          args.output_folder,
          args.number_of_points,
          args.batch_size,
          args.epochs,
          args.learning_rate,
          args.weighing_method,
          args.beta,
          args.number_of_workers,
          args.model_checkpoint,
          args.c_sample)
