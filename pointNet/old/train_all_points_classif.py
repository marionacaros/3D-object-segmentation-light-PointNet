import argparse
import torch.optim as optim
import torch.nn.functional as F
import time
from progressbar import progressbar
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import LidarDataset
from model.light_pointnet_IGBVI import ClassificationPointNet_IGBVI, SegmentationPointNet_IGBVI
import logging
import datetime
from sklearn.metrics import balanced_accuracy_score
import warnings
from utils import *
import glob
from prettytable import PrettyTable

warnings.filterwarnings('ignore')

MODELS = {
    'classification': ClassificationPointNet_IGBVI,
    'segmentation': SegmentationPointNet_IGBVI
}

DATASETS = {
    'lidar': LidarDataset
}

if torch.cuda.is_available():
    logging.info(f"cuda available")
    device = 'cuda'
else:
    logging.info(f"cuda not available")
    device = 'cpu'


def train(dataset,
          dataset_folder,
          task,
          n_points,
          batch_size,
          epochs,
          learning_rate,
          weighing_method,
          output_folder,
          number_of_workers,
          model_checkpoint,
          use_rnn):
    start_time = time.time()
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Weighing method: {weighing_method}")
    BETA = 0.999

    # Tensorboard location and plot names
    now = datetime.datetime.now()
    location = 'pointNet/runs/tower_detec/' + str(n_points) + 'p/'
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    RGBN = True
    if RGBN:
        with open('../data/RGBN/RGBN_train_moved_towers_files.txt', 'r') as f:
            tower_files_train = f.read().splitlines()
        with open('../data/RGBN/RGBN_val_moved_towers_files.txt', 'r') as f:
            tower_files_val = f.read().splitlines()
        with open('../data/RGBN/RGBN_train_landscape_files.txt', 'r') as f:
            bckg_files_train = f.read().splitlines()
        with open('../data/RGBN/RGBN_val_landscape_files.txt', 'r') as f:
            bckg_files_val = f.read().splitlines()
    else:
        with open('../data/val_moved_towers_files.txt', 'r') as f:
            tower_files_val = f.read().splitlines()
        with open('../data/train_moved_towers_files.txt', 'r') as f:
            tower_files_train = f.read().splitlines()
        with open('../data/train_landscape_files.txt', 'r') as f:
            bckg_files_train = f.read().splitlines()
        with open('../data/val_landscape_files.txt', 'r') as f:
            bckg_files_val = f.read().splitlines()


    # Datasets train / val / test
    if task == 'classification':
        custom_collate_fn = collate_classif_padd

        writer_train = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + '_IGBVI_train'  + str(BETA))
        writer_val = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + '_IGBVI_val' + str(BETA))
        logging.info(f"Tensorboard runs: {writer_train.get_logdir()}")

        train_dataset = DATASETS[dataset](dataset_folder=os.path.join(dataset_folder, 'pc_towers_40x40', 'data_no_ground'),
                                          task=task, number_of_points=n_points,
                                          towers_files=tower_files_train,
                                          landscape_files=bckg_files_train,
                                          fixed_num_points=True)
        val_dataset = DATASETS[dataset](dataset_folder=os.path.join(dataset_folder, 'pc_towers_40x40', 'data_no_ground'),
                                        task=task, number_of_points=n_points,
                                        towers_files=tower_files_val,
                                        landscape_files=bckg_files_val,
                                        fixed_num_points=True)

        logging.info(f'Samples with towers in train: {train_dataset.len_towers}')
        logging.info(f'Samples without towers in train: {train_dataset.len_landscape}')
        logging.info(
            f'Proportion towers/landscape: {round((train_dataset.len_towers / train_dataset.len_landscape) * 100, 3)}%')
        logging.info(f'Samples with towers in val: {val_dataset.len_towers}')
        logging.info(f'Samples without towers in val: {val_dataset.len_landscape}')
        logging.info(
            f'Proportion towers/landscape: {round((val_dataset.len_towers / val_dataset.len_landscape) * 100, 3)}%')


    logging.info(f'Samples for training: {len(train_dataset)}')
    logging.info(f'Samples for validation: {len(val_dataset)}')
    logging.info(f'Task: {train_dataset.task}')


    # Datalaoders
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=number_of_workers,
                                                   drop_last=True,
                                                   collate_fn=custom_collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=number_of_workers,
                                                 drop_last=True,
                                                 collate_fn=custom_collate_fn)

    if task == 'segmentation':
        if use_rnn:
            print()
            # model = RNNSegmentationPointNet_I(num_classes=train_dataset.NUM_SEGMENTATION_CLASSES,
            #                                 hidden_size=256,
            #                                 point_dimension=train_dataset.POINT_DIMENSION)
        else:
            model = SegmentationPointNet_IGBVI(num_classes=train_dataset.NUM_SEGMENTATION_CLASSES,
                                            point_dimension=train_dataset.POINT_DIMENSION)
    elif task == 'classification':
        model = ClassificationPointNet_IGBVI(num_classes=train_dataset.NUM_CLASSIFICATION_CLASSES,
                                       point_dimension=train_dataset.POINT_DIMENSION,
                                       dataset=train_dataset)
    else:
        raise Exception('Unknown task !')

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

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    best_vloss = 1_000_000.
    epochs_since_improvement = 0

    # training loop
    for epoch in progressbar(range(epochs), redirect_stdout=True):
        epoch_train_loss = []
        regu_train_loss = []
        epoch_train_acc = []
        epoch_train_acc_w = []
        seen_points = 0

        if epochs_since_improvement == 10:
            adjust_learning_rate(optimizer, 0.5)
        elif epoch == 10:
            adjust_learning_rate(optimizer, 0.5)

        # --------------------------------------------- train loop ---------------------------------------------
        for data in train_dataloader:
            points, targets, filenames = data  # [7557, batch, dims], [4]

            points = points.view(batch_size, -1, 10)  # [batch, n_samples, dims]
            targets = targets.view(batch_size, -1).to(device)  # [batch, n_samples]
            pc_pred = torch.Tensor().to(device)

            # Pytorch accumulates gradients. We need to clear them out before each instance
            optimizer.zero_grad()
            model = model.train()

            if use_rnn:
                hidden = model.initHidden(points)

            j = 0
            while seen_points < points.shape[1]:
                end_batch = n_points * (j + 1)
                if end_batch < points.shape[1]:
                    in_points = points[:, j * n_points: end_batch, :]
                else:
                    points_needed = end_batch - points.shape[1]
                    if points_needed == 2048 and points.shape[1] == 2048:
                        in_points = points
                        extra_points = points
                        extra_targets = targets
                    else:
                        rdm_list = np.random.randint(0, points.shape[1], points_needed)
                        in_points = points[:, j * n_points:, :]
                        extra_points = points[:, rdm_list, :]
                        extra_targets = targets[:, rdm_list]

                    targets = torch.cat((targets, extra_targets), dim=1)
                    in_points = torch.cat([in_points, extra_points], dim=1)

                # forward pass
                if use_rnn:
                    preds, hidden, feature_transform = model(in_points.to(device), hidden)  # [b, n_points, 2] [2, b, 128] [b,64,64]
                else:
                    preds, feature_transform = model(in_points.to(device))  # classification: [batch, 2], [batch, 64, 64]
                    preds = preds.unsqueeze(2)  # classification: [batch, 2, 1]
                    pc_pred = preds.view(-1, 2)
                targets = targets.view(-1)

                c_weights, sample_weights = \
                    get_weights_transformed_for_sample(weighing_method,
                                                       n_classes=2,
                                                       samples_per_cls=[train_dataset.len_landscape + val_dataset.len_landscape, train_dataset.len_towers + val_dataset.len_towers],
                                                       beta=BETA,
                                                       labels=targets)
                c_weights = c_weights.to(device)

                identity = torch.eye(feature_transform.shape[-1])
                identity = identity.to(device)
                regularization_loss = torch.norm(
                    identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1)))
                regu_train_loss.append(regularization_loss.cpu().item())

                loss = F.nll_loss(pc_pred, targets, weight=c_weights) + 0.001 * regularization_loss
                epoch_train_loss.append(loss.cpu().item())

                loss.backward()
                optimizer.step()

                pc_pred = pc_pred.data.max(1)[1]
                corrects = pc_pred.eq(targets.data).cpu().sum()
                if task == 'classification':
                    accuracy = corrects.item() / float(batch_size)
                    accuracy_w = balanced_accuracy_score(targets.cpu(), pc_pred.cpu(), sample_weight=sample_weights)
                    epoch_train_acc_w.append(accuracy_w)

                elif task == 'segmentation':
                    accuracy = corrects.item() / float(targets.shape[0])
                epoch_train_acc.append(accuracy)

                seen_points += in_points.shape[1]
                j += 1

            seen_points = 0

        # --------------------------------------------- val loop ---------------------------------------------
        epoch_val_loss = []
        epoch_val_acc = []
        epoch_val_acc_w = []
        detected_positive = []
        detected_negative = []
        targets_pos = []
        targets_neg = []

        with torch.no_grad():
            for data in val_dataloader:
                points, targets, filenames = data  # [7557, 4, 12]
                points = points.view(batch_size, -1, 10)  # [batch, n_samples, dims]
                targets = targets.view(batch_size, -1)  # [batch, n_samples]

                targets = targets.to(device)

                optimizer.zero_grad()
                model = model.eval()

                if use_rnn:
                    hidden = model.initHidden(points)
                j = 0
                seen_points = 0
                while seen_points < points.shape[1]:
                    end_batch = n_points * (j + 1)
                    if end_batch < points.shape[1]:
                        in_points = points[:, j * n_points: end_batch, :]
                    else:
                        points_needed = end_batch - points.shape[1]
                        if points_needed == 2048 and points.shape[1] == 2048:
                            in_points = points
                            extra_points = points
                            extra_targets = targets
                        else:
                            rdm_list = np.random.randint(0, points.shape[1], points_needed)
                            in_points = points[:, j * n_points:, :]
                            extra_points = points[:, rdm_list, :]
                            extra_targets = targets[:, rdm_list]

                        targets = torch.cat((targets, extra_targets), dim=1)
                        in_points = torch.cat([in_points, extra_points], dim=1)

                    if use_rnn:
                        preds, hidden, feature_transform = model(in_points.to(device), hidden)  # [batch, n_points, 2] [2, batch, 128]
                    else:
                        preds, feature_transform = model(in_points.to(device))

                    preds = preds.unsqueeze(2)
                    pc_pred = preds.view(-1, 2)
                    targets = targets.view(-1)

                    c_weights, sample_weights = \
                        get_weights_transformed_for_sample(weighing_method,
                                                           n_classes=2,
                                                           samples_per_cls=[train_dataset.len_landscape + val_dataset.len_landscape, train_dataset.len_towers + val_dataset.len_towers],
                                                           beta=BETA,
                                                           labels=targets)
                    c_weights = c_weights.to(device)

                    loss = F.nll_loss(pc_pred, targets, weight=c_weights)
                    epoch_val_loss.append(loss.cpu().item())
                    pc_pred = pc_pred.data.max(1)[1]
                    corrects = pc_pred.eq(targets.data).cpu().sum()

                    # sum of targets in batch
                    targets_pos.append((np.array(targets.cpu()) == np.ones(len(targets))).sum())
                    targets_neg.append((np.array(targets.cpu()) == np.zeros(len(targets))).sum())
                    detected_positive.append(
                        (np.array(pc_pred.cpu()) == np.ones(len(pc_pred))).sum())  # bool with positions of 1s
                    detected_negative.append(
                        (np.array(pc_pred.cpu()) == np.zeros(len(pc_pred))).sum())  # bool with positions of 0s

                    if task == 'classification':
                        accuracy = corrects.item() / float(batch_size)
                        accuracy_w = balanced_accuracy_score(targets.cpu(), pc_pred.cpu(), sample_weight=sample_weights)
                        epoch_val_acc_w.append(accuracy_w)

                    elif task == 'segmentation':
                        accuracy = corrects.item() / float(targets.shape[0])

                    epoch_val_acc.append(accuracy)

                    seen_points += in_points.shape[1]
                    j += 1



        # ------------------------------------------------------------------------------------------------------
        # Tensorboard
        writer_train.add_scalar('loss', np.mean(epoch_train_loss), epoch)
        writer_val.add_scalar('loss', np.mean(epoch_val_loss), epoch)
        writer_train.add_scalar('mean_detected_positive', np.mean(targets_pos), epoch)
        writer_val.add_scalar('mean_detected_positive', np.mean(detected_positive), epoch)
        writer_train.add_scalar('mean_detected_negative', np.mean(targets_neg), epoch)
        writer_val.add_scalar('mean_detected_negative', np.mean(detected_negative), epoch)
        writer_train.add_scalar('regularization_loss', np.mean(regu_train_loss), epoch)
        writer_train.add_scalar('accuracy_weighted', np.mean(epoch_train_acc_w), epoch)
        writer_val.add_scalar('accuracy_weighted', np.mean(epoch_val_acc_w), epoch)
        writer_train.add_scalar('accuracy', np.mean(epoch_train_acc), epoch)
        writer_val.add_scalar('accuracy', np.mean(epoch_val_acc), epoch)
        writer_val.add_scalar('epochs_since_improvement', epochs_since_improvement, epoch)
        writer_val.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        writer_train.flush()
        writer_val.flush()

        # print(f'Epoch {epoch}: train loss: {np.round(np.mean(epoch_train_loss),5)}, '
        #       f'val loss: {np.round(np.mean(epoch_val_loss),5)},'
        #       f'train accuracy: {np.round(np.mean(epoch_train_acc),5)},  '
        #       f'val accuracy: {np.round(np.mean(epoch_val_acc),5)}')
        # print(f'Weights: {c_weights}')

        if np.mean(epoch_val_loss) < best_vloss:
            # Save checkpoint
            if task == 'classification':
                name = now.strftime("%m-%d-%H:%M") + weighing_method + str(BETA)
            elif task == 'segmentation':
                name = now.strftime("%m-%d-%H:%M") + '_seg'
            save_checkpoint(name, epoch, epochs_since_improvement, model, optimizer, accuracy, batch_size,
                            learning_rate, n_points, weighing_method)
            epochs_since_improvement = 0
            best_vloss = np.mean(epoch_val_loss)

        else:
            epochs_since_improvement += 1

        # with open(os.path.join(output_folder, 'training_log.csv'), 'a') as fid:
        #     fid.write('%s,%s,%s,%s,%s\n' % (epoch,
        #                                     np.mean(epoch_train_loss),
        #                                     np.mean(epoch_val_loss),
        #                                     np.mean(epoch_train_acc_w),
        #                                     np.mean(epoch_val_acc_w)))
        train_loss.append(np.mean(epoch_train_loss))
        test_loss.append(np.mean(epoch_val_loss))
        train_acc.append(np.mean(epoch_train_acc_w))
        test_acc.append(np.mean(epoch_val_acc_w))

    # plot_losses(train_loss, test_loss, save_to_file=os.path.join(output_folder, 'loss_plot.png'))
    # plot_accuracies(train_acc, test_acc, save_to_file=os.path.join(output_folder, 'accuracy_plot.png'))
    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))


def save_checkpoint(name, epoch, epochs_since_improvement, model, optimizer, accuracy, batch_size, learning_rate,
                    number_of_points, weighing_method):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'batch_size': batch_size,
        'lr': learning_rate,
        'number_of_points': number_of_points,
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'accuracy': accuracy,
        'weighing_method': weighing_method
    }
    filename = 'checkpoint_' + name + '.pth'

    torch.save(state, 'pointNet/checkpoints/' + filename)


def adjust_learning_rate(optimizer, shrink_factor=0.1):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='lidar', type=str, choices=['lidar', 'shapenet', 'mnist'],
                        help='dataset to train on')
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('task', type=str, choices=['classification', 'segmentation'], help='type of task')
    parser.add_argument('output_folder', type=str, help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2000, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weighing_method', type=str, default='ISNS',
                        help='sample weighing method: ISNS or INS or EFS')
    parser.add_argument('--number_of_workers', type=int, default=1, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')
    parser.add_argument('--use_rnn', type=bool, default=False, help='True if want to use RNN model')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    sys.path.insert(0, '/home/m.caros/work/objectDetection')

    train(args.dataset,
          args.dataset_folder,
          args.task,
          args.number_of_points,
          args.batch_size,
          args.epochs,
          args.learning_rate,
          args.weighing_method,
          args.output_folder,
          args.number_of_workers,
          args.model_checkpoint,
          args.use_rnn)

# python pointNet/train_all_points_classif.py lidar /dades/LIDAR/towers_detection/datasets classification pointNet/results/ --batch_size 32 --epochs 100 --learning_rate 0.001 --weighing_method EFS --number_of_points 4096 --number_of_workers 4