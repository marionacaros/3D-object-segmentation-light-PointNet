from utils.utils import *
from utils.get_metrics import *
import os
import argparse
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.utils.tensorboard import SummaryWriter
from pointNet.datasets import LidarDataset
from pointNet.model.light_pointnet_IGBVI import ClassificationPointNet
import logging
import datetime
from sklearn.metrics import balanced_accuracy_score
import warnings
from prettytable import PrettyTable

logging.basicConfig(format='%(message)s', #[%(asctime)s %(levelname)s]
                    level=logging.INFO,
                    datefmt='%d-%m %H:%M:%S')

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
        c_sample):

    start_time = time.time()
    logging.info(f"Weighing method: {weighing_method}")
    logging.info(f"Constrained sampling: {c_sample}")

    # Tensorboard location and plot names
    now = datetime.datetime.now()
    location = 'pointNet/runs/tower_detec/prod_files/'

    # Datasets train / val / test
    with open(os.path.join(path_list_files, 'train_cls_files.txt'), 'r') as f:
        train_files = f.read().splitlines()
    with open(os.path.join(path_list_files, 'val_cls_files.txt'), 'r') as f:
        val_files = f.read().splitlines()

    logging.info(f'Dataset folder: {dataset_folder}')

    if 'RGBN' in path_list_files:
        writer_train = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'cls_train')
        writer_val = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'cls_val')
    else:
        writer_train = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + '_trainI')
        writer_val = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + '_valI')

    logging.info(f"Tensorboard runs: {writer_train.get_logdir()}")

    # Initialize datasets
    train_dataset = LidarDataset(dataset_folder=dataset_folder,
                                 task='classification', number_of_points=n_points,
                                 files=train_files,
                                 fixed_num_points=True,
                                 c_sample=c_sample)
    val_dataset = LidarDataset(dataset_folder=dataset_folder,
                               task='classification', number_of_points=n_points,
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

    if 'RGBN' in path_list_files:
        model = ClassificationPointNet(num_classes=train_dataset.NUM_CLASSIFICATION_CLASSES,
                                       point_dimension=train_dataset.POINT_DIMENSION,
                                       dataset=train_dataset,
                                       device=device)
    else:
        model = ClassificationPointNet(num_classes=train_dataset.NUM_CLASSIFICATION_CLASSES,
                                       point_dimension=train_dataset.POINT_DIMENSION,
                                       dataset=train_dataset)
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
    c_weights = get_weights4class(weighing_method,
                                  n_classes=2,
                                  samples_per_cls=[train_dataset.len_landscape + val_dataset.len_landscape,
                                                   train_dataset.len_towers + val_dataset.len_towers],
                                  beta=beta).to(device)
    logging.info(f'Weights: {c_weights}')

    for epoch in progressbar(range(epochs), redirect_stdout=True):
        epoch_train_loss = []
        regu_train_loss = []
        not_regu_train_loss = []
        epoch_train_acc = []
        epoch_train_acc_w = []
        epoch_val_loss = []
        epoch_val_acc = []
        epoch_val_acc_w = []
        detected_positive = []
        detected_negative = []
        targets_pos = []
        targets_neg = []

        if epochs_since_improvement == 10 or epoch == 15:
            adjust_learning_rate(optimizer, 0.5)

        # --------------------------------------------- train loop ---------------------------------------------
        for data in train_dataloader:
            points, targets, file_name = data  # [batch, n_samples, dims] [batch, n_samples]

            points = points.data.numpy()
            points[:, :, :3] = rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)

            points, targets = points.to(device), targets.to(device)

            optimizer.zero_grad()
            model = model.train()
            preds, feature_transform = model(points)

            identity = torch.eye(feature_transform.shape[-1])
            identity = identity.to(device)
            regularization_loss = torch.norm(identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1)))
            loss = F.nll_loss(preds, targets, weight=c_weights)
            not_regu_train_loss.append(loss.cpu().item())
            loss = loss + 0.001 * regularization_loss
            epoch_train_loss.append(loss.cpu().item())
            regu_train_loss.append(regularization_loss.cpu().item())

            loss.backward()
            optimizer.step()
            preds = preds.data.max(1)[1]
            corrects = preds.eq(targets.data).cpu().sum()

            # accuracy for tensorboard
            accuracy = corrects.item() / float(batch_size)
            sample_weights = get_weights4sample(c_weights.cpu(), labels=targets).numpy()
            accuracy_w = balanced_accuracy_score(targets.cpu(), preds.cpu(), sample_weight=sample_weights)
            epoch_train_acc_w.append(accuracy_w)

            epoch_train_acc.append(accuracy)
            # print(f'train loss: {np.mean(epoch_train_loss)}, train accuracy: {np.mean(epoch_train_acc)}')

        # --------------------------------------------- val loop ---------------------------------------------

        for data in val_dataloader:
            points, targets, file_name = data
            points, targets = points.to(device), targets.to(device)

            model = model.eval()
            preds, feature_transform = model(points)  # [batch, n_points, 2]

            loss = F.nll_loss(preds, targets, weight=c_weights)
            epoch_val_loss.append(loss.cpu().item())
            preds = preds.data.max(1)[1]
            corrects = preds.eq(targets.data).cpu().sum()

            # tensorboard
            targets_pos.append((np.array(targets.cpu()) == np.ones(len(targets))).sum())
            targets_neg.append((np.array(targets.cpu()) == np.zeros(len(targets))).sum())
            detected_positive.append((np.array(preds.cpu()) == np.ones(len(preds))).sum())  # bool with pos of 1s
            detected_negative.append((np.array(preds.cpu()) == np.zeros(len(preds))).sum())  # bool with pos of 0s

            accuracy = corrects.item() / float(batch_size)
            sample_weights = get_weights4sample(c_weights.cpu(), labels=targets).numpy()
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
        writer_train.add_scalar('loss_regularization', np.mean(regu_train_loss), epoch)
        writer_train.add_scalar('accuracy_weighted', np.mean(epoch_train_acc_w), epoch)
        writer_val.add_scalar('accuracy_weighted', np.mean(epoch_val_acc_w), epoch)
        writer_train.add_scalar('accuracy', np.mean(epoch_train_acc), epoch)
        writer_val.add_scalar('accuracy', np.mean(epoch_val_acc), epoch)
        writer_val.add_scalar('epochs_since_improvement', epochs_since_improvement, epoch)
        writer_val.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer_val.add_scalar('c_weights', c_weights[1].cpu(), epoch)

        writer_train.flush()
        writer_val.flush()

        if np.mean(epoch_val_loss) < best_vloss:
            # Save checkpoint
            name = now.strftime("%m-%d-%H:%M") + str(beta)
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
    parser.add_argument('--dataset_folder', type=str, help='path to the dataset folder',
                        default='/dades/LIDAR/towers_detection/datasets/pc_towers_40x40_10p/normalized_2048')
    parser.add_argument('--path_list_files', type=str,
                        default='train_test_files/RGBN_x10_40x40')
    parser.add_argument('--output_folder', type=str, help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2048, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=70, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weighing_method', type=str, default='EFS',
                        help='sample weighing method: ISNS or INS or EFS')
    parser.add_argument('--beta', type=float, default=0.999, help='model checkpoint path')
    parser.add_argument('--number_of_workers', type=int, default=8, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')
    parser.add_argument('--c_sample', type=bool, default=True, help='use constrained sampling')

    args = parser.parse_args()

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
