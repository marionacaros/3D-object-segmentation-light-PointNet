import argparse

import torch.optim as optim
import torch.nn.functional as F
import time
from progressbar import progressbar
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import LidarDataset
from model.light_pointnet_IGBVI import SegmentationPointNet_IGBVI
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
          n_points,
          batch_size,
          epochs,
          learning_rate,
          weighing_method,
          output_folder,
          number_of_workers,
          model_checkpoint,
          use_rnn,
          beta,
          sample):

    task='segmentation'
    start_time = time.time()
    logging.info(f"Weighing method: {weighing_method}")
    logging.info(f"Use RNN: {use_rnn}")

    # Tensorboard location and plot names
    now = datetime.datetime.now()
    location = 'pointNet/runs/tower_detec/' + str(n_points) + 'p/'

    if output_folder:
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

    RGBN = True
    if RGBN:
        with open('pointNet/data/RGBN/RGBN_train_moved_towers_files.txt', 'r') as f:
            tower_files_train = f.read().splitlines()
        with open('pointNet/data/RGBN/RGBN_val_moved_towers_files.txt', 'r') as f:
            tower_files_val = f.read().splitlines()
        with open('pointNet/data/RGBN/RGBN_train_landscape_files.txt', 'r') as f:
            bckg_files_train = f.read().splitlines()
        with open('pointNet/data/RGBN/RGBN_val_landscape_files.txt', 'r') as f:
            bckg_files_val = f.read().splitlines()
    else:
        with open('pointNet/data/val_moved_towers_files.txt', 'r') as f:
            tower_files_val = f.read().splitlines()
        with open('pointNet/data/train_moved_towers_files.txt', 'r') as f:
            tower_files_train = f.read().splitlines()
        with open('pointNet/data/train_landscape_files.txt', 'r') as f:
            bckg_files_train = f.read().splitlines()
        with open('pointNet/data/val_landscape_files.txt', 'r') as f:
            bckg_files_val = f.read().splitlines()

    custom_collate_fn = collate_segmen_padd

    writer_train = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'seg_smart_train' )
    writer_val = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + 'seg_smart_val')

    logging.info(f"Tensorboard runs: {writer_train.get_logdir()}")

    landscape_files_train = bckg_files_train[:int(len(bckg_files_train) * 0.05)]
    landscape_files_val = bckg_files_val[:int(len(bckg_files_val) * 0.05)]
    logging.info(f'Samples with landscape for segmentation: {len(landscape_files_train)+len(landscape_files_val)}')

    train_dataset = LidarDataset(dataset_folder=os.path.join(dataset_folder, 'pc_towers_40x40', 'sampled_'+str(n_points)),
                                      task=task, number_of_points=n_points,
                                      towers_files=tower_files_train,
                                      landscape_files=landscape_files_train,
                                      fixed_num_points=True)
    val_dataset = LidarDataset(dataset_folder=os.path.join(dataset_folder, 'pc_towers_40x40', 'sampled_'+str(n_points)),
                                    task=task, number_of_points=n_points,
                                    towers_files=tower_files_val,
                                    landscape_files=landscape_files_val,
                                    fixed_num_points=True)

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

    if use_rnn:
        print()
        # model = RNNSegmentationPointNet_I(num_classes=train_dataset.NUM_SEGMENTATION_CLASSES,
        #                                 hidden_size=256,
        #                                 point_dimension=train_dataset.POINT_DIMENSION)
    else:
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

    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    best_vloss = 1_000_000.
    epochs_since_improvement = 0

    for epoch in progressbar(range(epochs), redirect_stdout=True):
        epoch_train_loss = []
        regu_train_loss = []
        epoch_train_acc = []
        epoch_train_acc_w = []
        all_weights=[]

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
                points_tower=100
                points_landscape=4000

            c_weights, sample_weights = get_weights_transformed_for_sample(weighing_method,
                                                                           n_classes=2,
                                                                           samples_per_cls=[points_landscape, points_tower],
                                                                           beta=beta,
                                                                           labels=targets)
            c_weights = c_weights.to(device)

            identity = torch.eye(feature_transform.shape[-1])
            identity = identity.to(device)
            regularization_loss = torch.norm(identity - torch.bmm(feature_transform, feature_transform.transpose(2, 1)))
            regu_train_loss.append(regularization_loss.cpu().item())

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
                c_weights, sample_weights = \
                    get_weights_transformed_for_sample(weighing_method,
                                                       n_classes=2,
                                                       samples_per_cls=[points_landscape, points_tower],
                                                       beta=beta,
                                                       labels=targets)
                c_weights = c_weights.to(device)

                loss = F.nll_loss(preds, targets,weight=c_weights)
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
        writer_train.add_scalar('regularization_loss', np.mean(regu_train_loss), epoch)
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
            if task == 'classification':
                name = now.strftime("%m-%d-%H:%M") + weighing_method + str(beta)
            elif task == 'segmentation':
                name = now.strftime("%m-%d-%H:%M") + '_seg_smart'
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
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    logging.info(f"Decaying learning rate to {optimizer.param_groups[0]['lr']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('--number_of_points', type=int, default=2000, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weighing_method', type=str, default='ISNS',
                        help='sample weighing method: ISNS or INS or EFS')
    parser.add_argument('--number_of_workers', type=int, default=1, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')
    parser.add_argument('--use_rnn', type=bool, default=False, help='True if want to use RNN model')
    parser.add_argument('--beta', type=float, default=0.999, help='beta for weights')
    parser.add_argument('--sample', type=bool, default=False, help='use smart sampled data')
    parser.add_argument('--output_folder', type=str, default=None, help='output folder')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    sys.path.insert(0, '/home/m.caros/work/objectDetection')

    train(
          args.dataset_folder,
          args.number_of_points,
          args.batch_size,
          args.epochs,
          args.learning_rate,
          args.weighing_method,
          args.output_folder,
          args.number_of_workers,
          args.model_checkpoint,
          args.use_rnn,
          args.beta,
          args.sample)

# python pointNet/train_segmentation.py /dades/LIDAR/towers_detection/datasets  --batch_size 32 --epochs 100 --learning_rate 0.001 --weighing_method EFS --number_of_points 4096 --number_of_workers 4