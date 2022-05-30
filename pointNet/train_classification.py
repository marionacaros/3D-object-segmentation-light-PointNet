import argparse
import torch.optim as optim
import torch.nn.functional as F
import time
from progressbar import progressbar
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import LidarDataset
from model.light_pointnet import ClassificationPointNet
from model.light_pointnet_IGBVI import ClassificationPointNet_IGBVI
# from model.pointnet import *

import logging
import datetime
from sklearn.metrics import balanced_accuracy_score
import warnings
from utils import *
import glob
from prettytable import PrettyTable

warnings.filterwarnings('ignore')

def train(
          dataset_folder,
          number_of_points,
          batch_size,
          epochs,
          learning_rate,
          weighing_method,
          output_folder,
          number_of_workers,
          model_checkpoint,
          beta,
          sampled,
          RGBN = True):

    start_time = time.time()
    logging.info(f"Weighing method: {weighing_method}")
    logging.info(f"Smart Sample: {sampled}")

    # Tensorboard location and plot names
    now = datetime.datetime.now()
    location = 'pointNet/runs/tower_detec/' + str(number_of_points) + 'p/'

    if output_folder:
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

    # Datasets train / val / test
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

    logging.info(f'Samples with towers in train: {len(tower_files_train)}')
    logging.info(f'Dataset folder: {dataset_folder}')

    if RGBN:
        writer_train = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + '_ItrainRGBN_rdm'  + str(beta))
        writer_val = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + '_IvalRGBN_rdm'  + str(beta))
    else:
        writer_train = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + '_ItrainI'  + str(beta))
        writer_val = SummaryWriter(location + now.strftime("%m-%d-%H:%M") + '_IvalI'  + str(beta))

    logging.info(f"Tensorboard runs: {writer_train.get_logdir()}")

    if sampled:
        dir_data = 'sampled_4096'#+str(number_of_points)
    else:
        dir_data = 'data_no_ground'

    train_dataset = LidarDataset(dataset_folder=os.path.join(dataset_folder, 'pc_towers_40x40', dir_data),
                                      task='classification', number_of_points=number_of_points,
                                      towers_files = tower_files_train,
                                      landscape_files = bckg_files_train,
                                      fixed_num_points = True)
    val_dataset = LidarDataset(dataset_folder=os.path.join(dataset_folder, 'pc_towers_40x40', dir_data),
                                    task='classification', number_of_points=number_of_points,
                                    towers_files=tower_files_val,
                                    landscape_files=bckg_files_val,
                                    fixed_num_points=True)

    logging.info(f'Samples with towers in train: {train_dataset.len_towers}')
    logging.info(f'Samples without towers in train: {train_dataset.len_landscape}')
    logging.info(f'Proportion towers/landscape: {round((train_dataset.len_towers/train_dataset.len_landscape)*100,3)}%')

    logging.info(f'Samples with towers in val: {val_dataset.len_towers}')
    logging.info(f'Samples without towers in val: {val_dataset.len_landscape}')
    logging.info(f'Proportion towers/landscape: {round((val_dataset.len_towers/val_dataset.len_landscape)*100,3)}%')

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

    if RGBN:
        model = ClassificationPointNet_IGBVI(num_classes=train_dataset.NUM_CLASSIFICATION_CLASSES,
                                         point_dimension=train_dataset.POINT_DIMENSION,
                                         dataset=train_dataset)
    else:
        model = ClassificationPointNet(num_classes=train_dataset.NUM_CLASSIFICATION_CLASSES,
                                             point_dimension=train_dataset.POINT_DIMENSION,
                                             dataset=train_dataset)


    if torch.cuda.is_available():
        logging.info(f"cuda available")
        model.cuda()
        device = 'cuda'
    else:
        logging.info(f"cuda not available")
        device = 'cpu'

    # print model and parameters
    # INPUT_SHAPE = (3, 2000)
    # summary(model, INPUT_SHAPE)
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
    c_weights = None
    sample_weights = None

    #  add graph
    # point, targets, file_name = next(iter(train_dataloader))
    # point = point[0,:,:]  # [2000,12]
    # point = point.unsqueeze(0)
    # writer_train.add_graph(model, point.cuda())

    for epoch in progressbar(range(epochs), redirect_stdout=True):
        epoch_train_loss = []
        regu_train_loss = []
        not_regu_train_loss = []
        epoch_train_acc = []
        epoch_train_acc_w = []
        batch_number = 0

        if epochs_since_improvement == 10:
            adjust_learning_rate(optimizer, 0.5)
        elif epoch == 10:
            adjust_learning_rate(optimizer, 0.5)
        # --------------------------------------------- train loop ---------------------------------------------
        for data in train_dataloader:

            batch_number += 1
            points, targets, file_name = data  # [batch, n_samples, dims] [batch, n_samples]

            # get weights for imbalanced loss
            c_weights, sample_weights = \
                get_weights_transformed_for_sample(weighing_method,
                                                   n_classes=2,
                                                   samples_per_cls=[train_dataset.len_landscape + val_dataset.len_landscape, train_dataset.len_towers + val_dataset.len_towers],
                                                   beta=beta,
                                                   labels=targets)
            c_weights = c_weights.to(device)
            if torch.cuda.is_available():
                points, targets = points.cuda(), targets.cuda()

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
            accuracy_w = balanced_accuracy_score(targets.cpu(), preds.cpu(), sample_weight=sample_weights)
            epoch_train_acc_w.append(accuracy_w)

            epoch_train_acc.append(accuracy)
            # print(f'train loss: {np.mean(epoch_train_loss)}, train accuracy: {np.mean(epoch_train_acc)}')

        # --------------------------------------------- val loop ---------------------------------------------
        epoch_val_loss = []
        epoch_val_acc = []
        epoch_val_acc_w = []
        detected_positive = []
        detected_negative = []

        for batch_number, data in enumerate(val_dataloader):
            points, targets, file_name = data

            c_weights, sample_weights = \
                get_weights_transformed_for_sample(weighing_method, 2, [train_dataset.len_landscape, train_dataset.len_towers], beta=beta, labels=targets)
            c_weights = c_weights.to(device)

            if torch.cuda.is_available():
                points, targets = points.cuda(), targets.cuda()
            model = model.eval()
            preds, feature_transform = model(points)   # [batch, n_points, 2]

            loss = F.nll_loss(preds, targets, weight=c_weights)
            epoch_val_loss.append(loss.cpu().item())
            preds = preds.data.max(1)[1]
            corrects = preds.eq(targets.data).cpu().sum()

            targets_pos = (np.array(targets.cpu()) == np.ones(len(targets))).sum()
            targets_neg = (np.array(targets.cpu()) == np.zeros(len(targets))).sum()
            detected_positive.append((np.array(preds.cpu()) == np.ones(len(preds))).sum())  # boolean with positions of 1s
            detected_negative.append((np.array(preds.cpu()) == np.zeros(len(preds))).sum()) # boolean with positions of 0s

            accuracy = corrects.item() / float(batch_size)
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

        writer_train.flush()
        writer_val.flush()

        # print(f'Epoch {epoch}: train loss: {np.mean(epoch_train_loss)}, val loss: {np.mean(epoch_val_loss)},'
        #       f'train accuracy: {np.mean(epoch_train_acc)},  val accuracy: {np.mean(epoch_val_acc)}')
        print(f'Weights: {c_weights}')

        if np.mean(epoch_val_loss) < best_vloss:
            # Save checkpoint
            name = now.strftime("%m-%d-%H:%M") + str(beta)
            save_checkpoint(name, epoch, epochs_since_improvement, model, optimizer, accuracy, batch_size,
                            learning_rate, number_of_points, weighing_method)
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
    print(f'Weights: {c_weights}')
    print("--- TOTAL TIME: %s min ---" % (round((time.time() - start_time) / 60, 3)))


def save_checkpoint(name, epoch, epochs_since_improvement, model, optimizer, accuracy, batch_size, learning_rate,
                    number_of_points,weighing_method):
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


def adjust_learning_rate(optimizer, shrink_factor=0.5):
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
    parser.add_argument('dataset_folder', type=str, help='path to the dataset folder')
    parser.add_argument('--output_folder', type=str, default=None, help='output folder')
    parser.add_argument('--number_of_points', type=int, default=2048, help='number of points per cloud')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weighing_method', type=str, default='EFS', help='sample weighing method: ISNS or INS or EFS')
    parser.add_argument('--number_of_workers', type=int, default=4, help='number of workers for the dataloader')
    parser.add_argument('--model_checkpoint', type=str, default='', help='model checkpoint path')
    parser.add_argument('--beta', type=float, default=0.999, help='beta for weights')
    parser.add_argument('--sample', type=bool, default=True, help='use smart sampled data')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')
    sys.path.insert(0, '/home/m.caros/work/objectDetection/pointNet')

    train(args.dataset_folder,
          args.number_of_points,
          args.batch_size,
          args.epochs,
          args.learning_rate,
          args.weighing_method,
          args.output_folder,
          args.number_of_workers,
          args.model_checkpoint,
          args.beta,
          args.sample)

