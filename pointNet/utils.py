import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import plotly.graph_objects as go
import os
import sys
import torch


def collate_segmen_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here
    '''
    # get sequence lengths
    # lengths = torch.tensor([t[0].shape[0] for t in batch]).to(device)
    targets = [torch.LongTensor(t[1]) for t in batch]
    batch_data = [torch.Tensor(t[0]) for t in batch]
    # padd
    targets = torch.nn.utils.rnn.pad_sequence(targets)
    batch_data = torch.nn.utils.rnn.pad_sequence(batch_data)

    # file names
    filenames = [t[2] for t in batch]

    # compute mask
    # mask = (batch_data != 0)
    return batch_data, targets, filenames


def collate_classif_padd(batch, n_points=2048):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    # get sequence lengths
    lengths = torch.tensor([t[0].shape[0] for t in batch])
    targets = [t[1] for t in batch]
    targets = torch.LongTensor(targets)
    batch_data = [torch.Tensor(t[0]) for t in batch]

    # padd
    batch_data = torch.nn.utils.rnn.pad_sequence(batch_data,batch_first=True, padding_value=0.0)  # [max_length,B,D]
    # file names
    filenames = [t[2] for t in batch]

    # compute mask
    # mask = (batch_data != 0)
    return batch_data, targets, filenames, lengths


def blockPrint():
    # Disable
    sys.stdout = open(os.devnull, 'w')
    sys._jupyter_stdout = sys.stdout


def enablePrint():
    sys.stdout = sys.__stdout__
    sys._jupyter_stdout = sys.stdout


class hiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def transform_2d_img_to_point_cloud(img):
    img_array = np.asarray(img)
    indices = np.argwhere(img_array > 127)
    for i in range(2):
        indices[i] = (indices[i] - img_array.shape[i] / 2) / img_array.shape[i]
    return indices.astype(np.float32)


def plot_losses(train_loss, test_loss, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_loss)
    plt.plot(range(epochs), train_loss, 'bo', label='Training loss')
    plt.plot(range(epochs), test_loss, 'b', label='Test loss')
    plt.title('Training and test loss')
    plt.legend()
    if save_to_file:
        fig.savefig('figures/Loss.png', dpi=200)


def plot_accuracies(train_acc, test_acc, save_to_file=None):
    fig = plt.figure()
    epochs = len(train_acc)
    plt.plot(range(epochs), train_acc, 'bo', label='Training accuracy')
    plt.plot(range(epochs), test_acc, 'b', label='Test accuracy')
    plt.title('Training and test accuracy')
    plt.legend()
    if save_to_file:
        fig.savefig(save_to_file)


def draw_geometries(geometries, point_size=5):
    graph_objects = []

    for geometry in geometries:
        geometry_type = geometry.get_geometry_type()

        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            points = np.asarray(geometry.points)
            colors = None
            if geometry.has_colors():
                colors = np.asarray(geometry.colors)
            elif geometry.has_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
            else:
                geometry.paint_uniform_color((1.0, 0.0, 0.0))
                colors = np.asarray(geometry.colors)

            scatter_3d = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers',
                                      marker=dict(size=point_size, color=colors))
            graph_objects.append(scatter_3d)

        if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
            triangles = np.asarray(geometry.triangles)
            vertices = np.asarray(geometry.vertices)
            colors = None
            if geometry.has_triangle_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.triangle_normals) * 0.5
                colors = tuple(map(tuple, colors))
            else:
                colors = (1.0, 0.0, 0.0)

            mesh_3d = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=triangles[:, 0],
                                j=triangles[:, 1], k=triangles[:, 2], facecolor=colors, opacity=0.50)
            graph_objects.append(mesh_3d)

    fig = go.Figure(
        data=graph_objects,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )
    )
    fig.show()


def plot_3d(points, name, n_points=2000):
    points = points.view(n_points, -1).numpy()
    fig = plt.figure(figsize=[10, 10])
    ax = plt.axes(projection='3d')
    sc = ax.scatter(points[:,0], points[:, 1], points[:, 2], c=points[ :, 3], s=10, marker='o',cmap="viridis", alpha=0.5)
    plt.colorbar(sc, shrink=0.5, pad=0.05)
    directory = 'figures/results_models'
    plt.title(name + ' classes: '+str(set(points[:,3].astype('int'))))
    plt.show()
    plt.savefig(os.path.join(directory, name + '.png'), bbox_inches='tight', dpi=100)
    plt.close()


def plot_3d_subplots(points_tNet, fileName, points_i):

    fig = plt.figure(figsize=[12, 6])
    #  First subplot
    # ===============
    # set up the axes for the first plot
    # print('points_input', points_i.shape)
    # print('points_tNet', points_tNet.shape)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.title.set_text('Input data: '+ fileName)
    sc = ax.scatter(points_i[0, :], points_i[1, :], points_i[2, :], c=points_i[2, :], s=10,
                    marker='o',
                    cmap="winter", alpha=0.5)
    # fig.colorbar(sc, ax=ax, shrink=0.5)  #
    # Second subplot
    # ===============
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    sc2 = ax.scatter(points_tNet[0, :], points_tNet[1, :], points_tNet[2, :], c=points_tNet[2, :], s=10,
                    marker='o',
                    cmap="winter", alpha=0.5)
    ax.title.set_text('Output of tNet')
    plt.show()
    directory = 'figures/plots_train/'
    name = 'tNetOut_' + str(fileName) + '.png'
    plt.savefig(os.path.join(directory, name), bbox_inches='tight', dpi=150)
    plt.close()


def plot_hist(points, rdm_num):
    n_bins = 50
    # fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # # We can set the number of bins with the *bins* keyword argument.
    # axs[0].hist(points[0, 0, :], bins=n_bins)
    # axs[1].hist(points[0, 1, :], bins=n_bins)
    # axs[0].title.set_text('x')
    # axs[1].title.set_text('y')
    # plt.show()

    # 2D histogram
    fig, ax = plt.subplots(tight_layout=True)
    hist = ax.hist2d(points[0, :], points[1, :], bins=n_bins)
    fig.colorbar(hist[3], ax=ax)
    # plt.show()
    directory = 'figures'
    name = 'hist_tNet_out_' + str(rdm_num)
    plt.savefig(os.path.join(directory, name), bbox_inches='tight', dpi=100)
    plt.close()


def get_weights_effective_num_of_samples(n_of_classes, beta, samples_per_cls):
    """The authors suggest experimenting with different beta values: 0.9, 0.99, 0.999, 0.9999."""
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights4class = (1.0 - beta) / np.array(effective_num)
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights_inverse_num_of_samples(n_of_classes, samples_per_cls, power=1.0):
    weights4class = 1.0 / np.array(np.power(samples_per_cls, power))  # [0.03724195 0.00244003]
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights_sklearn(n_of_classes, samples_per_cls):
    weights4class = np.sum(samples_per_cls) / np.multiply(n_of_classes, samples_per_cls)
    weights4class = weights4class / np.sum(weights4class)
    return weights4class


def get_weights_transformed_for_sample(sample_weighing_method, n_classes, samples_per_cls, beta=None, labels=None,
                                       task='classification'):
    """
    This function applies the given Sample Weighting Scheme and returns the sample weights normalized over a batch
    Args:
        sample weighing_method: str, options available: "EFS" "INS" "ISNS"
        n_classes: int, representing the total number of classes in the entire train set
        samples_per_cls: A python list of size [n_classes]
        labels: torch tensor of size [batch] containing labels
        beta: float,
    Returns:
        weights_for_samples: torch. tensor of size [batch, n_classes]
    """

    if sample_weighing_method == 'EFS':
        weights4class = get_weights_effective_num_of_samples(n_classes, beta, samples_per_cls)
    elif sample_weighing_method == 'INS':
        weights4class = get_weights_inverse_num_of_samples(n_classes, samples_per_cls)
    elif sample_weighing_method == 'ISNS':
        weights4class = get_weights_inverse_num_of_samples(n_classes, samples_per_cls, 0.5)  # [0.9385, 0.0615]
    elif sample_weighing_method == 'sklearn':
        weights4class = get_weights_sklearn(n_classes, samples_per_cls)
    else:
        return None

    weights4class = torch.tensor(weights4class).float()
    weights4samples = None

    if task == 'classification':
        # one-hot encoding
        labels = labels.to('cpu').numpy()  # [batch] labels defines columns of non-zero elements
        one_hot = np.zeros((labels.size, 2))  # [batch, 2]
        rows = np.arange(labels.size)
        one_hot[rows, labels] = 1

        weights4samples = weights4class.unsqueeze(0)
        weights4samples = weights4samples.repeat(labels.shape[0], 1)
        weights4samples = torch.tensor(np.array(weights4samples * one_hot))
        weights4samples = weights4samples.sum(1).cpu()

    return weights4class, weights4samples

