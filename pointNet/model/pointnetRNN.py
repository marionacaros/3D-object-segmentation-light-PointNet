import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformationNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim

        self.conv_1 = nn.Conv1d(input_dim, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 256, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(256)
        self.bn_4 = nn.BatchNorm1d(256)
        self.bn_5 = nn.BatchNorm1d(128)

        self.fc_1 = nn.Linear(256, 256)
        self.fc_2 = nn.Linear(256, 128)
        self.fc_3 = nn.Linear(128, self.output_dim * self.output_dim)

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, 256)

        x = F.relu(self.bn_4(self.fc_1(x))) # peta
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)

        identity_matrix = torch.eye(self.output_dim)
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.cuda()
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x


class BasePointNet(nn.Module):

    def __init__(self, point_dimension, return_local_features=False, dataset=''):
        super(BasePointNet, self).__init__()
        self.dataset = dataset
        self.return_local_features = return_local_features
        self.input_transform = TransformationNet(input_dim=point_dimension, output_dim=point_dimension)
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64)

        # self.conv_1 = nn.Conv1d(point_dimension, 64, 1)
        self.conv_1 = nn.Conv1d(7, 64, 1)  # changed from 3 ch to 7 channels to take I, NDVI, RGB into account
        # self.conv_1 = nn.Conv1d(3, 64, 1)
        self.conv_2 = nn.Conv1d(64, 64, 1)
        self.conv_3 = nn.Conv1d(64, 64, 1)
        self.conv_4 = nn.Conv1d(64, 128, 1)
        self.conv_5 = nn.Conv1d(128, 256, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.bn_3 = nn.BatchNorm1d(64)
        self.bn_4 = nn.BatchNorm1d(128)
        self.bn_5 = nn.BatchNorm1d(256)

    def forward(self, x):
        num_points = x.shape[1]  # torch.Size([BATCH, SAMPLES, DIMS])

        x_tnet = x[:, :, :2]  # only apply T-NET to x and y
        input_transform = self.input_transform(x_tnet)
        x_tnet = torch.bmm(x_tnet, input_transform)  # Performs a batch matrix-matrix product
        x_tnet = torch.cat([x_tnet, x[:, :, 2].unsqueeze(2), x[:, :, 4].unsqueeze(2)], dim=2)  # concat z and intensity
        x_tnet = torch.cat([x_tnet, x[:, :, 8].unsqueeze(2), x[:, :, 9].unsqueeze(2), x[:, :, 11].unsqueeze(2)],
                           dim=2)  # concat Green Blue NDVI
        x_tnet = x_tnet.transpose(2, 1)  # [batch, dims, n_points]

        x = F.relu(self.bn_1(self.conv_1(x_tnet)))
        x = F.relu(self.bn_2(self.conv_2(x)))  # [batch, 64, 2000]
        x = x.transpose(2, 1)  # [batch, 2000, 64]

        feature_transform = self.feature_transform(x)  # [batch, 64, 64]

        x = torch.bmm(x, feature_transform)
        local_point_features = x  # [batch, 2000, 64]

        x = x.transpose(2, 1)
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))
        x = nn.MaxPool1d(num_points)(x)
        global_feature = x.view(-1, 256)  # [ batch, 1024, 1]

        if self.return_local_features:
            global_feature = global_feature.view(-1, 256, 1).repeat(1, 1, num_points)
            return torch.cat([global_feature.transpose(2, 1), local_point_features], 2), feature_transform
        else:
            return global_feature, feature_transform


class ClassificationPointNet(nn.Module):

    def __init__(self, num_classes, dropout=0.3, point_dimension=3, dataset=''):
        super(ClassificationPointNet, self).__init__()
        self.dataset = dataset

        self.base_pointnet = BasePointNet(return_local_features=False, point_dimension=point_dimension,
                                          dataset=self.dataset)

        self.fc_1 = nn.Linear(256, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, num_classes)

        self.bn_1 = nn.BatchNorm1d(128)
        self.bn_2 = nn.BatchNorm1d(64)

        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        global_feature, feature_transform = self.base_pointnet(x)

        x = F.relu(self.bn_1(self.fc_1(global_feature)))
        x = F.relu(self.bn_2(self.fc_2(x)))
        x = self.dropout_1(x)

        return F.log_softmax(self.fc_3(x), dim=1), feature_transform


class RNNSegmentationPointNet(nn.Module):

    def __init__(self, num_classes, hidden_size, point_dimension=3):
        super(RNNSegmentationPointNet, self).__init__()
        self.hidden_size = hidden_size
        self.base_pointnet = BasePointNet(return_local_features=True, point_dimension=point_dimension)

        self.conv_1 = nn.Conv1d(576, 256, 1)
        self.conv_2 = nn.Conv1d(256, 128, 1)
        self.conv_3 = nn.Conv1d(128, 64, 1)

        self.conv_4 = nn.Conv1d(64, num_classes, 1)

        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(64)

        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, hidden):
        local_global_features, feature_transform = self.base_pointnet(x)  # [batch, n_point, 1088]
        local_features = local_global_features[:, :, 256:]  # [batch, n_point, 64]
        num_points = local_features.shape[1]
        global_feature = local_global_features[:, 0, :256]  # [batch, 1024]
        global_feature = global_feature.view(-1, 1, 256)

        global_feat_rnn, hidden = self.gru(global_feature, hidden)

        global_feat_rnn = global_feat_rnn.view(-1, 512, 1).repeat(1, 1, num_points)  # [batch, 1024, n_points]
        local_global_rnn = torch.cat([global_feat_rnn.transpose(2, 1), local_features], 2)

        # local_global_rnn  # [batch, n_points, 1088]
        x = local_global_rnn.transpose(2, 1)  # [batch, 1088, n_points]
        x = F.relu(self.bn_1(self.conv_1(x)))  # [batch, 512, n_points]
        x = F.relu(self.bn_2(self.conv_2(x)))  # [batch, 256, n_points]
        x = F.relu(self.bn_3(self.conv_3(x)))  # [batch, 128, n_points]

        x = self.conv_4(x)
        x = x.transpose(2, 1)
        out = self.softmax(x)

        return out, hidden, feature_transform

    def initHidden(self, x):
        """tensor of shape (D * {num_layers}, N, H_out) containing the initial hidden state for the input sequence.
        Defaults to zeros if not provided """
        return torch.zeros(2, x.shape[0], self.hidden_size, device='cuda')  # [layers, x.size[0], hidden]


class SegmentationPointNet(nn.Module):

    def __init__(self, num_classes, point_dimension=3):
        super(SegmentationPointNet, self).__init__()
        self.base_pointnet = BasePointNet(return_local_features=True, point_dimension=point_dimension)

        self.conv_1 = nn.Conv1d(1088, 512, 1)
        self.conv_2 = nn.Conv1d(512, 256, 1)
        self.conv_3 = nn.Conv1d(256, 128, 1)
        self.conv_4 = nn.Conv1d(128, num_classes, 1)

        self.bn_1 = nn.BatchNorm1d(512)
        self.bn_2 = nn.BatchNorm1d(256)
        self.bn_3 = nn.BatchNorm1d(128)

    def forward(self, x):
        local_global_features, feature_transform = self.base_pointnet(x)
        x = local_global_features
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = self.conv_4(x)
        x = x.transpose(2, 1)

        return F.log_softmax(x, dim=-1), feature_transform, local_global_features
