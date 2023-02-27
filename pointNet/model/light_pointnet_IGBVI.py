import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformationNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(TransformationNet, self).__init__()
        self.output_dim = output_dim

        self.conv_1 = nn.Conv1d(input_dim, 64, 1, bias=False)
        self.conv_2 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv_3 = nn.Conv1d(128, 256, 1, bias=False)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(256)
        self.bn_4 = nn.BatchNorm1d(256)
        self.bn_5 = nn.BatchNorm1d(128)

        self.fc_1 = nn.Linear(256, 256, bias=False)
        self.fc_2 = nn.Linear(256, 128, bias=False)
        self.fc_3 = nn.Linear(128, self.output_dim * self.output_dim)

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, 256)

        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)

        identity_matrix = torch.eye(self.output_dim)
        if torch.cuda.is_available():
            identity_matrix = identity_matrix.cuda()
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x


class BasePointNet(nn.Module):

        def __init__(self, point_dimension, return_local_features=False, device='cuda'):
        super(BasePointNet, self).__init__()
        self.return_local_features = return_local_features
        self.input_transform = TransformationNet(input_dim=point_dimension, output_dim=point_dimension, device=device)
        self.feature_transform = TransformationNet(input_dim=64, output_dim=64, device=device)

        self.conv_1 = nn.Conv1d(7, 64, 1, bias=False)  # 7 channels to take I, NDVI, RGB into account
        self.conv_2 = nn.Conv1d(64, 64, 1, bias=False)
        self.conv_3 = nn.Conv1d(64, 64, 1, bias=False)
        self.conv_4 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv_5 = nn.Conv1d(128, 256, 1, bias=False)

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
        x_tnet = torch.cat([x_tnet, x[:, :, 2:]], dim=2)  # concat al features to TNet out
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
        global_feature = x.view(-1, 256)

        if self.return_local_features:
            global_feature = global_feature.view(-1, 256, 1).repeat(1, 1, num_points)
            return torch.cat([global_feature.transpose(2, 1), local_point_features], 2), feature_transform
        else:
            return global_feature, feature_transform


class ClassificationPointNet_IGBVI(nn.Module):

    def __init__(self, num_classes, dropout=0.3, point_dimension=3, dataset=''):
        super(ClassificationPointNet_IGBVI, self).__init__()
        self.dataset = dataset

        self.base_pointnet = BasePointNet(return_local_features=False, point_dimension=point_dimension,
                                          dataset=self.dataset)

        self.fc_1 = nn.Linear(256, 128, bias=False)
        self.fc_2 = nn.Linear(128, 64, bias=False)
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


class SegmentationPointNet_IGBVI(nn.Module):

    def __init__(self, num_classes, point_dimension=3):
        super(SegmentationPointNet_IGBVI, self).__init__()
        self.base_pointnet = BasePointNet(return_local_features=True, point_dimension=point_dimension)

        self.conv_1 = nn.Conv1d(320, 256, 1)  # 256+64
        self.conv_2 = nn.Conv1d(256, 128, 1)
        self.conv_3 = nn.Conv1d(128, 64, 1)
        self.conv_4 = nn.Conv1d(64, num_classes, 1)

        self.bn_1 = nn.BatchNorm1d(256)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(64)

    def forward(self, x):
        local_global_features, feature_transform = self.base_pointnet(x)
        x = local_global_features
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))

        x = self.conv_4(x)
        x = x.transpose(2, 1)

        return F.log_softmax(x, dim=-1), feature_transform
