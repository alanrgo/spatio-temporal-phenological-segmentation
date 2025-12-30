import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNetInitial(nn.Module):
    """Initial convnet block for each temporal input"""
    def __init__(self, crop_size, num_timestamps=13, channels=3):
        super(ConvNetInitial, self).__init__()
        self.crop_size = crop_size
        self.conv1 = nn.Conv2d(num_timestamps * channels, 64, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # x shape: [batch, crop_size, crop_size, 13*3]
        # Reshape to [batch, channels, height, width]
        x = x.permute(0, 3, 1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.pool1(x)
        
        return x


class ConvNet25Temporal(nn.Module):
    """ConvNet for temporal phenological segmentation"""
    def __init__(self, crop_size=25, num_inputs=39, num_classes=6, num_timestamps=13, channels=3):
        super(ConvNet25Temporal, self).__init__()
        self.crop_size = crop_size
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        
        # Initial convnets for each temporal input
        self.initial_convnets = nn.ModuleList([
            ConvNetInitial(crop_size, num_timestamps, channels) 
            for _ in range(num_inputs)
        ])
        
        # After pooling: (25-4+1)/2 = 11
        # Concatenated features: 64 * num_inputs channels
        self.conv2 = nn.Conv2d(64 * num_inputs, 128, kernel_size=4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After conv2 and pool2: (11-4+1)/2 = 4
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        # After conv3 and pool3: (4-3+1)/1 = 2, then pool: 1x1
        self.fc1 = nn.Linear(1 * 1 * 256, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(1024, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        # x is a list/tuple of tensors: [x[0], x[1], ..., x[num_inputs-1]]
        # Each x[i] has shape: [batch, crop_size, crop_size, 13*3]
        
        pools = []
        for i in range(self.num_inputs):
            pool = self.initial_convnets[i](x[i])
            pools.append(pool)
        
        # Concatenate along channel dimension
        pool_concat = torch.cat(pools, dim=1)  # [batch, 64*num_inputs, H, W]
        
        # Main network
        x = self.conv2(pool_concat)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x
