import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_block(n_in, n_filters, kernel_size=(3, 3, 3), dilation_rate=1):
    return nn.Sequential(nn.Conv3d(n_in, n_filters, kernel_size, dilation=dilation_rate),
                         nn.BatchNorm3d(n_filters),
                         nn.ReLU()
                        )

class OstiaDetector(nn.Module):
    def __init__(self, n_filters=32, D=500):
        super().__init__()

        self.conv1 = conv_block(1, n_filters)
        self.conv2 = conv_block(n_filters, n_filters)
        self.conv3 = conv_block(n_filters, n_filters, dilation_rate=2)
        self.conv4 = conv_block(n_filters, n_filters, dilation_rate=4)

        # Tracker
        self.track_conv1 = conv_block(n_filters, 64)
        self.track_conv2 = conv_block(64, 64, kernel_size=(1, 1, 1))
        self.track_conv3 = nn.Conv3d(64, 1, (1, 1, 1))
        self.flatten_t = nn.Flatten()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Tracker
        x_t = self.track_conv1(x)
        x_t = self.track_conv2(x_t)
        x_t = self.track_conv3(x_t)
        x_t = self.flatten_t(x_t)

        return x_t
    
