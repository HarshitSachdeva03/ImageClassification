import torch
import torch.nn as nn

IMG_SIZE=84
# CNN Model
class CustomCNN(nn.Module):
    def __init__(self, conv_config, fc_config, num_classes, use_maxpool=True, stride=1, 
                 use_batchnorm=True, use_dropout=True, dropout_rate=0.5):  # Add new parameters
        super(CustomCNN, self).__init__()
        layers = []
        in_channels = 3
        for i, out_channels in enumerate(conv_config):
            s = stride[i] if isinstance(stride, list) else stride
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=s, padding=1))
            if use_batchnorm:  # Make BatchNorm optional
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            if use_maxpool:
                layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = out_channels
        
        self.conv = nn.Sequential(*layers)

        # Compute output size after conv layers
        conv_output_size = IMG_SIZE
        for i in range(len(conv_config)):
            s = stride[i] if isinstance(stride, list) else stride
            conv_output_size //= s
            if use_maxpool:
                conv_output_size //= 2
        flatten_dim = in_channels * conv_output_size * conv_output_size
        
        fc_layers = []
        for fc in fc_config:
            fc_layers.append(nn.Linear(flatten_dim, fc))
            fc_layers.append(nn.ReLU())
            if use_dropout:  # Make dropout optional
                fc_layers.append(nn.Dropout(p=dropout_rate))
            flatten_dim = fc
        fc_layers.append(nn.Linear(flatten_dim, num_classes))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x