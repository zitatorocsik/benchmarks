import torch
import torch.nn as nn
from moabb.datasets import BNCI2014_001
from speechbrain.nnet.pooling import AvgPool2d
from speechbrain.nnet.normalization import BatchNorm2d
from speechbrain.nnet.CNN import Conv2d
from speechbrain.nnet.activations import ReLU
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.pooling import MaxPool2d


class VGG16(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_classes=2):
        super(VGG16, self).__init__()
        self.conv_block1 = nn.Sequential(
            Conv2d(
                in_channels=input_shape[0],
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            BatchNorm2d(input_size=64),
            ReLU(),
            Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            BatchNorm2d(input_size=64),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.conv_block2 = nn.Sequential(
            Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            BatchNorm2d(input_size=128),
            ReLU(),
            Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            BatchNorm2d(input_size=128),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.conv_block3 = nn.Sequential(
            Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            BatchNorm2d(input_size=256),
            ReLU(),
            Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            BatchNorm2d(input_size=256),
            ReLU(),
            Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            BatchNorm2d(input_size=256),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.conv_block4 = nn.Sequential(
            Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            BatchNorm2d(input_size=512),
            ReLU(),
            Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            BatchNorm2d(input_size=512),
            ReLU(),
            Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            BatchNorm2d(input_size=512),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.conv_block5 = nn.Sequential(
            Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            BatchNorm2d(input_size=512),
            ReLU(),
            Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            BatchNorm2d(input_size=512),
            ReLU(),
            Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            BatchNorm2d(input_size=512),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.fc_layers = nn.Sequential(
            Linear(input_size=512 * 7 * 7, n_neurons=4096),
            ReLU(),
            Linear(input_size=4096, n_neurons=4096),
            ReLU(),
            Linear(input_size=4096, n_neurons=num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


