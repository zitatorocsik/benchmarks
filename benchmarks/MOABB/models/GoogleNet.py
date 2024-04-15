"""GoogLeNet from https://doi.org/10.48550/arXiv.1409.4842.
GoogleNet a deep convolutional neural network architecture codenamed "Inception", 
which was responsible for setting the new state of the art for classification and 
detection in the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC 2014)
A carefully crafted design that allows for increasing the depth and 
width of the network while keeping the computational budget constant.

    [1] C. Szegedy et al., “Going Deeper with Convolutions.” arXiv, Sep. 16, 2014. doi: 10.48550/arXiv.1409.4842.

Authors
 * Zita Torocsik, 2024
 * Implemented from the model architecture given in publication [1] with changes to fit EEG-signal decoding purpose. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, out1x1pool):
        super(InceptionModule, self).__init__()
        # 1x1 convolution branch

        self.branch1 = sb.nnet.CNN.Conv2d(in_channels=in_channels, out_channels=out1x1, kernel_size=(1,1), padding="same")

        # 1x1 convolution followed by 3x3 convolution branch
        self.branch2 = nn.Sequential(
            sb.nnet.CNN.Conv2d(in_channels=in_channels, out_channels=red3x3, kernel_size=(1,1), padding="same"),
            sb.nnet.CNN.Conv2d(in_channels=red3x3, out_channels=out3x3, kernel_size=(3,3), padding="same")
        )

        # 1x1 convolution followed by 5x5 convolution branch
        self.branch3 = nn.Sequential(
            sb.nnet.CNN.Conv2d(in_channels=in_channels, out_channels=red5x5, kernel_size=(1,1), padding="same"),
            sb.nnet.CNN.Conv2d(in_channels=red5x5, out_channels=out5x5, kernel_size=(5,5), padding="same")
        )

        # 3x3 max pooling followed by 1x1 convolution branch
        self.branch4 = nn.Sequential(
            sb.nnet.pooling.Pooling2d(pool_type="max",kernel_size=(3,3), stride=1, padding=1),
            sb.nnet.CNN.Conv2d(in_channels=in_channels, out_channels=out1x1pool, kernel_size=(1,1), padding="same")
        )

    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)
        branch4_output = self.branch4(x)

        outputs = [branch1_output, branch2_output, branch3_output, branch4_output]
        return torch.cat(outputs, 3)  # Concatenate along the channel dimension

class GoogleNet(nn.Module):
    """GoogleNet.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
    dense_n_neurons: int
        Number of output neurons.

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 200, 32, 1])
    #>>> model = GoogleNet(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
    """
    def __init__(self, input_shape, n_classes):
        super(GoogleNet, self).__init__()
        self.input_shape = input_shape
        self.stem = nn.Sequential(
            sb.nnet.CNN.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,7), stride=2, padding="causal"),
            nn.ReLU(inplace=True),
            sb.nnet.pooling.Pooling2d(pool_type="max",kernel_size=(3,3), stride=2, padding=1)
        )

        self.inception3a = InceptionModule(64, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
 

        self.fc = nn.Linear(124, n_classes) 

        # add softmax
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        x = self.stem(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = F.avg_pool2d(x, x.size()[2:])  # Global average pooling
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


