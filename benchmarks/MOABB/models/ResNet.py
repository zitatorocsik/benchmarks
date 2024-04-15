"""ResNet from https://doi.org/10.1155/2021/5599615.
Residual neural network (ResNet) proposed for motor execution and motor imagery decoding from single-trial EEG signals.
This implementation is based on the netowrk model architecture given in the publication:

    [1] K. H. Cheah, H. Nisar, V. V. Yap, C.-Y. Lee, and G. R. Sinha, 
    “Optimizing Residual Networks and VGG for Classification of EEG Signals: Identifying Ideal Channels for Emotion Recognition,” 
    J Healthc Eng, vol. 2021, p. 5599615, Mar. 2021, doi: 10.1155/2021/5599615.


Authors
 * Zita Torocsik, 2024
 * Implemented from the model architecture given in publication [1]. 
"""


import torch
import speechbrain as sb

class ResNet(torch.nn.Module):
    """ResNet.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
    dropout: float
        Dropout probability.
    dense_n_neurons: int
        Number of output neurons.

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 200, 32, 1])
    #>>> model = ResNet(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
    """
    def __init__(
        self,
        input_shape=None,
        dropout=0.25,
        dense_n_neurons=4,
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        self.default_sf = 250  # sampling rate of the original publication (Hz)

        C = input_shape[2]
        self.conv_module = torch.nn.Sequential()

        # conv 2d
        self.conv_module.add_module(
            "conv0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(5,1),
                padding="valid",
                bias=True,
                swap=True,
            ),)
        # batch norm
        self.conv_module.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
                input_size=32, momentum=0.1, affine=True,
            ),)
        
        # activation
        self.conv_module.add_module("act_0", torch.nn.ELU())

        # max pooling
        self.conv_module.add_module(
            "pool0",
            sb.nnet.pooling.Pooling2d(
                kernel_size=(2,1),
                pool_type="max",
            ),)
        
        # dropout
        self.conv_module.add_module(
            "drop0",
            torch.nn.Dropout(p=dropout),)

        # conv2d
        self.conv_module.add_module(
            "conv1",
            sb.nnet.CNN.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(1,C),
                padding="valid",
                bias=True,
                swap=True,
            ),)
        
        # batch norm
        self.conv_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=64, momentum=0.1, affine=True,
            ),)
        
        # activation
        self.conv_module.add_module("act_1", torch.nn.ELU())

        # max pooling

        self.conv_module.add_module(
            "pool1",
            sb.nnet.pooling.Pooling2d(
                kernel_size=(2,1),
                pool_type="max",
            ),)
        
        self.conv_module.add_module(
            "drop0",
            torch.nn.Dropout(p=dropout),)
        
        out = self.conv_module(
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )
        dense_input_size = self._num_flat_features(out)

        
        # dense block
        self.dense_module = torch.nn.Sequential()
        self.dense_module.add_module(
            "flatten", torch.nn.Flatten(),
        )
        self.dense_module.add_module(
            "dense",
            torch.nn.Linear(dense_input_size, dense_n_neurons),

        )
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1))

    def _num_flat_features(self, x):
        """
        Function was created by Davide Borra (2021) for the EEGNet model and reused here.
        Returns the number of flattened features from a tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input feature map.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        x = self.conv_module(x)
        x = self.dense_module(x)
        return x