"""DeepConvNet from https://doi.org/10.48550/arXiv.1611.08024.
Deep convolutional neural network proposed for motor execution and motor imagery decoding from single-trial EEG signals.
This implementation is based on the netowrk model architecture given in the publication:

    [1] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, 
    “EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces,” 
    J. Neural Eng., vol. 15, no. 5, p. 056013, Jul. 2018, doi: 10.1088/1741-2552/aace8c.


Authors
 * Zita Torocsik, 2024
 * Implemented from the model architecture given in publication [1]. 
"""

import torch
import speechbrain as sb

class DeepConvNet(torch.nn.Module):
    """DeepConvNet.

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
    #>>> model = DeepConvNet(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
    """
    
    def __init__(
        self,
        input_shape=None,
        dropout=0.5,
        dense_n_neurons=4,
    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        self.default_sf = 250  # sampling rate of the original publication (Hz)

        C = input_shape[2]
        self.conv_module = torch.nn.Sequential()

        # first convolution
        self.conv_module.add_module(
            "conv0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=25,
                kernel_size=(5,1),
                padding="valid",
                bias=True,
                swap=True,
            ),)
        # second convolution
        self.conv_module.add_module(
            "conv1",
            sb.nnet.CNN.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(1, C),
                padding="valid",
                bias=False,
                swap=True,
            ),)
        
        # batch norm
        self.conv_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=25, momentum=0.1, affine=True,
            ),
        )

        # activation layer
        self.conv_module.add_module("act_1", torch.nn.ELU())

        # pooling layer
        self.conv_module.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                pool_type="max",
                kernel_size=(2,1),
                pool_axis=[1, 2],
            ),
        )

        # dropout
        self.conv_module.add_module(
            "dropout_1", torch.nn.Dropout(p=dropout),
        )

        # third convolution 
        self.conv_module.add_module(
            "conv2",
            sb.nnet.CNN.Conv2d(
                in_channels=25,
                out_channels=50,
                kernel_size=(5,1),
                padding="valid",
                bias=True,
                swap=True,
            ),)
        
        # batch norm
        self.conv_module.add_module(
            "bnorm_2",
            sb.nnet.normalization.BatchNorm2d(
                input_size=50, momentum=0.1, affine=True,
            ),
        )

        # activation layer
        self.conv_module.add_module("act_2", torch.nn.ELU())

        # pooling
        self.conv_module.add_module(
            "pool_2",
            sb.nnet.pooling.Pooling2d(
                pool_type="max",
                kernel_size=(2,1),
                pool_axis=[1, 2],
            ),
        )

        # dropout
        self.conv_module.add_module(
            "dropout_2", torch.nn.Dropout(p=dropout),
        )

        # fourth convolution
        self.conv_module.add_module(
            "conv3",
            sb.nnet.CNN.Conv2d(
                in_channels=50,
                out_channels=100,
                kernel_size=(5,1),
                padding="valid",
                bias=True,
                swap=True,
            ),)
        
        # batch norm
        self.conv_module.add_module(
            "bnorm_3",
            sb.nnet.normalization.BatchNorm2d(
                input_size=100, momentum=0.1, affine=True,
            ),
        )
        # activation layer
        self.conv_module.add_module("act_3", torch.nn.ELU())

        # pooling
        self.conv_module.add_module(
            "pool_3",
            sb.nnet.pooling.Pooling2d(
                pool_type="max",
                kernel_size=(2,1),
                pool_axis=[1, 2],
            ),
        )

        # dropout
        self.conv_module.add_module(
            "dropout_3", torch.nn.Dropout(p=dropout),
        )
        # fifth convolution
        self.conv_module.add_module(
            "conv4",
            sb.nnet.CNN.Conv2d(
                in_channels=100,
                out_channels=200,
                kernel_size=(5,1),
                padding="valid",
                bias=True,
                swap=True,
            ),)
        
        # batch norm
        self.conv_module.add_module(
            "bnorm_4",
            sb.nnet.normalization.BatchNorm2d(
                input_size=200, momentum=0.1, affine=True,
            ),
        )
        # activation layer
        self.conv_module.add_module("act_4", torch.nn.ELU())
        # pooling
        self.conv_module.add_module(
            "pool_4",
            sb.nnet.pooling.Pooling2d(
                pool_type="max",
                kernel_size=(2,1),
                pool_axis=[1, 2],
            ),
        )
        # dropout
        self.conv_module.add_module(
            "dropout_4", torch.nn.Dropout(p=dropout),
        )

        out = self.conv_module(
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        )
        dense_input_size = self._num_flat_features(out)

        
        # dense module
        self.dense_module = torch.nn.Sequential()
        self.dense_module.add_module(
            "flatten", torch.nn.Flatten(),
        )
        self.dense_module.add_module(
            "dense",
            torch.nn.Linear(dense_input_size, dense_n_neurons),

        )
        # Softmax layer
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
