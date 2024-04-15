"""CNN-LSTM from https://doi.org/10.3389/fncom.2022.1019776.
A composite deep learning model was created by merging a homogenous CNN and LSTM classifier with the ResNet152 model.
This implementation is based on the netowrk model architecture given in the publication:

    [1] B. Chakravarthi, S.-C. Ng, M. R. Ezilarasan, and M.-F. Leung, 
    “EEG-based emotion recognition using hybrid CNN and LSTM classification,” 
    Front Comput Neurosci, vol. 16, p. 1019776, Oct. 2022, doi: 10.3389/fncom.2022.1019776.


Authors
 * Zita Torocsik, 2024
 * Implemented from the model architecture given in publication [1]. 
"""

import torch
import speechbrain as sb

import torch
import speechbrain as sb

class CNNLSTM(torch.nn.Module):
    """CNNLSTM.

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
    #>>> model = CNNLSTM(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
    """
    def __init__(
        self,
        input_shape=None,
        dense_n_neurons=4,
        dropout=0.25,

    ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        self.default_sf = 250  # sampling rate of the original publication (Hz)

        C = input_shape[2]
        self.conv_module = torch.nn.Sequential()

        # convolutional block 1
        self.conv_module.add_module(
            "conv0",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(3,3),
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
            "pool_0",
            sb.nnet.pooling.Pooling2d(
                kernel_size=(2,2),
                pool_type="max",
            ),)

        # dropout
        self.conv_module.add_module(
            "drop_0",
            torch.nn.Dropout(p=dropout),
        )
        self.conv_module.add_module(
            "conv1",
            sb.nnet.CNN.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3,3),
                padding="valid",
                bias=False,
                swap=True,
            ),)
        # batch norm
        self.conv_module.add_module(
            "bnorm_1",
            sb.nnet.normalization.BatchNorm2d(
                input_size=32, momentum=0.1, affine=True,
            ),)
        # activation
        self.conv_module.add_module("act_1", torch.nn.ELU())
        # max pooling
        self.conv_module.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                kernel_size=(2,2),
                pool_type="max",
            ),)

        # dropout
        self.conv_module.add_module(
            "drop_1",
            torch.nn.Dropout(p=dropout),
        )

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
        # dropout and relu
        self.dense_module.add_module(
            "drop_2",
            torch.nn.Dropout(p=dropout),
        )
        self.dense_module.add_module("act_2", torch.nn.ELU())

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