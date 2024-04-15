"""VGG16 from https://doi.org/10.1155/2021/5599615.
VGG16 network architecture proposed for motor execution and motor imagery decoding from single-trial EEG signals.
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

class VGG16(torch.nn.Module):
    """VGG16.

    Arguments
    ---------
    input_shape: tuple
        The shape of the input.
    dense_n_neurons: int
        Number of output neurons.
    dropout: float
        Dropout probability.

    Example
    -------
    #>>> inp_tensor = torch.rand([1, 200, 32, 1])
    #>>> model = VGG16(input_shape=inp_tensor.shape)
    #>>> output = model(inp_tensor)
    #>>> output.shape
    #torch.Size([1,4])
    """
    def __init__(
        self,
        input_shape=None,
        dense_n_neurons=4,
        dropout=0.5,

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
                out_channels=25,
                kernel_size=(5,1),
                padding="valid",
                bias=True,
                swap=True,
            ),)
        
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
            ),)
        
        # activation
        self.conv_module.add_module("act_1", torch.nn.ELU())
        
        # max pooling
        self.conv_module.add_module(
            "pool_1",
            sb.nnet.pooling.Pooling2d(
                kernel_size=(2,1),
                pool_axis=[1, 2],
                pool_type="max",
            ),)
        
        # dropout
        self.conv_module.add_module(
            "drop_1",
            torch.nn.Dropout(p=dropout),
        )

        
        # convolutional block 2
        self.conv_module.add_module(
            "conv2",
            sb.nnet.CNN.Conv2d(
                in_channels=25,
                out_channels=50,
                kernel_size=(5, 1),
                padding="valid",
                bias=False,
                swap=True,
            ),)
        
        self.conv_module.add_module(
            "conv3",
            sb.nnet.CNN.Conv2d(
                in_channels=50,
                out_channels=50,
                kernel_size=(5,1),
                padding="valid",
                bias=False,
                swap=True,
            ),)
        
        # batch norm 2
        self.conv_module.add_module(
            "bnorm_2",
            sb.nnet.normalization.BatchNorm2d(
                input_size=50, momentum=0.1, affine=True,
            ),)
        
        # activation 2
        self.conv_module.add_module("act_2", torch.nn.ELU())

        # max pooling
        self.conv_module.add_module(
            "pool_2",
            sb.nnet.pooling.Pooling2d(
                kernel_size=(2,1),
                pool_axis=[1, 2],
                pool_type="max",
            ),)
        
        # dropout 2
        self.conv_module.add_module(
            "drop_2",
            torch.nn.Dropout(p=dropout),)
        
        
        # convolutional block 3
        self.conv_module.add_module(
            "conv4",
            sb.nnet.CNN.Conv2d(
                in_channels=50,
                out_channels=75,
                kernel_size=(5,1),
                padding="valid",
                bias=False,
                swap=True,
            ),)
        
        self.conv_module.add_module(
            "conv5",
            sb.nnet.CNN.Conv2d(
                in_channels=75,
                out_channels=100,
                kernel_size=(5,1),
                padding="valid",
                bias=False,
                swap=True,
            ),)
        
        self.conv_module.add_module(
            "conv6",
            sb.nnet.CNN.Conv2d(
                in_channels=100,
                out_channels=125,
                kernel_size=(5,1),
                padding="valid",
                bias=False,
                swap=True,
            ),)
        
        # batch norm
        self.conv_module.add_module(
            "bnorm_3",
            sb.nnet.normalization.BatchNorm2d(
                input_size=125, momentum=0.1, affine=True,
            ),)
        
        # activation
        self.conv_module.add_module("act_3", torch.nn.ELU())

        # max pooling
        self.conv_module.add_module(
            "pool_3",
            sb.nnet.pooling.Pooling2d(
                kernel_size=(2,1),
                pool_axis=[1, 2],
                pool_type="max",
            ),)
        
        # dropout
        self.conv_module.add_module(
            "drop_3",
            torch.nn.Dropout(p=dropout),)
        
        # convolutional block 4
        self.conv_module.add_module(
            "conv7",
            sb.nnet.CNN.Conv2d(
                in_channels=125,
                out_channels=150,
                kernel_size=(5,1),
                padding="valid",
                bias=False,
                swap=True,
            ),)
        
        self.conv_module.add_module(
            "conv8",
            sb.nnet.CNN.Conv2d(
                in_channels=150,
                out_channels=175,
                kernel_size=(5,1),
                padding="valid",
                bias=False,
                swap=True,
            ),)
        
        self.conv_module.add_module(
            "conv9",
            sb.nnet.CNN.Conv2d(
                in_channels=175,
                out_channels=175,
                kernel_size=(5,1),
                padding="valid",
                bias=False,
                swap=True,
            ),)
        
        # batch norm
        self.conv_module.add_module(
            "bnorm_4",
            sb.nnet.normalization.BatchNorm2d(
                input_size=175, momentum=0.1, affine=True,
            ),)
        
        # activation
        self.conv_module.add_module("act_4", torch.nn.ELU())

        
        # max pooling
        self.conv_module.add_module(
            "pool_4",
            sb.nnet.pooling.Pooling2d(
                kernel_size=(2,1),
                pool_axis=[1, 2],
                pool_type="max",
            ),)
        
        # dropout
        self.conv_module.add_module(
            "drop_4",
            torch.nn.Dropout(p=dropout),)
        
        
        # convolutional block 5
        self.conv_module.add_module(
            "conv10",
            sb.nnet.CNN.Conv2d(
                in_channels=175,
                out_channels=200,
                kernel_size=(5,1),
                padding="valid",
                bias=False,
                swap=True,
            ),)
        
        self.conv_module.add_module(
            "conv11",
            sb.nnet.CNN.Conv2d(
                in_channels=200,
                out_channels=225,
                kernel_size=(5,1),
                padding="valid",
                bias=False,
                swap=True,
            ),)
        
        self.conv_module.add_module(
            "conv12",
            sb.nnet.CNN.Conv2d(
                in_channels=225,
                out_channels=225,
                kernel_size=(5,1),
                padding="valid",
                bias=False,
                swap=True,
            ),)
        
        # batch norm
        self.conv_module.add_module(
            "bnorm_5",
            sb.nnet.normalization.BatchNorm2d(
                input_size=225, momentum=0.1, affine=True,
            ),)
        
        # activation
        self.conv_module.add_module("act_5", torch.nn.ELU())
        
        # max pooling
        self.conv_module.add_module(
            "pool_5",
            sb.nnet.pooling.Pooling2d(
                kernel_size=(2,1),
                pool_axis=[1, 2],
                pool_type="max",
            ),)
        
        # dropout
        self.conv_module.add_module(
            "drop_5",
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