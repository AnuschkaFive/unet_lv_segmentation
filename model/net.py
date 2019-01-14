"""
Defines the neural network architecture.
Based on code by Marco Pavlowski (and https://tuatini.me/practical-image-segmentation-with-unet/).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, hyper_params):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.use_bn = hyper_params.batch_norm
        self.bn = nn.BatchNorm2d(out_channels)
        activation = getattr(nn, hyper_params.activation, None)
        assert activation is not None, "Activation Fn {} couldn't be found!".format(hyper_params.activation)
        self.relu = activation()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            print("BatchNorm!")
            x = self.bn(x)
        x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, pool_kernel_size, pool_stride, hyper_params):
        super(StackEncoder, self).__init__()
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, hyper_params=hyper_params)
        self.dropout = nn.Dropout2d(p=hyper_params.dropout_rate)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, hyper_params=hyper_params)
        self.maxPool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

    def forward(self, x):
        x = self.convr1(x)
        x = self.dropout(x)
        x = self.convr2(x)
        x_trace = x
        x = self.maxPool(x)
        return x, x_trace


class StackDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, upsample_size, hyper_params):
        super(StackDecoder, self).__init__()
        self.use_crop_concat = in_channels != out_channels
        #self.upSample = nn.Upsample(size=upsample_size, mode="bilinear")
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, hyper_params=hyper_params)
        # Crop + concat step between these 2
        self.convr2 = ConvBnRelu(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, hyper_params=hyper_params)
        self.upsample_size = upsample_size
        
    def _crop_concat(self, upsampled, bypass):
        """
        Crop y to the (h, w) of x and concat them.
        Used for the expansive path.
        Returns:
            The concatenated tensor.
        """       
        #TODO: make work for uneven numbers!
        cH = (bypass.size()[2] - upsampled.size()[2]) // 2
        cW = (bypass.size()[3] - upsampled.size()[3]) // 2
        #print('Bypass Size: {} {}, Upsampled Size: {} {}'.format(bypass.size()[2], bypass.size()[3], upsampled.size()[2], upsampled.size()[3]))
        bypass = F.pad(bypass, (-cW, -cW, -cH, -cH))

        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, down_tensor):
        #x = self.upSample(x)
        x = F.interpolate(x, self.upsample_size)
        if not self.use_crop_concat:
            x = x.add(down_tensor)
        x = self.convr1(x)
        if self.use_crop_concat:
            x = self._crop_concat(x, down_tensor)
        x = self.convr2(x)
        return x


class UNetOriginal(nn.Module):
    def __init__(self, hyper_params):
        super(UNetOriginal, self).__init__()
        # TODO: works for input image sizes 360x360 (thanks Marco!)
        kSize = (3,3)
        pad = 1  # padding 1 so ksize 3 doesn't shorten the output
        stride = 1
        pool_kSize = (2,2)
        pool_stride = 2

        self.down1 = StackEncoder(1, 64, kernel_size=kSize, padding=pad, stride=stride, pool_kernel_size=pool_kSize, pool_stride=pool_stride, hyper_params=hyper_params)
        self.down2 = StackEncoder(64, 128, kernel_size=kSize, padding=pad, stride=stride, pool_kernel_size=pool_kSize, pool_stride=pool_stride, hyper_params=hyper_params)
        self.down3 = StackEncoder(128, 256, kernel_size=kSize, padding=pad, stride=stride, pool_kernel_size=pool_kSize, pool_stride=pool_stride, hyper_params=hyper_params)
        self.down4 = StackEncoder(256, 512, kernel_size=kSize, padding=pad, stride=stride, pool_kernel_size=pool_kSize, pool_stride=pool_stride, hyper_params=hyper_params)

        self.center = nn.Sequential(
            ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=0, hyper_params=hyper_params),
            ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=0, hyper_params=hyper_params)
        )

#        self.up1 = StackDecoder(in_channels=1024, out_channels=512, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(56, 56))
#        self.up2 = StackDecoder(in_channels=512, out_channels=256, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(104, 104))
#        self.up3 = StackDecoder(in_channels=256, out_channels=128, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(200, 200))
#        self.up4 = StackDecoder(in_channels=128, out_channels=64, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(392, 392))
        self.up1 = StackDecoder(in_channels=1024, out_channels=512, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(40, 40), hyper_params=hyper_params)
        self.up2 = StackDecoder(in_channels=512, out_channels=256, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(80, 80), hyper_params=hyper_params)
        self.up3 = StackDecoder(in_channels=256, out_channels=128, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(160, 160), hyper_params=hyper_params)
        self.up4 = StackDecoder(in_channels=128, out_channels=64, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(320, 320), hyper_params=hyper_params)
        # 1x1 convolution at the last layer
        # Different from the paper is the output size here
        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)

    def forward(self, x):
        x, x_trace1 = self.down1(x)  # Calls the forward() method of each layer
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        x = self.center(x)

        x = self.up1(x, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)
        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)
        #out = torch.squeeze(out, dim=1)
        return out

class UNetOriginalPretrained(nn.Module):
    def __init__(self, hyper_params):
        super(UNetOriginalPretrained, self).__init__()
        # TODO: works for input image sizes 360x360 (thanks Marco!)
        kSize = (3,3)
        pad = 1  # padding 1 so ksize 3 doesn't shorten the output
        stride = 1
        pool_kSize = (2,2)
        pool_stride = 2
        
        encoder = models.vgg11(pretrained=True).features

        self.down1 = StackEncoder(1, 64, kernel_size=kSize, padding=pad, stride=stride, pool_kernel_size=pool_kSize, pool_stride=pool_stride, hyper_params=hyper_params)
        self.down1.convr1.conv.weight = torch.nn.Parameter(torch.narrow(torch.tensor(encoder[0].weight), 1, 0, 1))
        self.down1.convr1.conv.bias = encoder[0].bias
        self.down2 = StackEncoder(64, 128, kernel_size=kSize, padding=pad, stride=stride, pool_kernel_size=pool_kSize, pool_stride=pool_stride, hyper_params=hyper_params)
        self.down2.convr1.conv.weight = encoder[3].weight
        self.down2.convr1.conv.bias = encoder[3].bias
        self.down3 = StackEncoder(128, 256, kernel_size=kSize, padding=pad, stride=stride, pool_kernel_size=pool_kSize, pool_stride=pool_stride, hyper_params=hyper_params)
        self.down3.convr1.conv.weight = encoder[6].weight
        self.down3.convr1.conv.bias = encoder[6].bias        
        self.down4 = StackEncoder(256, 512, kernel_size=kSize, padding=pad, stride=stride, pool_kernel_size=pool_kSize, pool_stride=pool_stride, hyper_params=hyper_params)
        self.down4.convr1.conv.weight = encoder[11].weight
        self.down4.convr1.conv.bias = encoder[11].bias 
        
        self.center = nn.Sequential(
            ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=0, hyper_params=hyper_params),
            ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=0, hyper_params=hyper_params)
        )

        self.up1 = StackDecoder(in_channels=1024, out_channels=512, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(40, 40), hyper_params=hyper_params)
        self.up2 = StackDecoder(in_channels=512, out_channels=256, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(80, 80), hyper_params=hyper_params)
        self.up3 = StackDecoder(in_channels=256, out_channels=128, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(160, 160), hyper_params=hyper_params)
        self.up4 = StackDecoder(in_channels=128, out_channels=64, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(320, 320), hyper_params=hyper_params)
        # 1x1 convolution at the last layer
        # Different from the paper is the output size here
        self.output_seg_map = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)

    def forward(self, x):
        x, x_trace1 = self.down1(x)  # Calls the forward() method of each layer
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        x = self.center(x)

        x = self.up1(x, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)
        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)
        #out = torch.squeeze(out, dim=1)
        return out
    
    
class ModelA(nn.Module):
    def __init__(self, hyper_params):
        super(ModelA, self).__init__()
        # TODO: works for input image sizes 360x360 (thanks Marco!)
        kSize = (3,3)
        pad = 1  # padding 1 so ksize 3 doesn't shorten the output
        stride = 1
        pool_kSize = (2,2)
        pool_stride = 2

        self.down1 = StackEncoder(1, 32, kernel_size=kSize, padding=pad, stride=stride, pool_kernel_size=pool_kSize, pool_stride=pool_stride, hyper_params=hyper_params)
        self.down2 = StackEncoder(32, 32, kernel_size=kSize, padding=pad, stride=stride, pool_kernel_size=pool_kSize, pool_stride=pool_stride, hyper_params=hyper_params)
        self.down3 = StackEncoder(32, 32, kernel_size=kSize, padding=pad, stride=stride, pool_kernel_size=pool_kSize, pool_stride=pool_stride, hyper_params=hyper_params)
        self.down4 = StackEncoder(32, 32, kernel_size=kSize, padding=pad, stride=stride, pool_kernel_size=pool_kSize, pool_stride=pool_stride, hyper_params=hyper_params)

        self.center = nn.Sequential(
            ConvBnRelu(32, 32, kernel_size=(3, 3), stride=1, padding=0, hyper_params=hyper_params),
            ConvBnRelu(32, 32, kernel_size=(3, 3), stride=1, padding=0, hyper_params=hyper_params)
        )
    
        self.up1 = StackDecoder(in_channels=32, out_channels=32, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(40, 40), hyper_params=hyper_params)
        self.up2 = StackDecoder(in_channels=32, out_channels=32, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(80, 80), hyper_params=hyper_params)
        self.up3 = StackDecoder(in_channels=32, out_channels=32, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(160, 160), hyper_params=hyper_params)
        self.up4 = StackDecoder(in_channels=32, out_channels=32, kernel_size=kSize, padding=pad, stride=stride, upsample_size=(320, 320), hyper_params=hyper_params)
        # 1x1 convolution at the last layer
        # Different from the paper is the output size here
        self.output_seg_map = nn.Conv2d(32, 1, kernel_size=(1, 1), padding=0, stride=1)

    def forward(self, x):
        x, x_trace1 = self.down1(x)  # Calls the forward() method of each layer
        x, x_trace2 = self.down2(x)
        x, x_trace3 = self.down3(x)
        x, x_trace4 = self.down4(x)

        x = self.center(x)

        x = self.up1(x, x_trace4)
        x = self.up2(x, x_trace3)
        x = self.up3(x, x_trace2)
        x = self.up4(x, x_trace1)

        out = self.output_seg_map(x)
        #out = torch.squeeze(out, dim=1)
        return out


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    #nameclass = m.__class__
    print(classname)
    if classname.find('Conv2d') != -1:
        m.weight.data.fill_(0.0)
#    elif classname.find('BatchNorm') != -1:
#        m.weight.data.normal_(1.0, 0.02)
#        m.bias.data.fill_(0)

#class Net(nn.Module):
#    """
#    This is the standard way to define your own network in PyTorch. You typically choose the components
#    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
#    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
#    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
#    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
#    you can go about defining your own network.
#    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
#    """
#
#    def __init__(self, params):
#        """
#        We define an convolutional network that predicts the sign from an image. The components
#        required are:
#        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
#        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
#        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags
#        Args:
#            params: (Params) contains num_channels
#        """
#        super(Net, self).__init__()
#        self.num_channels = params.num_channels
#        
#        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
#        # stride, padding). We also include batch normalisation layers that help stabilise training.
#        # For more details on how to use these layers, check out the documentation.
#        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
#        self.bn1 = nn.BatchNorm2d(self.num_channels)
#        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
#        self.bn2 = nn.BatchNorm2d(self.num_channels*2)
#        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
#        self.bn3 = nn.BatchNorm2d(self.num_channels*4)
#
#        # 2 fully connected layers to transform the output of the convolution layers to the final output
#        self.fc1 = nn.Linear(8*8*self.num_channels*4, self.num_channels*4)
#        self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
#        self.fc2 = nn.Linear(self.num_channels*4, 6)       
#        self.dropout_rate = params.dropout_rate
#
#    def forward(self, s):
#        """
#        This function defines how we use the components of our network to operate on an input batch.
#        Args:
#            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .
#        Returns:
#            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.
#        Note: the dimensions after each step are provided
#        """
#        #                                                  -> batch_size x 3 x 64 x 64
#        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
#        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 64 x 64
#        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 32 x 32
#        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 32 x 32
#        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 16 x 16
#        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 16 x 16
#        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 8 x 8
#
#        # flatten the output for each image
#        s = s.view(-1, 8*8*self.num_channels*4)             # batch_size x 8*8*num_channels*4
#
#        # apply 2 fully connected layers with dropout
#        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), 
#            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
#        s = self.fc2(s)                                     # batch_size x 6
#
#        # apply log softmax on each image's output (this is recommended over applying softmax
#        # since it is numerically more stable)
#        return F.log_softmax(s, dim=1)




