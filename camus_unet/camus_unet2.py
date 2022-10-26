"""
Replica of the U-Net 2 architecture presented in the original CAMUS manuscript
- https://arxiv.org/pdf/1908.06948.pdf
- https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8649738
"""

import torch
import torch.nn as nn


def double_conv2d(in_channel, out_channel):
    """
    (convolution => [BN] => ReLU) * 2
    """
    convLayer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )
    return convLayer


class CamusUnet2(nn.Module):
    def __init__(self):
        super(CamusUnet2, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downConvLayer_1 = double_conv2d(1, 48)
        self.downConvLayer_2 = double_conv2d(48, 96)
        self.downConvLayer_3 = double_conv2d(96, 192)
        self.downConvLayer_4 = double_conv2d(192, 384)
        self.downConvLayer_5 = double_conv2d(384, 768)

        self.upTransConv1 = nn.ConvTranspose2d(in_channels=768, out_channels=384, kernel_size=(2, 2), stride=(2, 2))
        self.upConvLayer_1 = double_conv2d(768, 384)

        self.upTransConv2 = nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=(2, 2), stride=(2, 2))
        self.upConvLayer_2 = double_conv2d(384, 192)

        self.upTransConv3 = nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=(2, 2), stride=(2, 2))
        self.upConvLayer_3 = double_conv2d(192, 96)

        self.upTransConv4 = nn.ConvTranspose2d(in_channels=96, out_channels=48, kernel_size=(2, 2), stride=(2, 2))
        self.upConvLayer_4 = double_conv2d(96, 48)

        self.out = nn.Conv2d(in_channels=48, out_channels=4, kernel_size=(1,1))

    def forward(self, image):
        """
        image shape: batch_size, channel, height, width
        """
        """
        Encoder
        """
        # Layer1
        d1_out = self.downConvLayer_1(image)

        # Layer2
        d2_max_pool_out = self.max_pool(d1_out)
        d2_out = self.downConvLayer_2(d2_max_pool_out)

        # Layer3
        d3_max_pool_out = self.max_pool(d2_out)
        d3_out = self.downConvLayer_3(d3_max_pool_out)

        # Layer4
        d4_max_pool_out = self.max_pool(d3_out)
        d4_out = self.downConvLayer_4(d4_max_pool_out)

        # Layer5
        d5_max_pool_out = self.max_pool(d4_out)
        d5_out = self.downConvLayer_5(d5_max_pool_out)

        """
        Decoder
        """
        up_sampling1 = self.upTransConv1(d5_out)
        u1_out = self.upConvLayer_1(torch.cat([d4_out, up_sampling1], axis=1))

        up_sampling2 = self.upTransConv2(u1_out)
        u2_out = self.upConvLayer_2(torch.cat([d3_out, up_sampling2], axis=1))

        up_sampling3 = self.upTransConv3(u2_out)
        u3_out = self.upConvLayer_3(torch.cat([d2_out, up_sampling3], axis=1))

        up_sampling4 = self.upTransConv4(u3_out)
        u4_out = self.upConvLayer_4(torch.cat([d1_out, up_sampling4], axis=1))

        final_out = self.out(u4_out)
        return final_out
