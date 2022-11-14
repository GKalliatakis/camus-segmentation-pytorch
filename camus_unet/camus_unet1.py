"""
Replica of the U-Net 1 architecture presented in the original CAMUS manuscript
- https://arxiv.org/pdf/1908.06948.pdf
- https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8649738


Upsampling vs Transposed convolutions:
    The original U-Net paper uses transposed convolutions (a.k.a. upconvolutions,
    a.k.a. fractionally-strided convolutions, a.k.a deconvolutions) in the "up" pathway.
    Other implementations use (bilinear) upsampling, possibly followed by a 1x1 convolution.
    The benefit of using upsampling is that it has no parameters and if you include the 1x1 convolution,
    it will still have less parameters than the transposed convolution.
    The downside is that it can't use weights to combine the spatial information in a smart way,
    so transposed convolutions can potentially handle more fine-grained detail.

"""
import torch
import torch.nn as nn


def double_conv2d(in_channel, out_channel):
    """
    (convolution => ReLU) * 2
    """
    convLayer = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return convLayer


class CamusUnet1(nn.Module):
    def __init__(self, bilinear=True):
        """
        bilinear (bool): Whether to use bilinear interpolation or transposed convolutions for upsampling.
            The original CAMUS manuscript uses bilinear interpolation for U-NET 1 architecture.
        """
        super(CamusUnet1, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.d1 = double_conv2d(1, 32)
        self.d2 = double_conv2d(32, 32)
        self.d3 = double_conv2d(32, 64)
        self.d4 = double_conv2d(64, 128)
        self.d5 = double_conv2d(128, 128)
        self.d6 = double_conv2d(128, 128)

        if bilinear:
            self.UpSampling1 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1)))
            self.UpSampling2 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                             nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)))
            self.UpSampling3 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                             nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1)))
            self.UpSampling4 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 1)))
        else:
            self.UpSampling1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
            self.UpSampling2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2, 2), stride=(2, 2))
            self.UpSampling3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2, 2), stride=(2, 2))
            self.UpSampling4 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(2, 2), stride=(2, 2))

        self.u1 = double_conv2d(256, 128)  # stacking prev layer..so size increased by 2
        self.u2 = double_conv2d(128, 64)
        self.u3 = double_conv2d(64, 32)
        self.u4 = double_conv2d(64, 16)  # Layer Up layer: 32 + 32 =64 to Double conv to 16
        self.out = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(1, 1))

    def forward(self, image):
        """
        image shape: batch_size, channel, height, width
        """
        """
        Encoder
        """
        # Layer1
        d1_out = self.d1(image)

        # Layer2
        d2_max_pool_out = self.max_pool(d1_out)
        d2_out = self.d2(d2_max_pool_out)

        # Layer3
        d3_max_pool_out = self.max_pool(d2_out)
        d3_out = self.d3(d3_max_pool_out)

        # Layer4
        d4_max_pool_out = self.max_pool(d3_out)
        d4_out = self.d4(d4_max_pool_out)

        # Layer5
        d5_max_pool_out = self.max_pool(d4_out)
        d5_out = self.d5(d5_max_pool_out)

        # Layer6
        d6_max_pool_out = self.max_pool(d5_out)
        d6_out = self.d6(d6_max_pool_out)

        """
        Decoder
        """
        up_sampling1 = self.UpSampling1(d6_out)
        u1_out = self.u1(torch.cat([d5_out, up_sampling1], axis=1))

        up_sampling2 = self.UpSampling1(u1_out)
        u2_out = self.u1(torch.cat([d4_out, up_sampling2], axis=1))

        up_sampling3 = self.UpSampling2(u2_out)
        u3_out = self.u2(torch.cat([d3_out, up_sampling3], axis=1))

        up_sampling4 = self.UpSampling3(u3_out)
        u4_out = self.u3(torch.cat([d2_out, up_sampling4], axis=1))

        up_sampling5 = self.UpSampling4(u4_out)
        u5_out = self.u4(torch.cat([d1_out, up_sampling5], axis=1))

        final_out = self.out(u5_out)
        return final_out
