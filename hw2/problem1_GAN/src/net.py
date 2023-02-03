import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.autograd as autograd
from torch.autograd import Variable


# https://github.com/carpedm20/BEGAN-pytorch
# https://github.com/Natsu6767/DCGAN-PyTorch
# Define the Generator Network


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# ----------------------- these are models from ML ----------------------------
class DC_Generator_ML(nn.Module):
    """
    Input shape: (batch, in_dim)
    Output shape: (batch, 3, 64, 64)
    """

    def __init__(self, in_dim=100, feat_dim=64):
        super().__init__()

        # input: (batch, 100)
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, feat_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(feat_dim * 8 * 4 * 4),
            nn.ReLU(),
        )
        self.l2 = nn.Sequential(
            self.dconv_bn_relu(feat_dim * 8, feat_dim * 4),
            self.dconv_bn_relu(feat_dim * 4, feat_dim * 2),
            self.dconv_bn_relu(feat_dim * 2, feat_dim),
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(
                feat_dim,
                3,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )
        self.apply(weights_init)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear or nn.ConvTranspose2d):
        #         m.weight.data = nn.init.xavier_uniform_(
        #             m.weight.data, gain=nn.init.calculate_gain("relu")
        #         )

    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_dim,
                out_dim,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),  # double height and width
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2(y)
        y = self.l3(y)
        return y


class DC_Discriminator_ML(nn.Module):
    """
    Input shape: (batch, 3, 64, 64)
    Output shape: (batch)
    """

    def __init__(self, in_dim=3, feat_dim=64):
        super(DC_Discriminator_ML, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_dim, feat_dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            self.conv_bn_lrelu(feat_dim, feat_dim * 2),  # (batch, 3, 16, 16)
            self.conv_bn_lrelu(feat_dim * 2, feat_dim * 4),  # (batch, 3, 8, 8)
            self.conv_bn_lrelu(feat_dim * 4, feat_dim * 8),  # (batch, 3, 4, 4)
            nn.Conv2d(feat_dim * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )
        self.apply(weights_init)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear or nn.ConvTranspose2d):
        #         m.weight.data = nn.init.xavier_uniform_(
        #             m.weight.data, gain=nn.init.calculate_gain("relu")
        #         )

    def conv_bn_lrelu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 5, 2, 2),
            nn.InstanceNorm2d(out_dim),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        y = self.l1(x)  # [64,1,1,1]
        y = y.view(-1)  # [64]
        return y


# --------------------------------------from original DCGAN paper--------------------------------------------#
# reference: https://github.com/Natsu6767/DCGAN-PyTorch


class DC_Generator_origin(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input is the latent vector Z.
        self.tconv1 = nn.ConvTranspose2d(
            params["nz"],
            params["ngf"] * 8,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(params["ngf"] * 8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(
            params["ngf"] * 8, params["ngf"] * 4, 4, 2, 1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(params["ngf"] * 4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(
            params["ngf"] * 4, params["ngf"] * 2, 4, 2, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(params["ngf"] * 2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(
            params["ngf"] * 2, params["ngf"], 4, 2, 1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(params["ngf"])

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(
            params["ngf"], params["nc"], 4, 2, 1, bias=False
        )
        for m in self.modules():
            if isinstance(m, nn.Linear or nn.Conv2d):
                m.weight.data = nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

        # Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = F.tanh(self.tconv5(x))

        return x


# Define the Discriminator Network
class DC_Discriminator_origin(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(params["nc"], params["ndf"], 4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(params["ndf"], params["ndf"] * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params["ndf"] * 2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(
            params["ndf"] * 2, params["ndf"] * 4, 4, 2, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(params["ndf"] * 4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(
            params["ndf"] * 4, params["ndf"] * 8, 4, 2, 1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(params["ndf"] * 8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(params["ndf"] * 8, 1, 4, 1, 0, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear or nn.Conv2d):
                m.weight.data = nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)

        x = F.sigmoid(self.conv5(x))

        return x.squeeze()


# ------------------WGAN from ML class--------------------------------#


class WGAN_Generator(nn.Module):
    """
    Input shape: (batch, in_dim)
    Output shape: (batch, 3, 64, 64)
    """

    def __init__(self, in_dim, dim=64):
        super().__init__()

        # input: (batch, 100)
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU(),
        )
        self.l2 = nn.Sequential(
            self.dconv_bn_relu(dim * 8, dim * 4),  # (batch, dim * 16, 8, 8)
            self.dconv_bn_relu(dim * 4, dim * 2),  # (batch, dim * 16, 16, 16)
            self.dconv_bn_relu(dim * 2, dim),  # (batch, dim * 16, 32, 32)
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(
                dim, 3, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False
            ),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_dim,
                out_dim,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
                bias=False,
            ),  # double height and width
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2(y)
        y = self.l3(y)
        return y


class WGAN_Discriminator(nn.Module):
    """
    Input shape: (batch, 3, 64, 64)
    Output shape: (batch)
    """

    def __init__(self, in_dim, dim=64):
        super(WGAN_Discriminator, self).__init__()

        # input: (batch, 3, 64, 64)
        """
        NOTE FOR SETTING DISCRIMINATOR:

        Remove last sigmoid layer for WGAN
        """
        self.l1 = nn.Sequential(
            nn.Conv2d(
                in_dim, dim, kernel_size=4, stride=2, padding=1
            ),  # (batch, 3, 32, 32)
            nn.LeakyReLU(0.2),
            self.conv_bn_lrelu(dim, dim * 2),  # (batch, 3, 16, 16)
            self.conv_bn_lrelu(dim * 2, dim * 4),  # (batch, 3, 8, 8)
            self.conv_bn_lrelu(dim * 4, dim * 8),  # (batch, 3, 4, 4)
            nn.Conv2d(dim * 8, 1, kernel_size=4, stride=1, padding=0),
            # nn.Sigmoid(),
        )
        self.apply(weights_init)

    def conv_bn_lrelu(self, in_dim, out_dim):
        """
        NOTE FOR SETTING DISCRIMINATOR:

        You can't use nn.Batchnorm for WGAN-GP
        Use nn.InstanceNorm2d instead
        """

        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.BatchNorm2d(out_dim),
            nn.InstanceNorm2d(out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(-1)
        return y
