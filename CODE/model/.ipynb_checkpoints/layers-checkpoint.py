import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class SelfAttention(nn.Module):
    def __init__(self,  d_model):
        super(SelfAttention,self).__init__()

        self.d_model = d_model
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)


    def forward(self, xin):

        x=xin.reshape(xin.shape[0],xin.shape[1],-1).permute(0,2,1)
        # This applies the dense layer to each of the 50-dimensional each of the words of each of the sentences of x
        # Result is. number_of_sentences x number_words x 50.
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)

        # Result is number_of_sentences x number_of_words x number_of_words
        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(self.d_model)

        scores = F.softmax(scores, dim=-1)

        # Result is number_of_sentences x number_of_words x 50
        scores = torch.matmul(scores, v)
        scores = scores.permute(0,1,2)
        scores=scores.reshape(xin.shape[0],xin.shape[1],xin.shape[2],xin.shape[3])

        return scores



class DoubleConv(nn.Module): # HxWxD
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(mid_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(out_channels, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResConv(nn.Module): # HxWxD
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(mid_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(out_channels, out_channels),
        )
        if in_channels != out_channels:
            self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        else:
            self.identity = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.double_conv(x)
        out += self.identity(x)
        return self.relu(out)

class TrippleConv(nn.Module): # HxWxD
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.tripple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(mid_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(mid_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.GroupNorm(out_channels, out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.tripple_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, padding=0, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x1, x2): # C*H*W
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class DownRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpRes(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ResConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, padding=0, stride=2)
            self.conv = ResConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x1, x2): # C*H*W
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)