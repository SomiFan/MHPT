"""
cat.py 2023/3/7 22:24
Written by Wensheng Fan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class CSAttention(nn.Module):
    r""" Channel Self Attention (CSA) module.

    Args:
        dim (int): Number of input channels.
        latent_dim (int): Number of latent dimension.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    """

    def __init__(self, dim, latent_dim=None, qkv_bias=True, qk_scale=None):

        super().__init__()
        self.latent_dim = latent_dim or dim
        self.scale = qk_scale or self.latent_dim ** -0.5

        self.qk = nn.Linear(dim, self.latent_dim * 2, bias=qkv_bias)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        """
        Args:
            x: global descriptor (num_windows*B, N, C)
            y: input features of image
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = y.shape
        qk = self.qk(x).reshape(B_, N, 2, self.latent_dim).permute(2, 0, 1, 3)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)

        # dot product
        q = q * self.scale
        attn = self.softmax(torch.matmul(q, k.transpose(-2, -1)))

        x = torch.matmul(attn, y)
        return x


class CATBlock(nn.Module):
    """
    Channel Attention Transformer block
    """
    def __init__(self, input_resolution, in_chan, qkv_bias=True, norm_layer=nn.LayerNorm, latent_dim=None, qk_scale=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduced_res = input_resolution // 8
        self.in_chan = in_chan

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=in_chan, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_chan, out_channels=in_chan, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_chan, out_channels=in_chan, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )
        #self.conv_layers = nn.AdaptiveAvgPool2d(self.reduced_res)

        self.norm1 = norm_layer(self.reduced_res ** 2)
        self.norm2 = norm_layer(self.reduced_res ** 2)

        self.attn = CSAttention(dim=self.reduced_res ** 2, latent_dim=latent_dim, qkv_bias=qkv_bias, qk_scale=qk_scale)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.input_resolution, "input feature has wrong size"

        shortcut = x
        glob_descriptor = self.conv_layers(x)
        #glob_descriptor = F.interpolate(x, size=[self.reduced_res, self.reduced_res], mode='bilinear', align_corners=True)
        glob_descriptor = glob_descriptor.view(B, C, self.reduced_res ** 2)
        glob_descriptor = self.norm1(glob_descriptor)
        x = x.view(B, C, self.input_resolution ** 2)

        # W-MSA/SW-MSA
        x = self.attn(glob_descriptor, x)  # nW*B, window_size*window_size, C
        x = x.view(B, C, H, W)
        x = shortcut + x

        return x

    def flops(self):
        # calculate macs for 1 window with token length of N
        flops = 0
        N = self.in_chan
        # attn = (q @ k.transpose(-2, -1))
        flops += N * (self.reduced_res ** 2) * N
        #  x = (attn @ v) CxC matrix multiply CxHW matrix
        flops += N * N * (self.input_resolution ** 2)
        return flops


if __name__ == '__main__':
    from torchsummary import summary

    N2 = CATBlock(input_resolution=128, in_chan=60, latent_dim=128).cuda()
    from thop import profile
    from thop import clever_format

    u = torch.randn(1, 60, 128, 128).cuda()
    macs, _ = profile(N2, inputs=(u, ))
    macs, _ = clever_format([macs, _], '%.3f')

    print('Computational complexity:', macs)

    # count the Macs of attention
    flops = N2.flops()
    print(f"number of GMacs: {flops / 1e9}")
