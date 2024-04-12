"""
rcsab.py 2023/3/8 18:26
Written by Wensheng Fan
"""
import torch.nn as nn
from models.modules.cat import CATBlock


class RCSAB(nn.Module):
    def __init__(
            self, in_chans, out_chans, input_resolution, expansion=4, m_chans=0, latent_dim=None):

        super(RCSAB, self).__init__()
        if m_chans == 0:
            m_chans = in_chans * expansion
        self.expan_conv = nn.Sequential(
            nn.Conv2d(in_chans, m_chans, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(
            CATBlock(input_resolution=input_resolution, in_chan=m_chans, latent_dim=latent_dim),
            nn.Conv2d(m_chans, m_chans, kernel_size=3, stride=1, padding=1),
        )
        self.compression = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(m_chans, out_chans, kernel_size=1)
        )

    def forward(self, x):
        x = self.expan_conv(x)
        res = x
        res = self.body(res)
        res += x
        res = self.compression(res)
        return res