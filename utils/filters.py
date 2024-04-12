"""
filters.py 2022/5/18 12:47
Written by Wensheng Fan
"""
import torch
import torch.nn.functional as F
import numpy as np
import math


def sobelfilter2d(x, split=False):
    """ Apply sobel filter to input """
    N, C, H, W = x.shape
    # sfilter1 = torch.tensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32).reshape(1, 1, 3, 3)
    sfilter1 = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    sfilter2 = torch.transpose(sfilter1, 2, 3)
    sfilter = torch.cat((sfilter1, sfilter2), dim=0).repeat(C, 1, 1, 1).cuda()
    x = torch.repeat_interleave(x, 2, dim=1)
    x_pad = F.pad(x, pad=[1, 1, 1, 1], mode='reflect')
    return F.conv2d(x_pad, weight=sfilter, bias=None, stride=1, padding=0, groups=2 * C)


def _tf_fspecial_gauss(size, sigma):
    """ Function to mimic the 'fspecial' gaussian MATLAB function """
    x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

    x_data = np.expand_dims(x_data, axis=0)
    x_data = np.expand_dims(x_data, axis=0)

    y_data = np.expand_dims(y_data, axis=0)
    y_data = np.expand_dims(y_data, axis=0)

    x = torch.tensor(x_data, dtype=torch.float32)
    y = torch.tensor(y_data, dtype=torch.float32)

    g = torch.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    return g / torch.sum(g)


def gaussblur_fsigma(x, fsig):
    """
    Apply gaussian blur to a torch.Tensor input. The bigger fsig, the blurrer output.

    Args:
        x: 4-D input
        fsig: STD for Gaussian filter
    """
    szf = 6 * fsig - 1
    szf = max(math.floor(szf / 2) * 2 + 1, 3)

    N, C, H, W = x.shape
    bf = _tf_fspecial_gauss(szf, fsig).cuda()
    bf = bf.repeat(C, 1, 1, 1)

    pp = int((szf - 1) / 2)
    x_pad = F.pad(x, pad=[pp, pp, pp, pp], mode='reflect')
    return F.conv2d(x_pad, weight=bf, bias=None, stride=1, padding=0, groups=C)


def gaussblur_fsize(x, szf, rr=1):
    """ Apply gaussian blur to input

    Args:
        szf: filter size.
        rr: dilation rate
    """
    fsig = (szf + 1) / 6  # n = 6*sig-1
    N, C, H, W = x.shape

    bf = _tf_fspecial_gauss(szf, fsig).cuda()
    bf = bf.repeat(C, 1, 1, 1)

    pp = int((szf + (szf - 1) * (rr - 1) - 1) / 2)
    x_pad = F.pad(x, pad=[pp, pp, pp, pp], mode='reflect')
    return F.conv2d(x_pad, weight=bf, bias=None, stride=1, padding=0, dilation=rr, groups=C)
