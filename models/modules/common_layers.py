"""
common_layers.py 2023/3/10 12:38
Written by Wensheng Fan
"""
import torch
import torch.nn as nn


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    activation layer
    :param act_type: 'relu'/'lrelu'/'prelu'
    :param inplace: True or False
    :param neg_slope: for leakyReLU
    :param n_prelu: number of parameters for prelu
    :return: activation layer
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer
