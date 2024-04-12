"""
panformer.py 2023/11/22 16:45
Written by Wensheng Fan
"""

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from .modules import InvertibleConv1x1
# from .refine import Refine1
import torch.nn.init as init
import math
from functools import reduce
# from model.base_net import *
import scipy


def thops_sum(tensor, dim=None, keepdim=False):
    if dim is None:
        # sum up all dim
        return torch.sum(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.sum(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor


def thops_pixels(tensor):
    return int(tensor.size(2) * tensor.size(3))


class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
        self.process = nn.Sequential(
                nn.Conv2d(channel, channel, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        )

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x


class Refine1(nn.Module):

    def __init__(self,in_channels,panchannels,n_feat):
        super(Refine1, self).__init__()

        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(
             # CALayer(n_feat,4),
             # CALayer(n_feat,4),
             CALayer(n_feat,4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=in_channels-panchannels, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=False):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            pixels = thops_pixels(input)
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float()\
                              .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = thops_sum(self.log_s) * thops_pixels(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z, logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z, logdet

# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################
class Transformer_Fusion(nn.Module):
    def __init__(self,nc):
        super(Transformer_Fusion, self).__init__()
        self.conv_trans = nn.Sequential(
            nn.Conv2d(2*nc,nc,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(nc,nc,kernel_size=3,stride=1,padding=1))

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        # list
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]  # B 1 -1
        expanse = list(input.size())  # B C*9 L
        expanse[0] = -1
        # [-1,...,-1,...]
        expanse[dim] = -1  # -1 C*9 -1
        index = index.view(views).expand(expanse)  # B HW  B 1 HW  B C*9 HW
        return torch.gather(input, dim, index)

    def forward(self, lrsr_lv3, ref_lv3):
        ######################   search
        # 按照选定的尺寸与步长来切分矩阵
        # unfold函数的参数为（dim,size,step）,dim代表想要切分的维度，size代表切分块的尺寸，step代表切分的步长。
        lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
        # 三维是permute(0, 1, 2)
        # 0代表共有几块维度：本例中0对应着3块矩阵
        # 1代表每一块中有多少行：本例中1对应着每块有2行
        # 2代表每一块中有多少列：本例中2对应着每块有5列
        # 在不改变每一块（即）的前提下，对每一块的行列进行对调（即二维矩阵的转置）
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)

        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1) # [N, C*k*k, H*W]

        # 矩阵乘法
        R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold) #[N, Hr*Wr, H*W]
        # 按维度dim 返回最大值
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1) #[N, H*W]

        ### transfer
        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)

        # 硬注意力
        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)

        # 将提取出的滑动局部区域块还原成batch的张量形式
        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3, 3), padding=1) / (3.*3.)

        # 软
        S = R_lv3_star.view(R_lv3_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))

        res = self.conv_trans(torch.cat([T_lv3,lrsr_lv3],1))*S+lrsr_lv3

        return res


# 远程特征提取
class PatchFusion(nn.Module):
    def __init__(self,nc):
        super(PatchFusion, self).__init__()
        self.fuse = Transformer_Fusion(nc)

    def forward(self,msf,panf):
        # msf, panf = x[:, :self.ms_chans, :, :], x[:, self.ms_chans, :, :].unsqueeze(1)
        ori = msf
        b,c,h,w = ori.size()
        msf = F.unfold(msf,kernel_size=(24, 24), stride=8, padding=8)
        panf = F.unfold(panf, kernel_size=(24, 24), stride=8, padding=8)
        msf = msf.view(-1,c,24,24)
        panf = panf.view(-1,c,24,24)
        fusef = self.fuse(msf,panf)
        fusef = fusef.view(b,c*24*24,-1)
        fusef = F.fold(fusef, output_size=ori.size()[-2:], kernel_size=(24, 24), stride=8, padding=8)
        return fusef

#########################################################################################



def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope=0.1, use_HIN=True):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

    def forward(self, x):
        out = self.conv_1(x)
        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=16, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc)
        self.conv2 = UNetConvBlock(gc, channel_out)
        # self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1,self.conv2], 0.1)
        else:
            initialize_weights([self.conv1,self.conv2], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        # x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x2


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
            # return UNetBlock(channel_in, channel_out)
        else:
            return None

    return constructor


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out


class PanFormer(nn.Module):
    def __init__(self, ms_chans=4, nc=8,**kwargs):
        super(PanFormer, self).__init__()
        self.ms_chans = ms_chans
        #self.img_size = img_size
        #self.patch_size = cfg["PSIZE"]

        self.preconvms = nn.Conv2d(ms_chans, nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.preconvpan = nn.Conv2d(1, nc, kernel_size=3, stride=1, padding=1, bias=False)

        # local feature branch
        self.local_branch = nn.Sequential(
            nn.Conv2d(nc*2, nc*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(nc*2, nc*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(nc*2, nc, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.lrfb = PatchFusion(nc)
        self.innff1 = InvBlock(subnet('DBNet'), nc*2, nc)
        self.innff2 = InvBlock(subnet('DBNet'), nc * 2, nc)
        self.innff3= InvBlock(subnet('DBNet'), nc * 2, nc)
        self.postconv = Refine1(in_channels=nc*8, panchannels=nc*8-ms_chans, n_feat=nc*8)

        # modality aware feature extraction convs
        #N = 8 * (img_size // self.patch_size) * (img_size // self.patch_size)
        #self.mfeq = nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1, bias=False)
        #self.mfek = nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1, bias=False)
        #self.mfev1 = nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1, bias=False)
        #self.mfev2 = nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1, bias=False)

        # long-range feature branch
        #self.lrconv1 = nn.Conv2d(10 * 8, 8, kernel_size=3, stride=1, padding=1, bias=False)
        #self.lrconv2 = nn.Conv2d(10 * 8, 8, kernel_size=3, stride=1, padding=1, bias=False)
        #self.lrconv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)


    def forward(self, x):
        input_pan, input_ms = x[:, self.ms_chans, :, :].unsqueeze(1), x[:, :self.ms_chans, :, :]
        m0 = self.preconvms(input_ms)
        p0 = self.preconvpan(input_pan)

        mp0 = torch.cat((m0, p0), dim=1)

        # 局部特征提取
        l0 = self.local_branch(mp0)

        # 远程特征提取
        g0 = self.lrfb(m0, p0)
        l1g1 = self.innff1(torch.cat((l0, g0), 1))
        l2g2 = self.innff2(l1g1)
        l3g3 = self.innff3(l2g2)

        out = input_ms + self.postconv(torch.cat((torch.cat((l0, g0), 1), l1g1, l2g2, l3g3),1))
        return out


