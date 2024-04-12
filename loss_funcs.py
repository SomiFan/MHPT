import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io
import numpy as np
from skimage import io
from utils import sobelfilter2d, gaussblur_fsize, gaussblur_fsigma, normalize_torch, qnr


class SpcSimilarityLoss(nn.Module):
    """
    implementation of spectral loss in paper: NLRNet: An Efficient Nonlocal Attention ResNet for Pansharpening
    pass hyperparameter in __init__
    pass fused img and target img in forward
    """

    def __init__(self):
        super(SpcSimilarityLoss, self).__init__()
        return

    def forward(self, fused: torch.Tensor, target: torch.Tensor):
        sim_mat = torch.cosine_similarity(fused, target, dim=1)
        return 1 - torch.mean(sim_mat)


class BandRelationLoss(nn.Module):
    """
    implementation of band loss in paper: NLRNet: An Efficient Nonlocal Attention ResNet for Pansharpening
    """

    def __init__(self):
        super(BandRelationLoss, self).__init__()
        return

    def forward(self, fused: torch.Tensor, target: torch.Tensor):
        s = fused.shape
        band_loss = 0.0
        for b in range(s[1] - 1):
            band_loss += torch.norm(
                ((fused[:, b + 1, :, :] - fused[:, b, :, :]) - (target[:, b + 1, :, :] - target[:, b, :, :])), p=1)
        band_loss += torch.norm(((fused[:, 0, :, :] - fused[:, -1, :, :]) - (target[:, 0, :, :] - target[:, -1, :, :])),
                                p=1)
        band_loss /= (s[0] * s[1])
        return band_loss


class SiSLoss(nn.Module):
    """
    Implementation of shift-invariant spectral (SiS) losses
    in CVPR 2021-SIPSA-Net: Shift-Invariant Pan Sharpening
    with Moving Object Alignment for Satellite Imagery
    Args:
        s_stride: length of shift stride
        ns: number of shift strides, default (-4, 4)
    """

    def __init__(self, s_stride=4, ns=4):
        super(SiSLoss, self).__init__()
        self.s_stride = s_stride
        self.ns = ns
        return

    def forward(self, ms_img: torch.Tensor, target: torch.Tensor):
        """
        :param ms_img: fused ms image or aligned ms image
        :param target: original ms image or up-sampled ms image
        :return: SiSloss value
        """
        N, C, H, W = target.shape
        shift_range = self.ns * 2 + 1
        target_pad = F.pad(target, pad=[self.ns*self.s_stride, self.ns*self.s_stride, self.ns*self.s_stride, self.ns*self.s_stride], mode='reflect')
        target = F.unfold(target_pad, (shift_range, shift_range), dilation=self.s_stride, padding=0, stride=1).view(N, C, -1, H, W)
        # assert target.shape[1] == C * (shift_range ** 2), "shifted target has wrong size"
        ms_img = torch.repeat_interleave(ms_img, shift_range ** 2, dim=1).view(N, C, -1, H, W)
        # return torch.mean(torch.min(torch.abs(ms_img - target), dim=2)[0])
        # min_sloss = torch.min(torch.sum(torch.abs(ms_img - target), dim=1, keepdim=True), dim=2)[0]
        # return torch.mean(min_sloss)
        return torch.mean(torch.min(torch.sum(torch.abs(ms_img - target), dim=1), dim=1)[0])


class EdgeLoss(nn.Module):
    """
    Implementation of edge detail losses
    in CVPR 2021-SIPSA-Net: Shift-Invariant Pan Sharpening
    with Moving Object Alignment for Satellite Imagery
    Args:
        s_stride: length of shift stride
        ns: number of shift strides, default (-4, 4)
    """

    def __init__(self):
        super(EdgeLoss, self).__init__()
        return

    def forward(self, ms_img: torch.Tensor, pan: torch.Tensor):
        """
        :param ms_img: fused ms image or aligned ms image
        :param pan: pan image or down-sampled pan
        :return: SiSloss value
        """
        pan_g2d = torch.abs(sobelfilter2d(pan))
        ms_gray = torch.mean(ms_img, 1, True)  # gray map
        ms_gray_g2d = torch.abs(sobelfilter2d(ms_gray))
        return F.l1_loss(ms_gray_g2d, pan_g2d, reduction='mean')


class MMLwFLoss(nn.Module):
    """
    Misalignment Masked Learning without Forgetting Loss
    Args:
    """

    def __init__(self):
        super(MMLwFLoss, self).__init__()
        return

    def forward(self, ms_img: torch.Tensor, ms_img_o: torch.Tensor, fr_pan: torch.Tensor):
        """
        :param ms_img: image from fine-tuned model
        :param ms_img_o: image from original pre-trained model
        :param fr_pan: full-resolution pan image
        :return: MMLwFLoss value
        """
        N, C, H, W = ms_img.shape
        ms_o_gray = torch.mean(ms_img_o, 1, True)
        i_f_sfed = torch.abs(sobelfilter2d(ms_o_gray))
        opan_sfed = torch.abs(sobelfilter2d(fr_pan))
        mask_gray = torch.abs(i_f_sfed - opan_sfed)
        mask_gray = normalize_torch(torch.sum(mask_gray, dim=1, keepdim=True)) * 255.0
        # mask_gray = normalize_torch(torch.abs(ms_o_gray - fr_pan)) * 255.0
        mask_bands = mask_gray.repeat(1, C, 1, 1)
        masked_ms = ms_img[mask_bands < 20]
        masked_ms_o = ms_img_o[mask_bands < 20]
        if len(masked_ms) == 0:
            return 0
        return F.l1_loss(masked_ms, masked_ms_o, reduction='mean')


class MMSiSLoss(nn.Module):
    """
    Misalignment Masked Shift-invariant Spectral (SiS) Loss
    Args:
        s_stride: length of shift stride
        ns: number of shift strides, default (-4, 4)
    """

    def __init__(self, s_stride=4, ns=4):
        super(MMSiSLoss, self).__init__()
        self.s_stride = s_stride
        self.ns = ns
        return

    def forward(self, ms_img: torch.Tensor, target: torch.Tensor, ms_img_o: torch.Tensor, fr_pan: torch.Tensor):
        """
        :param ms_img: fused ms image or aligned ms image
        :param target: original ms image or up-sampled ms image
        :param ms_img_o: image from original pre-trained model
        :param fr_pan: full-resolution pan image
        :return: SiSloss value
        """
        N, C, H, W = target.shape
        shift_range = self.ns * 2 + 1
        target_pad = F.pad(target, pad=[self.ns*self.s_stride, self.ns*self.s_stride, self.ns*self.s_stride, self.ns*self.s_stride], mode='reflect')
        target = F.unfold(target_pad, (shift_range, shift_range), dilation=self.s_stride, padding=0, stride=1).view(N, C, -1, H, W)
        # assert target.shape[1] == C * (shift_range ** 2), "shifted target has wrong size"
        ms_img = torch.repeat_interleave(ms_img, shift_range ** 2, dim=1).view(N, C, -1, H, W)
        # return torch.mean(torch.min(torch.abs(ms_img - target), dim=2)[0])
        min_sloss = torch.min(torch.sum(torch.abs(ms_img - target), dim=1), dim=1)[0]
        ms_o_gray = torch.mean(ms_img_o, 1)
        mask_gray = normalize_torch(torch.abs(ms_o_gray - torch.squeeze(fr_pan, 1))) * 255.0
        masked_sl = min_sloss[mask_gray > 10]
        if len(masked_sl) == 0:
            return 0
        return torch.mean(masked_sl)


class QNRLoss(nn.Module):
    """
    Loss function based on the QNR metric.
    Args:
    """

    def __init__(self):
        super(QNRLoss, self).__init__()
        return

    def forward(self, ps: torch.Tensor, pan: torch.Tensor, lrms: torch.Tensor):
        """
        :param ps: pan-sharpened ms image
        :param pan: original pan image
        :param lrms: original low resolution ms image
        :return: QNRLoss value
        """
        return qnr(ps, pan, lrms, isloss=True)


if __name__ == "__main__":
    """
    imdata = scipy.io.loadmat(r"F:\ResearchData\dataset\QBSX\fr_full_img.mat")
    m_patch = imdata['oms_patch'].astype(np.float32).transpose((2, 0, 1))
    inp = torch.from_numpy(m_patch).float().unsqueeze(0).cuda()
    # orgimg = np.transpose(inp.data.numpy()[0], (1, 2, 0))

    # io.imsave(f'./org.tif', orgimg[:, :, 0:3].astype(np.uint8))
    """
    """
    ms_gray = torch.mean(inp, 1, True)
    sfed = torch.abs(sobelfilter2d(ms_gray))
    sfedx = sfed[0, 0, :, :].cpu().numpy()
    sfedy = sfed[0, 1, :, :].cpu().numpy()
    io.imsave(f'./sfedx.tif', sfedx.astype(np.uint8))
    io.imsave(f'./sfedy.tif', sfedy.astype(np.uint8))
    sfed = torch.sum(sfed, dim=1)
    """
    """
    # sfed = gaussblur_fsigma(inp, 4)
    sfed = gaussblur_fsize(inp, 5)
    sfed = sfed.cpu().data.numpy()[0]

    io.imsave(f'./sfed2.tif', sfed.astype(np.uint8))
    """
    imdata_fused = scipy.io.loadmat(r"D:\Document\experiments\DL_Pansharp\SIPSA_QBFR\SIPSA_QBFR_FR_QB12_26.mat")
    i_f = imdata_fused['I_F'].astype(np.float32).transpose((2, 0, 1))
    i_f = torch.from_numpy(i_f).float().unsqueeze(0).cuda()
    imdata_myres = scipy.io.loadmat(r"D:\Document\experiments\DL_Pansharp\SWPF_T128_4_QB\SWPF_T128_4_QB_FR_QB12_594.mat")
    i_f_my = imdata_myres['I_F'].astype(np.float32).transpose((2, 0, 1))
    i_f_my = torch.from_numpy(i_f_my).float().unsqueeze(0).cuda()
    m_patch = io.imread(r"F:\ResearchData\dataset\QB\forpresentation\oms\oms_patch12.tif")
    p_patch = io.imread(r"F:\ResearchData\dataset\QB\forpresentation\opan\opan_patch12.tif")
    m_patch = torch.from_numpy(m_patch.astype(np.float32).transpose((2, 0, 1))).float().unsqueeze(0).cuda()
    # m_norm = np.array([scale_range(i, -1, 1) for i in m_patch])
    p_patch = torch.from_numpy(p_patch.astype(np.float32)).float().unsqueeze(0).unsqueeze(0).cuda()
    # p_norm = np.array(scale_range(p_patch, -1, 1))
    ms_up = F.interpolate(m_patch, scale_factor=4, mode='nearest')
    sis = SiSLoss(ns=4)
    el = EdgeLoss()
    print(f"sipsa-net SiSLoss: {sis(i_f, ms_up)}")
    print(f"swpf SiSLoss: {sis(i_f_my, ms_up)}")
    print(f"sipsa-net EdgeLoss: {el(i_f, p_patch)}")
    print(f"swpf EdgeLoss: {el(i_f_my, p_patch)}")

