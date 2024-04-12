import numpy as np
import math
import scipy.io
from skimage.transform import resize
from scipy.ndimage import sobel
from numpy.linalg import norm
from utils.util import pad_sr
import torch
import torch.nn.functional as F

def ERGAS(result, target, ratio=4):
    result = np.double(result)
    target = np.double(target)
    if result.shape != target.shape:
        raise ValueError('result and target arrays must have the same shape!')

    Err = result - target
    ERGAS_index = 0
    for i in range(len(Err[0][0])):
        # sum1=0
        sum1 = sum(sum(Err[:, :, i] ** 2))

        mean1 = sum1 / (len(Err) * len(Err[0]))
        # sum2=0
        I1_2 = result[:, :, i]
        sum2 = sum(sum(I1_2))

        mean2 = sum2 / (len(Err) * len(Err[0]))
        ERGAS_index += mean1 / (mean2 ** 2)
    ERGAS_index = (100 / ratio) * np.sqrt((1 / len(Err[0][0])) * ERGAS_index)

    return ERGAS_index


def sam(I1_ma, I2_ma):
    # I2_ma=I2['MSHR']
    I2_ma = I2_ma.astype(np.double)
    # I1_ma=I1['MSWV_db']
    I1_ma = I1_ma.astype(np.double)
    M = I1_ma.shape[0]
    N = I1_ma.shape[1]
    prod_scal = I1_ma[:, :, 0] * I2_ma[:, :, 0]
    prod_scal += I1_ma[:, :, 1] * I2_ma[:, :, 1]
    prod_scal += I1_ma[:, :, 2] * I2_ma[:, :, 2]  # I1,I2沿第3维的点积

    norm_orig = I1_ma[:, :, 0] * I1_ma[:, :, 0]
    norm_orig += I1_ma[:, :, 1] * I1_ma[:, :, 1]
    norm_orig += I1_ma[:, :, 2] * I1_ma[:, :, 2]

    norm_fusa = I2_ma[:, :, 0] * I2_ma[:, :, 0]
    norm_fusa += I2_ma[:, :, 1] * I2_ma[:, :, 1]
    norm_fusa += I2_ma[:, :, 2] * I2_ma[:, :, 2]
    # print(prod_scal)
    prod_norm = norm_orig * norm_fusa
    prod_norm = np.sqrt(prod_norm)
    prod_norm[prod_norm == 0] = 1e-9
    # for i in range(len(prod_norm)):
    #    for j in range(len(prod_norm[0])):
    #        prod_norm[i][j]=math.sqrt(prod_norm[i][j])
    prod_map = prod_norm
    # sue=0
    # for i in range(len(prod_map)):
    #    for j in range(len(prod_map[0])):
    #        if prod_map[i][j]==0:
    #            prod_map[i][j]=eps

    # SAM_map=prod_scal*np.linalg.inv(prod_map)
    SAM_map = prod_scal / prod_map
    for i in range(len(SAM_map)):
        for j in range(len(SAM_map[0])):
            SAM_map[i][j] = math.acos(SAM_map[i][j])

            # sue+=1
    # print(sue)
    prod_scal = np.reshape(prod_scal, (M * N, 1), order="F")
    # prod_scal2=np.zeros(shape=(M*N,1))
    # for i in range(len(prod_scal)):
    #    for j in range(len(prod_scal[0])):
    #        prod_scal2[i*N+j]=prod_scal[i][j]
    prod_norm = np.reshape(prod_norm, (M * N, 1), order="F")
    # prod_norm2=np.zeros(shape=(M*N,1))
    # for i in range(len(prod_norm)):
    #    for j in range(len(prod_norm[0])):
    #        prod_norm2[i*N+j]=prod_norm[i][j]

    for i in range(len(prod_norm)):
        if prod_norm[i] == 0:
            prod_norm[i] = []
            prod_scal[i] = []

    temp = prod_scal / prod_norm
    for i in range(len(temp)):
        for j in range(len(temp[0])):
            temp[i][j] = math.acos(temp[i][j])
    # angolo=0
    angolo = sum(sum(temp))
    # for i in range(len(temp)):
    #    for j in range(len(temp[0])):
    #        angolo+=temp[i][j]
    angolo = angolo / len(prod_norm)
    SAM_index = angolo.real * 180 / math.pi
    return SAM_index

# Q index metric
def qindex(y, x, N=32, eps=1e-16, keepdims0=True):
    """Args: y: 4D network outputs, 4D x: target images"""
    B, C, H, W = y.shape
    y = pad_sr(y.transpose(0, 1).reshape(-1, H, W).unsqueeze(1), N)
    x = pad_sr(x.transpose(0, 1).reshape(-1, H, W).unsqueeze(1), N)
    y = F.pixel_unshuffle(y, N)
    x = F.pixel_unshuffle(x, N)
    ym = torch.mean(y, 1, True)
    xm = torch.mean(x, 1, True)
    y = y - ym
    x = x - xm
    syx = torch.mean(x*y, 1, True)
    sxx = torch.mean(x*x, 1, True)
    syy = torch.mean(y*y, 1, True)
    q = 4*syx*xm*ym / ((sxx+syy)*(xm*xm+ym*ym) + eps)
    _, _, H, W = y.shape
    q = torch.mean(q.reshape(C, B, 1, H, W).transpose(0, 1).reshape(B, -1, H, W), (2,3), True)
    if not keepdims0:
        q = torch.mean(q)
    return q

# QNR subfunction metric - D_lambda
def qnrL(y, x):  # y=PS, x=LRMS
    """Args: y: 4D network output, x: 4D low-res MS input"""
    B, C, H, W = y.shape
    qy = qindex(y.repeat(1, C, 1, 1), torch.repeat_interleave(y, C, dim=1))
    qx = qindex(x.repeat(1, C, 1, 1), torch.repeat_interleave(x, C, dim=1))
    q = torch.abs(qy-qx)
    q = torch.mean(torch.sum(q, 1, keepdim=True)/(C*(C-1)))
    return q

# QNR subfunction metric - D_s
def qnrS(ps, pan, ms, sc=4):  # ms=LRMS
    """Args: ps: 4D network output, pan: 4D PAN input, ms: 4D low-res MS input, sc: scale ratio"""
    B, C, H, W = ps.shape
    pw = F.interpolate(pan, scale_factor=1/sc, mode='bicubic', align_corners=False, recompute_scale_factor=False)
    qy = qindex(ps, pan.repeat(1, C, 1, 1))
    qx = qindex(ms, pw.repeat(1, C, 1, 1))
    q = torch.abs(qy-qx)
    q = torch.mean(q)
    return q

def QNR(ps, pan, ms, sc=4, isloss=False, getAll=False):
    """QNR metric.

    Args: ps: 4D network output,
    pan: 4D PAN input,
    ms: 4D low-resolution MS input,
    sc: scale ratio,
    isloss: can be used as a loss,
    getAll: outputs 3 metric scores
    """
    DL = qnrL(ps, ms)
    DS = qnrS(ps, pan, ms)
    if isloss:
        q = DL + DS
    else:
        q = torch.abs(1.-DL)*torch.abs(1.-DS)
    if getAll:
        q = torch.stack([q, DL, DS])
    return q

if __name__ == "__main__":
    imdata = scipy.io.loadmat(r"F:\ResearchData\dataset\QBSX\fr_full_img.mat")
    p_patch = imdata['opan_patch'].astype(np.float32)
    m_patch = imdata['oms_patch'].astype(np.float32).transpose((2, 0, 1))
    C, H, W = m_patch.shape
    ms_up = np.array([resize(i, (H * 4, W * 4), 3) for i in m_patch])
    oms = torch.from_numpy(m_patch).float().unsqueeze(0)
    out = torch.from_numpy(ms_up).float().unsqueeze(0)
    opan = torch.from_numpy(p_patch).float().unsqueeze(0).unsqueeze(0)
    print(QNR(out, opan, oms, getAll=True)[0])

