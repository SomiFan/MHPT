from torch.utils.data import Dataset
import numpy as np
import os
import glob
import scipy.io
from skimage import io, filters
from skimage.transform import resize
import cv2 as cv
from PIL import Image, ImageFilter
from utils.util import scale_range, bgr2rgb, get_edge


class QBDataset(Dataset):
    def __init__(self, cfg, mode='train', data_dir=r"F:\ResearchData\dataset_light\QB/"):
        self.cfg = cfg
        self.mode = mode
        self.data_dir = data_dir
        self.n_train_data = len(glob.glob(self.data_dir + 'lms/*.tif'))

    def __getitem__(self, index):
        if self.mode == "train":
            dscl_ms_fn = 'lms/lms_patch' + str(index) + '.tif'
            dscl_pan_fn = 'lpan/lpan_patch' + str(index) + '.tif'
            org_ms_fn = 'oms/oms_patch' + str(index) + '.tif'
        elif self.mode == "val":
            t_index = index + self.n_train_data
            dscl_ms_fn = 'test/lms/lms_patch' + str(t_index) + '.tif'
            dscl_pan_fn = 'test/lpan/lpan_patch' + str(t_index) + '.tif'
            org_ms_fn = 'test/oms/oms_patch' + str(t_index) + '.tif'

        mp_path = os.path.join(self.data_dir, dscl_ms_fn)
        lp_path = os.path.join(self.data_dir, org_ms_fn)
        m_patch = io.imread(mp_path)
        l_patch = io.imread(lp_path)
        m_patch = m_patch.astype(np.float32).transpose((2, 0, 1))
        l_patch = l_patch.astype(np.float32).transpose((2, 0, 1))
        # m_norm = np.array([scale_range(i, -1, 1) for i in m_patch])
        # l_norm = np.array([scale_range(i, -1, 1) for i in l_patch])
        pp_path = os.path.join(self.data_dir, dscl_pan_fn)
        p_patch = io.imread(pp_path)
        p_patch = p_patch.astype(np.float32)
        # p_norm = np.array(scale_range(p_patch, -1, 1))
        if self.cfg.MODEL.ORIGINAL_MS:
            return (m_patch, p_patch), l_patch
        ms_up = [resize(i, (256, 256), 3) for i in m_patch]
        # ms_up = np.clip(ms_up, -1.0, 1.0)

        inp = np.concatenate((ms_up, np.expand_dims(p_patch, axis=0)), axis=0)
        out = l_patch

        return inp.astype(np.float32), out, m_patch

    def __len__(self):
        if self.mode == 'train':
            return self.n_train_data
        elif self.mode == "val":
            return len(glob.glob(self.data_dir + 'test/lms/*.tif'))


class GF2Dataset(Dataset):
    def __init__(self, cfg, mode='train', data_dir=r"F:\ResearchData\dataset\GF2/"):
        self.cfg = cfg
        self.mode = mode
        self.data_dir = data_dir
        self.n_train_data = len(glob.glob(self.data_dir + 'lms/*.tif'))

    def __getitem__(self, index):
        if self.mode == "train":
            dscl_ms_fn = 'lms/lms_patch' + str(index) + '.tif'
            dscl_pan_fn = 'lpan/lpan_patch' + str(index) + '.tif'
            org_ms_fn = 'oms/oms_patch' + str(index) + '.tif'
        elif self.mode == "val":
            t_index = index + self.n_train_data
            dscl_ms_fn = 'test/lms/lms_patch' + str(t_index) + '.tif'
            dscl_pan_fn = 'test/lpan/lpan_patch' + str(t_index) + '.tif'
            org_ms_fn = 'test/oms/oms_patch' + str(t_index) + '.tif'

        mp_path = os.path.join(self.data_dir, dscl_ms_fn)
        lp_path = os.path.join(self.data_dir, org_ms_fn)
        m_patch = io.imread(mp_path)
        l_patch = io.imread(lp_path)
        m_patch = m_patch.astype(np.float32).transpose((2, 0, 1))
        l_patch = l_patch.astype(np.float32).transpose((2, 0, 1))
        # m_norm = np.array([scale_range(i, -1, 1) for i in m_patch])
        # l_norm = np.array([scale_range(i, -1, 1) for i in l_patch])
        pp_path = os.path.join(self.data_dir, dscl_pan_fn)
        p_patch = io.imread(pp_path)
        p_patch = p_patch.astype(np.float32)
        # p_norm = np.array(scale_range(p_patch, -1, 1))
        if self.cfg.MODEL.ORIGINAL_MS:
            return (m_patch, p_patch), l_patch
        ms_up = [resize(i, (256, 256), 3) for i in m_patch]
        # ms_up = np.clip(ms_up, -1.0, 1.0)

        inp = np.concatenate((ms_up, np.expand_dims(p_patch, axis=0)), axis=0)
        out = l_patch

        return inp.astype(np.float32), out, index

    def __len__(self):
        if self.mode == 'train':
            return self.n_train_data
        elif self.mode == "val":
            return len(glob.glob(self.data_dir + 'test/lms/*.tif'))


class WV3Dataset(Dataset):
    def __init__(self, cfg, mode='train', data_dir=r"F:\ResearchData\dataset\WV3/"):
        self.cfg = cfg
        self.mode = mode
        self.data_dir = data_dir
        self.n_train_data = len(glob.glob(self.data_dir + 'rr/*.mat'))

    def __getitem__(self, index):
        if self.mode == "train":
            data_fn = 'rr/rr' + str(index) + '.mat'
        elif self.mode == "val":
            t_index = index + self.n_train_data
            data_fn = 'test/rr/rr' + str(t_index) + '.mat'
        data_path = os.path.join(self.data_dir, data_fn)
        imdata = scipy.io.loadmat(data_path)
        l_patch = imdata['oms_patch'].astype(np.float32)
        m_patch = imdata['lms_patch'].astype(np.float32)
        p_patch = imdata['lpan_patch'].astype(np.float32)
        m_patch = m_patch.transpose((2, 0, 1))
        l_patch = l_patch.transpose((2, 0, 1))
        # m_norm = np.array([scale_range(i, -1, 1) for i in m_patch])
        # l_norm = np.array([scale_range(i, -1, 1) for i in l_patch])
        # p_norm = np.array(scale_range(p_patch, -1, 1))
        if self.cfg.MODEL.ORIGINAL_MS:
            return (m_patch, p_patch), l_patch
        ms_up = [resize(i, (256, 256), 3) for i in m_patch]
        # ms_up = np.clip(ms_up, -1.0, 1.0)
        inp = np.concatenate((ms_up, np.expand_dims(p_patch, axis=0)), axis=0)
        out = l_patch
        return inp.astype(np.float32), out, index

    def __len__(self):
        if self.mode == 'train':
            return self.n_train_data
        elif self.mode == "val":
            return len(glob.glob(self.data_dir + 'test/rr/*.mat'))


class GF2LDataset(Dataset):
    def __init__(self, cfg, mode='train', data_dir=r"F:\ResearchData\dataset\GF2L/"):
        self.cfg = cfg
        self.mode = mode
        self.data_dir = data_dir
        self.n_train_data = len(glob.glob(self.data_dir + 'lms/*.tif'))

    def __getitem__(self, index):
        if self.mode == "train":
            dscl_ms_fn = 'lms/lms_patch' + str(index) + '.tif'
            dscl_pan_fn = 'lpan/lpan_patch' + str(index) + '.tif'
            org_ms_fn = 'oms/oms_patch' + str(index) + '.tif'
        elif self.mode == "val":
            t_index = index + self.n_train_data
            dscl_ms_fn = 'test/lms/lms_patch' + str(t_index) + '.tif'
            dscl_pan_fn = 'test/lpan/lpan_patch' + str(t_index) + '.tif'
            org_ms_fn = 'test/oms/oms_patch' + str(t_index) + '.tif'

        mp_path = os.path.join(self.data_dir, dscl_ms_fn)
        lp_path = os.path.join(self.data_dir, org_ms_fn)
        m_patch = io.imread(mp_path)
        l_patch = io.imread(lp_path)
        m_patch = m_patch.astype(np.float32).transpose((2, 0, 1))
        l_patch = l_patch.astype(np.float32).transpose((2, 0, 1))
        # m_norm = np.array([scale_range(i, -1, 1) for i in m_patch])
        # l_norm = np.array([scale_range(i, -1, 1) for i in l_patch])
        pp_path = os.path.join(self.data_dir, dscl_pan_fn)
        p_patch = io.imread(pp_path)
        p_patch = p_patch.astype(np.float32)
        # p_norm = np.array(scale_range(p_patch, -1, 1))
        if self.cfg.MODEL.ORIGINAL_MS:
            return (m_patch, p_patch), l_patch
        ms_up = [resize(i, (256, 256), 3) for i in m_patch]
        # ms_up = np.clip(ms_up, -1.0, 1.0)

        inp = np.concatenate((ms_up, np.expand_dims(p_patch, axis=0)), axis=0)
        out = l_patch

        return inp.astype(np.float32), out, index

    def __len__(self):
        if self.mode == 'train':
            return self.n_train_data
        elif self.mode == "val":
            return len(glob.glob(self.data_dir + 'test/lms/*.tif'))


class SPOTDataset(Dataset):
    def __init__(self, cfg, mode='train', data_dir=r"F:\ResearchData\dataset\SPOT2/"):
        self.cfg = cfg
        self.mode = mode
        self.data_dir = data_dir
        self.n_train_data = len(glob.glob(self.data_dir + 'rr/*.mat'))

    def __getitem__(self, index):
        if self.mode == "train":
            data_fn = 'rr/rr' + str(index) + '.mat'
        elif self.mode == "val":
            t_index = index + self.n_train_data
            data_fn = 'test/rr/rr' + str(t_index) + '.mat'
        data_path = os.path.join(self.data_dir, data_fn)
        imdata = scipy.io.loadmat(data_path)
        l_patch = imdata['oms_patch'].astype(np.float32)
        m_patch = imdata['lms_patch'].astype(np.float32)
        p_patch = imdata['lpan_patch'].astype(np.float32)
        m_patch = m_patch.transpose((2, 0, 1))
        l_patch = l_patch.transpose((2, 0, 1))
        # m_norm = np.array([scale_range(i, -1, 1) for i in m_patch])
        # l_norm = np.array([scale_range(i, -1, 1) for i in l_patch])
        # p_norm = np.array(scale_range(p_patch, -1, 1))
        if self.cfg.MODEL.ORIGINAL_MS:
            return (m_patch, p_patch), l_patch
        ms_up = [resize(i, (64, 64), 3) for i in m_patch]
        # io.imsave(f"./upms{index}.tif",np.array(ms_up).transpose((1, 2, 0)).astype(np.uint8))
        # io.imsave(f"./oms{index}.tif", np.array(l_patch).transpose((1, 2, 0)).astype(np.uint8))
        # ms_up = np.clip(ms_up, -1.0, 1.0)
        inp = np.concatenate((ms_up, np.expand_dims(p_patch, axis=0)), axis=0)
        out = l_patch
        return inp.astype(np.float32), out, index

    def __len__(self):
        if self.mode == 'train':
            return self.n_train_data
        elif self.mode == "val":
            return len(glob.glob(self.data_dir + 'test/rr/*.mat'))


class QBSXDataset(Dataset):
    def __init__(self, cfg, mode='train', data_dir=r"F:\ResearchData\dataset\QBSX/"):
        self.cfg = cfg
        self.mode = mode
        self.data_dir = data_dir
        self.n_train_data = len(glob.glob(self.data_dir + 'rr/*.mat'))

    def __getitem__(self, index):
        if self.mode == "train":
            data_fn = 'rr/rr' + str(index) + '.mat'
        elif self.mode == "val":
            t_index = index + self.n_train_data
            data_fn = 'test/rr/rr' + str(t_index) + '.mat'
        data_path = os.path.join(self.data_dir, data_fn)
        imdata = scipy.io.loadmat(data_path)
        l_patch = imdata['oms_patch'].astype(np.float32)
        m_patch = imdata['lms_patch'].astype(np.float32)
        p_patch = imdata['lpan_patch'].astype(np.float32)
        m_patch = m_patch.transpose((2, 0, 1))
        l_patch = l_patch.transpose((2, 0, 1))
        # m_norm = np.array([scale_range(i, -1, 1) for i in m_patch])
        # l_norm = np.array([scale_range(i, -1, 1) for i in l_patch])
        # p_norm = np.array(scale_range(p_patch, -1, 1))
        if self.cfg.MODEL.ORIGINAL_MS:
            return (m_patch, p_patch), l_patch
        ms_up = [resize(i, (64, 64), 3) for i in m_patch]
        # ms_up = np.clip(ms_up, -1.0, 1.0)
        inp = np.concatenate((ms_up, np.expand_dims(p_patch, axis=0)), axis=0)
        out = l_patch
        return inp.astype(np.float32), out, index

    def __len__(self):
        if self.mode == 'train':
            return self.n_train_data
        elif self.mode == "val":
            return len(glob.glob(self.data_dir + 'test/rr/*.mat'))


class FRDataset(Dataset):
    def __init__(self, cfg, mode='train', data_dir=r"F:\ResearchData\dataset\QBFR128F/"):
        self.cfg = cfg
        self.mode = mode
        if cfg.DATA.TRAIN_SET_PATH:
            self.data_dir = cfg.DATA.TRAIN_SET_PATH
        else:
            self.data_dir = data_dir
        # self.n_train_data = len(glob.glob(self.data_dir + 'train/*.mat'))

    def __getitem__(self, index):
        if self.mode == "train":
            data_fn = 'train/fr' + str(index) + '.mat'
        elif self.mode == "val":
            #t_index = index + self.n_train_data
            data_fn = 'val/fr' + str(index) + '.mat'
        data_path = os.path.join(self.data_dir, data_fn)
        imdata = scipy.io.loadmat(data_path)
        m_patch = imdata['oms_patch'].astype(np.float32)
        p_patch = imdata['opan_patch'].astype(np.float32)
        m_patch = m_patch.transpose((2, 0, 1))
        # m_norm = np.array([scale_range(i, -1, 1) for i in m_patch])
        # l_norm = np.array([scale_range(i, -1, 1) for i in l_patch])
        # p_norm = np.array(scale_range(p_patch, -1, 1))
        if self.cfg.MODEL.ORIGINAL_MS:
            return (m_patch, p_patch), index
        ms_up = [resize(i, (self.cfg.MODEL.PAN_SIZE, self.cfg.MODEL.PAN_SIZE), 3) for i in m_patch]
        # ms_up = np.clip(ms_up, -1.0, 1.0)
        inp = np.concatenate((ms_up, np.expand_dims(p_patch, axis=0)), axis=0)
        return inp.astype(np.float32), index, m_patch

    def __len__(self):
        if self.mode == 'train':
            return len(glob.glob(self.data_dir + 'train/*.mat'))
        elif self.mode == "val":
            return len(glob.glob(self.data_dir + 'val/*.mat'))


class RRDataset(Dataset):
    def __init__(self, cfg, mode='train', data_dir=r"F:\ResearchData\dataset\QB128F/"):
        self.cfg = cfg
        self.mode = mode
        if cfg.DATA.TRAIN_SET_PATH:
            self.data_dir = cfg.DATA.TRAIN_SET_PATH
        else:
            self.data_dir = data_dir
        #self.n_train_data = len(glob.glob(self.data_dir + 'train/*.mat'))

    def __getitem__(self, index):
        if self.mode == "train":
            data_fn = 'train/rr' + str(index) + '.mat'
        elif self.mode == "val":
            #t_index = index + self.n_train_data
            data_fn = 'val/rr' + str(index) + '.mat'

        data_path = os.path.join(self.data_dir, data_fn)
        imdata = scipy.io.loadmat(data_path)
        l_patch = imdata['oms_patch'].astype(np.float32)
        m_patch = imdata['lms_patch'].astype(np.float32)
        p_patch = imdata['lpan_patch'].astype(np.float32)
        m_patch = m_patch.transpose((2, 0, 1))
        l_patch = l_patch.transpose((2, 0, 1))
        if self.cfg.MODEL.ORIGINAL_MS:
            return (m_patch, p_patch), l_patch
        ms_up = [resize(i, (self.cfg.MODEL.PAN_SIZE, self.cfg.MODEL.PAN_SIZE), 3) for i in m_patch]
        # ms_up = np.clip(ms_up, -1.0, 1.0)

        inp = np.concatenate((ms_up, np.expand_dims(p_patch, axis=0)), axis=0)
        out = l_patch

        return inp.astype(np.float32), out, m_patch

    def __len__(self):
        if self.mode == 'train':
            return len(glob.glob(self.data_dir + 'train/*.mat'))
        elif self.mode == "val":
            return len(glob.glob(self.data_dir + 'val/*.mat'))


class RRHPDataset(Dataset):
    def __init__(self, cfg, mode='train', data_dir=r"F:\ResearchData\dataset\QB128F/"):
        self.cfg = cfg
        self.mode = mode
        if cfg.DATA.TRAIN_SET_PATH:
            self.data_dir = cfg.DATA.TRAIN_SET_PATH
        else:
            self.data_dir = data_dir
        # self.n_train_data = len(glob.glob(self.data_dir + 'train/*.mat'))

    def __getitem__(self, index):
        if self.mode == "train":
            data_fn = 'train/rr' + str(index) + '.mat'
        elif self.mode == "val":
            # t_index = index + self.n_train_data
            data_fn = 'val/rr' + str(index) + '.mat'

        data_path = os.path.join(self.data_dir, data_fn)
        imdata = scipy.io.loadmat(data_path)
        l_patch = imdata['oms_patch'].astype(np.float32)
        m_patch = imdata['lms_patch'].astype(np.float32)
        p_patch = imdata['lpan_patch'].astype(np.float32)

        m_patch_hp = get_edge(m_patch)
        p_patch = get_edge(p_patch)

        m_patch_hp = m_patch_hp.transpose((2, 0, 1))
        l_patch = l_patch.transpose((2, 0, 1))
        m_patch = m_patch.transpose((2, 0, 1))
        ms_up = np.array([resize(i, (self.cfg.MODEL.PAN_SIZE, self.cfg.MODEL.PAN_SIZE), 3) for i in m_patch])

        return (m_patch_hp, p_patch, ms_up), l_patch

        # if self.cfg.MODEL.ORIGINAL_MS:
            # return (m_patch, p_patch), l_patch
        # ms_up = [resize(i, (self.cfg.MODEL.PAN_SIZE, self.cfg.MODEL.PAN_SIZE), 3) for i in m_patch]
        # ms_up = np.clip(ms_up, -1.0, 1.0)

        #inp = np.concatenate((ms_up, np.expand_dims(p_patch, axis=0)), axis=0)
        #out = l_patch

        #return inp.astype(np.float32), out, m_patch

    def __len__(self):
        if self.mode == 'train':
            return len(glob.glob(self.data_dir + 'train/*.mat'))
        elif self.mode == "val":
            return len(glob.glob(self.data_dir + 'val/*.mat'))


if __name__ == "__main__":
    imdata = scipy.io.loadmat(r"F:\newdatasets\QB128F\train\rr0.mat")
    l_patch = imdata['oms_patch'].astype(np.float32)
    m_patch = imdata['lms_patch'].astype(np.float32)
    p_patch = imdata['lpan_patch'].astype(np.float32)

    m_patch = get_edge(m_patch)
    p_patch = get_edge(p_patch)

    x = m_patch[:, :, 0:3]

    io.imsave("./box_filtered_ms1.jpg", m_patch[:, :, 0:3])
    io.imsave("./box_filtered_pan1.jpg", p_patch)