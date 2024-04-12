import os.path
import torch
import torch.nn.functional as F
import glob
import numpy as np
import scipy.io
# import cv2 as cv
from skimage import io
from skimage.transform import resize
from utils import scale_range, unnormalization, save_images, bgr2rgb, get_edge


@torch.no_grad()
def reduced_res_test(config, model, logger):
    model.eval()
    patch_size = config.MODEL.PAN_SIZE
    if config.DATA.DATASET == "wv3":
        data_num = len(glob.glob(config.DATA.TEST_SET_PATH + 'rr/*.mat'))
    else:
        data_num = len(glob.glob(config.DATA.TEST_SET_PATH + 'lms/*.tif'))
    for patch_num in range(data_num):
        patch_num = str(patch_num)
        if config.DATA.DATASET == "wv3":
            imdata = scipy.io.loadmat(os.path.join(config.DATA.TEST_SET_PATH, f"rr/rr{patch_num}.mat"))
            m_patch = imdata['lms_patch']
            p_patch = imdata['lpan_patch']
        else:
            m_patch = io.imread(os.path.join(config.DATA.TEST_SET_PATH, f"lms/lms_patch{patch_num}.tif"))
            p_patch = io.imread(os.path.join(config.DATA.TEST_SET_PATH, f"lpan/lpan_patch{patch_num}.tif"))
        m_patch = m_patch.astype(np.float32).transpose((2, 0, 1))
        # m_norm = np.array([scale_range(i, -1, 1) for i in m_patch])
        p_patch = p_patch.astype(np.float32)
        pan_img_size = np.size(p_patch, 0)
        # p_norm = np.array(scale_range(p_patch, -1, 1))
        ms_up = np.array([resize(i, (pan_img_size, pan_img_size), 3) for i in m_patch])
        if not config.MODEL.ORIGINAL_MS:
            inp = np.concatenate((ms_up, np.expand_dims(p_patch, axis=0)), axis=0).astype(np.float32)
        p_i = torch.zeros_like(torch.from_numpy(ms_up), dtype=torch.float32).unsqueeze(0)
        half_patch_size = patch_size // 2
        step_num = (pan_img_size // half_patch_size) - 1
        for i in range(step_num):
            for j in range(step_num):
                if config.MODEL.ORIGINAL_MS:
                    m_pat, p_pat = m_patch[:, i * half_patch_size // 4:(i * half_patch_size + patch_size) // 4,
                                   j * half_patch_size // 4:(j * half_patch_size + patch_size) // 4], p_patch[
                                                                                                      i * half_patch_size:(
                                                                                                                  i * half_patch_size + patch_size),
                                                                                                      j * half_patch_size:(
                                                                                                                  j * half_patch_size + patch_size)]
                    inp_patch = (torch.from_numpy(m_pat).float().unsqueeze(0),
                                 torch.from_numpy(p_pat).float().unsqueeze(0).unsqueeze(0))
                elif config.TRAIN.TYPE == "RRHP":
                    m_pat, p_pat = m_patch[:, i * half_patch_size // 4:(i * half_patch_size + patch_size) // 4,
                                   j * half_patch_size // 4:(j * half_patch_size + patch_size) // 4], p_patch[
                                                                                                      i * half_patch_size:(
                                                                                                              i * half_patch_size + patch_size),
                                                                                                      j * half_patch_size:(
                                                                                                              j * half_patch_size + patch_size)]
                    m_pat_hp = get_edge(m_pat.transpose((1, 2, 0))).transpose((2, 0, 1))
                    p_pat_hp = get_edge(p_pat)
                    m_pat_up = np.array([resize(i, (patch_size, patch_size), 3) for i in m_pat])
                    inp_patch = (torch.from_numpy(m_pat_hp).float().unsqueeze(0),
                                 torch.from_numpy(p_pat_hp).float().unsqueeze(0).unsqueeze(0),
                                 torch.from_numpy(m_pat_up).float().unsqueeze(0))
                else:
                    inp_patch = inp[:, i * half_patch_size:(i * half_patch_size + patch_size),
                                j * half_patch_size:(j * half_patch_size + patch_size)]
                    inp_patch = torch.from_numpy(inp_patch).float().unsqueeze(0)
                start_idx_i = 0
                start_idx_j = 0
                copy_size_i = patch_size
                copy_size_j = patch_size
                patch_idx_i = 0
                patch_idx_j = 0
                if i != 0:
                    start_idx_i = i * half_patch_size + (half_patch_size // 2)
                    copy_size_i = patch_size - (half_patch_size // 2)
                    patch_idx_i = half_patch_size // 2
                if j != 0:
                    start_idx_j = j * half_patch_size + (half_patch_size // 2)
                    copy_size_j = patch_size - (half_patch_size // 2)
                    patch_idx_j = half_patch_size // 2
                p_i[:, :, start_idx_i:start_idx_i + copy_size_i, start_idx_j:start_idx_j + copy_size_j] = \
                model(inp_patch)[:, :, patch_idx_i:, patch_idx_j:]
        save_path = os.path.join(config.OUTPUT,
                                 f"{config.MODEL.NAME.upper()}_{config.TAG.upper()}_RR_{config.DATA.DATASET.upper()}{patch_num}_{config.TRAIN.START_EPOCH - 1}")
        save_images(p_i, save_path)
        if config.DATA.DATASET == "wv3":
            unnormalization(p_i, save_path, 'predicted', "wv3")
        else:
            unnormalization(p_i, save_path, 'predicted')
        logger.info(
            f"{config.MODEL.NAME.upper()}_{config.TAG.upper()}_RR_{config.DATA.DATASET.upper()}{patch_num}_{config.TRAIN.START_EPOCH - 1}.jpg is done!")


@torch.no_grad()
def full_res_test(config, model, logger):
    model.eval()
    patch_size = config.MODEL.PAN_SIZE
    if config.DATA.DATASET == "wv3":
        data_num = len(glob.glob(config.DATA.TEST_SET_PATH + 'fr/*.mat'))
    else:
        data_num = len(glob.glob(config.DATA.TEST_SET_PATH + 'oms/*.tif'))
    for patch_num in range(data_num):
        patch_num = str(patch_num)
        if config.DATA.DATASET == "wv3":
            imdata = scipy.io.loadmat(os.path.join(config.DATA.TEST_SET_PATH, f"fr/fr{patch_num}.mat"))
            m_patch = imdata['oms_patch']
            p_patch = imdata['opan_patch']
        else:
            m_patch = io.imread(os.path.join(config.DATA.TEST_SET_PATH, f"oms/oms_patch{patch_num}.tif"))
            p_patch = io.imread(os.path.join(config.DATA.TEST_SET_PATH, f"opan/opan_patch{patch_num}.tif"))
        m_patch = m_patch.astype(np.float32).transpose((2, 0, 1))
        # m_norm = np.array([scale_range(i, -1, 1) for i in m_patch])
        p_patch = p_patch.astype(np.float32)
        pan_img_size = np.size(p_patch, 0)
        # p_norm = np.array(scale_range(p_patch, -1, 1))
        ms_up = np.array([resize(i, (pan_img_size, pan_img_size), 3) for i in m_patch])
        # ms_up = np.clip(ms_up, -1.0, 1.0)
        if not config.MODEL.ORIGINAL_MS:
            inp = np.concatenate((ms_up, np.expand_dims(p_patch, axis=0)), axis=0).astype(np.float32)
        if config.VERBOSE:
            aligned = torch.zeros_like(torch.from_numpy(m_patch), dtype=torch.float32).unsqueeze(0)
        p_i = torch.zeros_like(torch.from_numpy(ms_up), dtype=torch.float32).unsqueeze(0)
        half_patch_size = patch_size // 2
        step_num = (pan_img_size // half_patch_size) - 1
        for i in range(step_num):
            for j in range(step_num):
                if config.MODEL.ORIGINAL_MS:
                    m_pat, p_pat = m_patch[:, i * half_patch_size // 4:(i * half_patch_size + patch_size) // 4, j * half_patch_size // 4:(j * half_patch_size + patch_size) // 4], p_patch[i * half_patch_size:(i * half_patch_size + patch_size), j * half_patch_size:(j * half_patch_size + patch_size)]
                    inp_patch = (torch.from_numpy(m_pat).float().unsqueeze(0), torch.from_numpy(p_pat).float().unsqueeze(0).unsqueeze(0))
                elif config.TRAIN.TYPE == "RRHP":
                    m_pat, p_pat = m_patch[:, i * half_patch_size // 4:(i * half_patch_size + patch_size) // 4,
                                   j * half_patch_size // 4:(j * half_patch_size + patch_size) // 4], p_patch[
                                                                                                      i * half_patch_size:(
                                                                                                              i * half_patch_size + patch_size),
                                                                                                      j * half_patch_size:(
                                                                                                              j * half_patch_size + patch_size)]
                    m_pat_hp = get_edge(m_pat.transpose((1, 2, 0))).transpose((2, 0, 1))
                    p_pat_hp = get_edge(p_pat)
                    m_pat_up = np.array([resize(i, (patch_size, patch_size), 3) for i in m_pat])
                    inp_patch = (torch.from_numpy(m_pat_hp).float().unsqueeze(0),
                                 torch.from_numpy(p_pat_hp).float().unsqueeze(0).unsqueeze(0),
                                 torch.from_numpy(m_pat_up).float().unsqueeze(0))
                else:
                    inp_patch = inp[:, i * half_patch_size:(i * half_patch_size + patch_size), j * half_patch_size:(j * half_patch_size + patch_size)]
                    inp_patch = torch.from_numpy(inp_patch).float().unsqueeze(0)
                start_idx_i = 0
                start_idx_j = 0
                copy_size_i = patch_size
                copy_size_j = patch_size
                patch_idx_i = 0
                patch_idx_j = 0
                if i != 0:
                    start_idx_i = i * half_patch_size + (half_patch_size // 2)
                    copy_size_i = patch_size - (half_patch_size // 2)
                    patch_idx_i = half_patch_size // 2
                if j != 0:
                    start_idx_j = j * half_patch_size + (half_patch_size // 2)
                    copy_size_j = patch_size - (half_patch_size // 2)
                    patch_idx_j = half_patch_size // 2
                if config.VERBOSE:
                    aligned[:, :, start_idx_i:start_idx_i + copy_size_i, start_idx_j:start_idx_j + copy_size_j] = \
                    model(inp_patch)['byp'][:, :, patch_idx_i:, patch_idx_j:]
                p_i[:, :, start_idx_i:start_idx_i + copy_size_i, start_idx_j:start_idx_j + copy_size_j] = \
                model(inp_patch)[:, :, patch_idx_i:, patch_idx_j:]
        if config.VERBOSE:
            save_path = os.path.join(config.OUTPUT, f"{config.MODEL.NAME.upper()}_{config.TAG.upper()}_FR_{config.DATA.DATASET.upper()}{patch_num}_{config.TRAIN.START_EPOCH - 1}_aligned_ms")
            save_images(aligned, save_path)
            nearest_ms = F.interpolate(torch.tensor(m_patch).unsqueeze(0), scale_factor=4, mode='nearest')
            save_path = os.path.join(config.OUTPUT, f"{config.MODEL.NAME.upper()}_{config.TAG.upper()}_FR_{config.DATA.DATASET.upper()}{patch_num}_{config.TRAIN.START_EPOCH - 1}_nearest_msup")
            save_images(nearest_ms, save_path)
        save_path = os.path.join(config.OUTPUT, f"{config.MODEL.NAME.upper()}_{config.TAG.upper()}_FR_{config.DATA.DATASET.upper()}{patch_num}_{config.TRAIN.START_EPOCH - 1}")
        save_images(p_i, save_path)
        if config.DATA.DATASET == "wv3":
            unnormalization(p_i, save_path, 'predicted', "wv3")
        else:
            unnormalization(p_i, save_path, 'predicted')
        logger.info(
            f"{config.MODEL.NAME.upper()}_{config.TAG.upper()}_FR_{config.DATA.DATASET.upper()}{patch_num}_{config.TRAIN.START_EPOCH - 1}.jpg is done!")


@torch.no_grad()
def objective_test_rr(config, model, logger):
    model.eval()
    #start_ind = len(glob.glob(config.DATA.TESTSET_RR_PATH + 'rr/*.mat'))
    data_num = len(glob.glob(config.DATA.TESTSET_RR_PATH + 'test/*.mat'))
    #for patch_num in range(start_ind, start_ind + data_num):
    for patch_num in range(0, data_num):
        # patch_num = str(patch_num)
        #imdata = scipy.io.loadmat(os.path.join(config.DATA.TESTSET_RR_PATH, f"test/rr/rr{patch_num}.mat"))
        imdata = scipy.io.loadmat(os.path.join(config.DATA.TESTSET_RR_PATH, f"test/rr{patch_num}.mat"))
        m_patch = imdata['lms_patch']
        p_patch = imdata['lpan_patch']
        m_patch = m_patch.astype(np.float32).transpose((2, 0, 1))
        # m_norm = np.array([scale_range(i, -1, 1) for i in m_patch])
        p_patch = p_patch.astype(np.float32)
        # p_norm = np.array(scale_range(p_patch, -1, 1))
        if config.MODEL.ORIGINAL_MS:
            inp = (
            torch.from_numpy(m_patch).float().unsqueeze(0), torch.from_numpy(p_patch).float().unsqueeze(0).unsqueeze(0))
        elif config.TRAIN.TYPE == "RRHP":
            m_pat_hp = get_edge(m_patch.transpose((1, 2, 0))).transpose((2, 0, 1))
            p_pat_hp = get_edge(p_patch)
            m_pat_up = np.array([resize(i, (config.MODEL.PAN_SIZE, config.MODEL.PAN_SIZE), 3) for i in m_patch])
            inp = (torch.from_numpy(m_pat_hp).float().unsqueeze(0),
                         torch.from_numpy(p_pat_hp).float().unsqueeze(0).unsqueeze(0),
                         torch.from_numpy(m_pat_up).float().unsqueeze(0))
        else:
            ms_up = np.array([resize(i, (config.MODEL.PAN_SIZE, config.MODEL.PAN_SIZE), 3) for i in m_patch])
            # ms_up = np.clip(ms_up, -1.0, 1.0)
            inp = np.concatenate((ms_up, np.expand_dims(p_patch, axis=0)), axis=0).astype(np.float32)
            inp = torch.from_numpy(inp).float().unsqueeze(0)
        p_i = model(inp)
        save_path = os.path.join(config.OUTPUT,
                                 f"{config.MODEL.NAME.upper()}_{config.TAG.upper()}_RR_{config.DATA.DATASET.upper()}{patch_num}_{config.TRAIN.START_EPOCH - 1}_obj")
        save_images(p_i, save_path)
    logger.info(f"objective test rr is done!")


@torch.no_grad()
def objective_test_fr(config, model, logger):
    model.eval()
    #start_ind = len(glob.glob(config.DATA.TESTSET_FR_PATH + 'train/*.mat'))
    data_num = len(glob.glob(config.DATA.TESTSET_FR_PATH + 'test/*.mat'))
    #for patch_num in range(start_ind, start_ind + data_num):
    for patch_num in range(0, data_num):
        # patch_num = str(patch_num)
        imdata = scipy.io.loadmat(os.path.join(config.DATA.TESTSET_FR_PATH, f"test/fr{patch_num}.mat"))
        m_patch = imdata['oms_patch']
        p_patch = imdata['opan_patch']
        m_patch = m_patch.astype(np.float32).transpose((2, 0, 1))
        # m_norm = np.array([scale_range(i, -1, 1) for i in m_patch])
        p_patch = p_patch.astype(np.float32)
        # p_norm = np.array(scale_range(p_patch, -1, 1))
        if config.MODEL.ORIGINAL_MS:
            inp = (
            torch.from_numpy(m_patch).float().unsqueeze(0), torch.from_numpy(p_patch).float().unsqueeze(0).unsqueeze(0))
        elif config.TRAIN.TYPE == "RRHP":
            m_pat_hp = get_edge(m_patch.transpose((1, 2, 0))).transpose((2, 0, 1))
            p_pat_hp = get_edge(p_patch)
            m_pat_up = np.array([resize(i, (config.MODEL.PAN_SIZE, config.MODEL.PAN_SIZE), 3) for i in m_patch])
            inp = (torch.from_numpy(m_pat_hp).float().unsqueeze(0),
                         torch.from_numpy(p_pat_hp).float().unsqueeze(0).unsqueeze(0),
                         torch.from_numpy(m_pat_up).float().unsqueeze(0))
        else:
            ms_up = np.array([resize(i, (config.MODEL.PAN_SIZE, config.MODEL.PAN_SIZE), 3) for i in m_patch])
            # ms_up = np.clip(ms_up, -1.0, 1.0)
            inp = np.concatenate((ms_up, np.expand_dims(p_patch, axis=0)), axis=0).astype(np.float32)
            inp = torch.from_numpy(inp).float().unsqueeze(0)
        p_i = model(inp)
        save_path = os.path.join(config.OUTPUT,
                                 f"{config.MODEL.NAME.upper()}_{config.TAG.upper()}_FR_{config.DATA.DATASET.upper()}{patch_num}_{config.TRAIN.START_EPOCH - 1}_obj")
        save_images(p_i, save_path)
    logger.info(f"objective test fr is done!")


@torch.no_grad()
def sx_test(config, model, logger):
    model.eval()

    imdata = scipy.io.loadmat(r"F:\ResearchData\dataset\QBSX\fr_full_img.mat")
    m_patch = imdata['oms_patch']
    p_patch = imdata['opan_patch']
    # m_patch = cv.imread(r"F:\ResearchData\dataset\SPOT2/spot_ms3.tif")[100:164,100:164,:]
    # p_patch = cv.imread(r"F:\ResearchData\dataset\SPOT2/spot_pan3.tif")[400:656,400:656,0]
    m_patch = m_patch.astype(np.float32).transpose((2, 0, 1))
    C, H, W = m_patch.shape
    # m_norm = np.array([scale_range(i, -1, 1) for i in m_patch])
    # p_patch = p_patch[:,:,0].astype(np.float32)
    p_patch = p_patch.astype(np.float32)
    # p_norm = np.array(scale_range(p_patch, -1, 1))
    ms_up = np.array([resize(i, (H * 4, W * 4), 3) for i in m_patch])
    # ms_up = np.clip(ms_up, -1.0, 1.0)
    # io.imsave(r".\upms.tif",bgr2rgb(ms_up.transpose((1, 2, 0))).astype(np.uint8))
    inp = np.concatenate((ms_up, np.expand_dims(p_patch, axis=0)), axis=0).astype(np.float32)
    inp = torch.from_numpy(inp).float().unsqueeze(0)
    p_i, _ = model(inp)
    pred_img = np.clip(np.transpose(p_i.data.cpu().numpy()[0], (1, 2, 0)), 0, 255)
    io.imsave(f'./preqb.tif', pred_img[:, :, 0:3].astype(np.uint8))

    # ms_rgb = p_i.squeeze(0).detach().to('cpu').numpy().transpose((1, 2, 0))
    # ms_rgb = img.astype(np.float32).transpose((2, 0, 1))
    # ms_rgb = np.array([scale_range(i, 0, 1) for i in ms_rgb]).transpose((1, 2, 0))
    # ms_rgb = (ms_rgb-ms_rgb.min())/(ms_rgb.max()-ms_rgb.min())
    # ms_rgb = (ms_rgb + 1) / 2
    # ms_rgb = ms_rgb*255

    # ms_rgb = bgr2rgb(ms_rgb)
    # ms_rgb = np.clip(ms_rgb, 0, 255)

    """
    n = ms_rgb.shape[2]
    # lower_percent = 0.25
    # higher_percent = 99.75
    lower_percent = 0.05
    higher_percent = 99.95
    out = np.zeros_like(ms_rgb, dtype=np.uint8)
    for i in range(n):
        a = 0
        b = 255
        c = np.percentile(ms_rgb[:, :, i], lower_percent)
        d = np.percentile(ms_rgb[:, :, i], higher_percent)
        t = a + (ms_rgb[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t
    ms_rgb = out.astype(np.float32).transpose((2, 0, 1))

    ms_rgb = ms_rgb.astype(np.uint8)
    save_path = os.path.join(config.OUTPUT,
                             f"{config.MODEL.NAME.upper()}_{config.TAG.upper()}_FR_{config.DATA.DATASET.upper()}_{config.TRAIN.START_EPOCH - 1}_4.tif")
    io.imsave(save_path, ms_rgb)
    """
