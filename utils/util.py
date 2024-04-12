import numpy as np
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.utils as vutils
from skimage import io
import cv2 as cv
import math
import torch
import os
import sys
import logging
import functools
import scipy.io
import torch.nn.functional as F
from termcolor import colored
import time
from datetime import datetime
import dateutil.tz


# 绘制loss的变化曲线图
def save_figure(losses, path, epoch, label):
    # plt.plot(losses_d, label=label, color='b')
    # colors = ['r', 'b', 'm', 'y', 'g']
    # try:
    # if isinstance(losses[0], list):
    # for loss,c in zip(losses, colors):
    # plt.plot(loss, label=label, color=c)

    # except:
    if len(losses) == 2:
        plt.plot(losses[0], label='adv-loss', color='r')
        plt.plot(losses[1], label='recon-loss', color='g')

    else:
        plt.plot(losses, label=label, color='r')
        plt.title("Experiment: {} -- {}: {}".format(path, label, epoch))

    plt.legend()
    plt.savefig("results-{}/epoch{}-{}-loss.pdf".format(path, epoch, label, ))
    plt.close()


# 与opencv中的归一化函数normalize()中normType为NORM_MINMAX(数组的数值被平移或缩放到一个指定的范围，线性归一化)时一样，
# 当min=0,max=1时与我们常见的归一化一样
def scale_range(input, min, max):
    input += -(np.min(input))
    input /= (1e-9 + np.max(input) / (max - min + 1e-9))
    input += min
    return input


def rgb2gray(rgb):
    r, g, b, nir = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2], rgb[:, :, 3]
    # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = 0.25 * r + 0.25 * g + 0.25 * b + 0.25 * nir
    return gray


def visualize_tensor(imgs, epoch, it, name):
    fname = "tensors-{}/{}/{}-{}.jpg".format("opt.savePath", epoch, it, name)
    # size返回tensor的三维的tuple:torch.Size([2, 3, 4])
    vutils.save_image(
        tensor=imgs, filename=fname, normalize=True, nrow=imgs.size()[0] // 2)


# 求batch的平均metric值，第三个参数是metric函数，前两个是目标图像batch的tensor和预测图像batch的tensor
def avg_metric(target, prediction, metric, sensor='4bands'):
    sum = 0.0
    batch_size = len(target)
    for i in range(batch_size):
        target_img = np.transpose(target.data.cpu().numpy()[i], (1, 2, 0))
        pred_img = np.transpose(prediction.data.cpu().numpy()[i], (1, 2, 0))
        """
        if sensor == 'wv3':
            target_img = np.array([target_img[:, :, 1], target_img[:, :, 2], target_img[:, :, 4]], dtype=np.float32).transpose((1, 2, 0))
            pred_img = np.array([pred_img[:, :, 1], pred_img[:, :, 2], pred_img[:, :, 4]], dtype=np.float32).transpose((1, 2, 0))
        target_img=bgr2rgb(target_img[:, :, 0:3])
        target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min()) * 255
        pred_img=bgr2rgb(pred_img[:, :, 0:3])
        pred_img = (pred_img - pred_img.min()) / (pred_img.max() - pred_img.min()) * 255
        """
        # io.imsave(f'./pre{i}.jpg',pred_img)
        sum += metric(pred_img, target_img)
    return sum / batch_size


def show_image(im):
    if len(im.shape) == 2:
        matplotlib.use('TkAgg')
        im = scale_range(im, 0, 255).astype(np.uint8)
        plt.figure(figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
        plt.imshow(im, cmap='gray')
        plt.show()

    elif len(im.shape) == 3:
        matplotlib.use('TkAgg')
        im = np.array([scale_range(i, 0, 255) for i in im.transpose((2, 0, 1))]).transpose(1, 2, 0)[..., :3].astype(
            np.uint8)
        plt.figure(figsize=(16, 16), dpi=80, facecolor='w', edgecolor='k')
        plt.imshow(im)
        plt.show()


def patch_sixteen(images):
    size = images.shape[-1]
    patched = np.zeros((images.shape[1], images.shape[2] * 4, images.shape[3] * 4))
    for i in range(4):
        for j in range(4):
            patched[..., i * size:(i + 1) * size, j * size:(j + 1) * size] += images[i * 4 + j]
    return np.array(patched)


def divide_sixteen(image):
    size = image.shape[-1] / 4
    divided = []
    for i in range(4):
        for j in range(4):
            divided += image[..., int(i * size):int((i + 1) * size), int(j * size):int((j + 1) * size)],
    return np.array(divided)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def normalize_torch(data):
    _range = torch.max(data) - torch.min(data)
    return (data - torch.min(data)) / _range


def unnormalization(img, path, name, sensor="4bands"):
    if name == 'label':
        img = img.transpose((1, 2, 0))
    elif name == 'predicted':
        img = img.squeeze(0).detach().to('cpu').numpy().transpose((1, 2, 0))
    if sensor == "wv3":
        img = np.array([img[:, :, 1], img[:, :, 2], img[:, :, 4]], dtype=np.float32).transpose((1, 2, 0))
    img = img[:, :, 0:3]
    ms_rgb = bgr2rgb(img)
    ms_rgb = ms_rgb.astype(np.float32).transpose((2, 0, 1))
    ms_rgb = np.array([scale_range(i, 0, 1) for i in ms_rgb]).transpose((1, 2, 0))
    # ms_rgb = (ms_rgb-ms_rgb.min())/(ms_rgb.max()-ms_rgb.min())
    # ms_rgb = (ms_rgb + 1) / 2
    ms_rgb = ms_rgb * 255
    ms_rgb = ms_rgb.astype(np.uint8)
    path = path + '.jpg'
    io.imsave(path, ms_rgb)
    # mpimg.imsave(path, ms_rgb)
    # cv.imwrite(path, ms_rgb)


def bgr2rgb(img):
    img_rgb = np.zeros(img.shape, np.float32)
    img_rgb[:, :, 0] = img[:, :, 2]
    img_rgb[:, :, 1] = img[:, :, 1]
    img_rgb[:, :, 2] = img[:, :, 0]
    return img_rgb


def tif2jpg(img, path, name):
    if name == 'ms':
        img = img.squeeze(0).detach().numpy().transpose((1, 2, 0))
        img = img[:, :, 0:3]
        ms_rgb = bgr2rgb(img)
        n = ms_rgb.shape[2]
        lower_percent = 0.6
        higher_percent = 99.4
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
        path = path + '.jpg'
        io.imsave(path, out)
    elif name == 'pan':
        img = normalization(img)
        img = img * 255
        img = img.astype(np.uint8)
        path = path + '.jpg'
        io.imsave(path, img)
    # 把numpy的tif遥感图像转成jpg
    elif name == 'label':
        img = img.transpose((1, 2, 0))
        img = img[:, :, 0:3]
        ms_rgb = bgr2rgb(img)
        n = ms_rgb.shape[2]
        lower_percent = 0.6
        higher_percent = 99.4
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
        path = path + '.jpg'
        io.imsave(path, out)


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith("pth")]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime
        )
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def auto_resume_helper_ft(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith("ft.pth")]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime
        )
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(
        f"==============> Resuming form {config.MODEL.RESUME}...................."
    )
    if config.MODEL.RESUME.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    # 打印模型的 state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    logger.info(msg)
    min_loss = 1000.0
    best_ergas = 1000.0
    best_sam = 1000.0
    if (
            not config.TEST_MODE
            and not config.MODEL.RESUME_ONLY_MODEL
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if "epoch" in checkpoint:
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        config.freeze()
        logger.info(
            f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})"
        )
    if "min_loss" in checkpoint:
        min_loss = checkpoint["min_loss"]
    if "best_ergas" in checkpoint:
        best_ergas = checkpoint["best_ergas"]
    if "best_sam" in checkpoint:
        best_sam = checkpoint["best_sam"]

    del checkpoint
    torch.cuda.empty_cache()
    return min_loss, best_ergas, best_sam


def load_checkpointfr(config, model, optimizer, lr_scheduler, logger):
    logger.info(
        f"==============> Resuming form {config.MODEL.RESUME}...................."
    )
    if config.MODEL.RESUME.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)
    min_loss = 1000.0
    best_qnr = 0.0
    if (
            not config.TEST_MODE
            and not config.MODEL.RESUME_ONLY_MODEL
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
    ):
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if "epoch" in checkpoint:
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        config.freeze()
        logger.info(
            f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})"
        )
    if "min_loss" in checkpoint:
        min_loss = checkpoint["min_loss"]
    if "best_qnr" in checkpoint:
        best_qnr = checkpoint["best_qnr"]

    del checkpoint
    torch.cuda.empty_cache()
    return min_loss, best_qnr


def load_checkpoint_ft(config, model, model_o, optimizer, lr_scheduler, logger):
    min_loss = 1000.0
    best_qnr = 0.0
    logger.info(
        f"==============> Resuming pretrained model from {config.MODEL.RESUME_PRETRAIN}...................."
    )
    if config.MODEL.RESUME_PRETRAIN.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME_PRETRAIN, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(config.MODEL.RESUME_PRETRAIN, map_location="cpu")
    msg = model_o.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)
    if config.MODEL.RESUME:
        logger.info(
            f"==============> Resuming fine-tuned model from {config.MODEL.RESUME}...................."
        )
        if config.MODEL.RESUME.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                config.MODEL.RESUME, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")
        if (
                not config.TEST_MODE
                and not config.MODEL.RESUME_ONLY_MODEL
                and "optimizer" in checkpoint
                and "lr_scheduler" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "epoch" in checkpoint:
            config.defrost()
            config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
            config.freeze()
            logger.info(
                f"=> loading '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})"
            )
        if "min_loss" in checkpoint:
            min_loss = checkpoint["min_loss"]
        if "best_qnr" in checkpoint:
            best_qnr = checkpoint["best_qnr"]
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    logger.info(msg)

    del checkpoint
    torch.cuda.empty_cache()
    return min_loss, best_qnr


def save_checkpoint(
        config, epoch, model, min_loss, is_min_loss, best_ergas, is_best_ergas, best_sam, is_best_sam, optimizer,
        lr_scheduler, logger
):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "min_loss": min_loss,
        "best_ergas": best_ergas,
        "best_sam": best_sam,
        "epoch": epoch,
        "config": config,
    }

    if is_min_loss:
        torch.save(save_state, os.path.join(config.OUTPUT, f"ckpt_min_loss.pth"))
    if is_best_ergas:
        torch.save(save_state, os.path.join(config.OUTPUT, f"ckpt_best_ergas.pth"))
    if is_best_sam:
        torch.save(save_state, os.path.join(config.OUTPUT, f"ckpt_best_sam.pth"))

    # save_path = os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")
    save_path = os.path.join(config.OUTPUT, f"ckpt_latest.pth")
    # logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    # logger.info(f"{save_path} saved !!!")


def save_checkpoint_reg(
        config, epoch, model, min_loss, best_ergas, best_sam, optimizer,
        lr_scheduler, logger
):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "min_loss": min_loss,
        "best_ergas": best_ergas,
        "best_sam": best_sam,
        "epoch": epoch,
        "config": config,
    }

    # save_path = os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")
    save_path = os.path.join(config.OUTPUT, f"ckpt_{epoch}.pth")
    # logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    # logger.info(f"{save_path} saved !!!")


def save_checkpointfr(
        config, epoch, model, min_loss, is_min_loss, best_qnr, is_best_qnr, optimizer, lr_scheduler, logger
):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "min_loss": min_loss,
        "best_qnr": best_qnr,
        "epoch": epoch,
        "config": config,
    }
    if is_min_loss:
        torch.save(save_state, os.path.join(config.OUTPUT, f"ckpt_min_loss.pth"))

    if is_best_qnr:
        torch.save(save_state, os.path.join(config.OUTPUT, f"ckpt_best_qnr.pth"))

    # save_path = os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")
    save_path = os.path.join(config.OUTPUT, f"ckpt_latest.pth")
    # logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    # logger.info(f"{save_path} saved !!!")


def save_checkpoint_ft(
        config, epoch, model, min_loss, is_min_loss, best_qnr, is_best_qnr, optimizer, lr_scheduler, logger
):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "min_loss": min_loss,
        "best_qnr": best_qnr,
        "epoch": epoch,
        "config": config,
    }
    if is_min_loss:
        torch.save(save_state, os.path.join(config.OUTPUT, f"ckpt_min_loss_ft.pth"))

    if is_best_qnr:
        torch.save(save_state, os.path.join(config.OUTPUT, f"ckpt_best_qnr_ft.pth"))

    # save_path = os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")
    save_path = os.path.join(config.OUTPUT, f"ckpt_latest_ft.pth")
    # logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    # logger.info(f"{save_path} saved !!!")


@functools.lru_cache()
def create_logger(output_dir, dist_rank=0, name=""):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    # color_fmt = (
    #    colored("[%(asctime)s %(name)s]", "green")
    #    + colored("(%(filename)s %(lineno)d)", "yellow")
    #    + ": %(levelname)s %(message)s"
    # )
    color_fmt = "%(message)s"

    # create console handlers for master process
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(console_handler)

    # create file handlers
    file_handler = logging.FileHandler(
        os.path.join(output_dir, f"log.txt"), mode="a"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    return logger


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def save_images(fused_img, path, flag='none'):
    img = fused_img.squeeze(0).detach().to('cpu').numpy().transpose((1, 2, 0))
    if flag == 'normalized':
        img = (img + 1) / 2
        img = img * 255
    img = img.astype(np.float64)
    path = path + '.mat'
    scipy.io.savemat(path, {'I_F': img})


def pad_sr(x0, N):  # e.g. N = 2**x
    """
    Pad an image to have size that is integer multiple of N
    Args: x0: 4D image, N: integer multiple e.g. 2**x
    """
    hw0 = torch.tensor(x0.shape, dtype=torch.float32)[2:]
    hw1 = torch.ceil(hw0 / N) * N
    pp1 = torch.ceil((hw1 - hw0) / 2).to(torch.int32)
    pp2 = (hw1 - hw0 - pp1).to(torch.int32)
    n_pad = (pp1[1].item(), pp2[1].item(), pp1[0].item(), pp2[0].item())
    x0 = F.pad(x0, pad=n_pad, mode='reflect')
    return x0

def get_edge(data):  # for training: HxWxC
    rs = np.zeros_like(data)
    # N = data.shape[0]
    # for i in range(N):
    if len(data.shape) == 2:
        rs[:, :] = data[:, :] - cv.boxFilter(data[:, :], -1, (5, 5))
    else:
        rs[:, :, :] = data[:, :, :] - cv.boxFilter(data[:, :, :], -1, (5, 5))
    return rs


if __name__ == "__main__":
    oms = scipy.io.loadmat(r"D:\Document\experiments\DL_Pansharp\PNN_FT_QB\PNN_FT_QB_FR_QB12_237.mat")['I_F'].astype(np.float32).transpose(
        (2, 0, 1))
    opan = io.imread(r"F:\ResearchData\dataset\QB\forpresentation\opan\opan_patch12.tif").astype(np.float32)
    oms = torch.from_numpy(oms).float().unsqueeze(0)
    opan = torch.from_numpy(opan).float().unsqueeze(0).unsqueeze(0)
    # ms_up = F.interpolate(oms, scale_factor=4, mode='bicubic', align_corners=False)
    ms_gray = torch.mean(oms, 1, True)
    ms_gray_jpg = normalization(ms_gray[0, 0, :, :].cpu().numpy()) * 255
    io.imsave("./ms_gray_qb12.jpg", ms_gray_jpg.astype(np.uint8))
    opan_jpg = normalization(opan[0, 0, :, :].cpu().numpy()) * 255
    io.imsave("./pan_qb12.jpg", opan_jpg.astype(np.uint8))
    resid_gray = torch.abs(ms_gray - opan)[0, 0, :, :].cpu().numpy()
    resid_gray = normalization(resid_gray) * 255
    io.imsave("./resid_gray_gray_qb12.jpg", resid_gray.astype(np.uint8))
    mask_gray = resid_gray
    mask_gray[resid_gray < 20] = 0
    mask_gray[resid_gray >= 20] = 255
    io.imsave("./mask_gray_qb12.jpg", mask_gray.astype(np.uint8))
    i_f = io.imread(r"D:\Document\experiments\DL_Pansharp\PNN_FT_QB\PNN_FT_QB_FR_QB12_237.jpg").astype(
        np.uint8).transpose((2, 0, 1))
    i_f = torch.from_numpy(i_f).unsqueeze(0)
    for i in range(3):
        i_f[0, i, :, :, ][resid_gray < 25] = 0
    io.imsave("./i_f_masked_qb12.jpg", i_f.squeeze(0).numpy().transpose((1, 2, 0)))
