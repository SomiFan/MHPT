import argparse
from cfg import get_config

import numpy as np
import random
import os
import time
import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing

# from torchstat import stat

torch.multiprocessing.set_sharing_strategy('file_system')

from tensorboardX import SummaryWriter
from models import build_model
from data import build_loader
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from utils import auto_resume_helper, load_checkpoint, save_checkpoint, create_logger, save_checkpointfr, \
    load_checkpointfr, save_checkpoint_reg
from train_functions import validate, throughput, train_one_epoch, validatefr
from test_functions import reduced_res_test, full_res_test, objective_test_rr, objective_test_fr
from loss_funcs import SpcSimilarityLoss, BandRelationLoss, SiSLoss, EdgeLoss


def parse_option():
    parser = argparse.ArgumentParser('HRPFormer training and evaluation script', add_help=False)
    parser.add_argument(
        "--cfg",
        type=str,
        metavar="FILE",
        default=r"./configs/pancsc/pancsc_t.yaml",
        help="path to config file, model name and type are in the yaml cfg file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, default=1, help="batch size for single GPU")
    parser.add_argument("--tag", type=str, default="gf2", help="tag of experiment")
    parser.add_argument("--ckpt-choice", type=str, default="50",
                        help="the model to be tested")
    # choices=["best_ergas", "min_loss", "latest", "best_sam", "best_qnr", "80"]
    parser.add_argument("--test", action="store_true", help="Perform test only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )
    # set training dataset
    parser.add_argument(
        "--data-set", default="gf2", choices=["gf2", "qb", "wv3", "gf2l", "spot", "qbsx"], type=str
    )
    parser.add_argument(
        "--tsp", default="", type=str, help="Training Set Path"
    )
    parser.add_argument('--base-lr', type=float, default=0.0005, help='basic learning rate')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='threads for data loading, too many would cause error, for ur own pc suggest 1')
    parser.add_argument("--verbose", action="store_false", help="output verbose for analysis")
    parser.add_argument('--gpu-set', type=int, default=1, help='if >1, should modify the code otherwise wont be valid')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


def main(config):
    (
        train_loader,
        val_loader,
    ) = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)

    optimizer = build_optimizer(config, model)

    if device == "cuda":
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    logger.info(str(model))

    # 模型参数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    # flops
    if device == "cuda":
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    # stat(model_without_ddp.to('cpu'), (5, 128, 128))

    # from ptflops import get_model_complexity_info
    # with torch.cuda.device(0):
    #    macs, _ = get_model_complexity_info(model_without_ddp, (5, 128, 128), as_strings=True, print_per_layer_stat=True, verbose=True)
    #    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))

    #from thop import profile
    #from thop import clever_format

    #input_img = torch.randn(1, 5, 128, 128).cuda()
    #macs, _ = profile(model_without_ddp, inputs=(input_img,))
    #macs, _ = clever_format([macs, _], '%.3f')

    #print('Computational complexity:', macs)

    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    # Define Losses
    criterion = {}
    if config.TRAIN.L1:
        # criterion['l1'] = nn.L1Loss().cuda()
        criterion['l1'] = nn.L1Loss().to(device)
    if config.TRAIN.L2:
        criterion['l2'] = nn.MSELoss().to(device)
    if config.TRAIN.SPCSIM:
        criterion['spcsim'] = SpcSimilarityLoss().to(device)
    if config.TRAIN.BR:
        criterion['br'] = BandRelationLoss().to(device)
    if config.TRAIN.FAMSIS:
        criterion['fam_sis'] = SiSLoss(s_stride=1, ns=config.TRAIN.SIS_NS).to(device)
    if config.TRAIN.PSMSIS:
        criterion['psm_sis'] = SiSLoss(ns=config.TRAIN.SIS_NS).to(device)
    if config.TRAIN.FAMEDGE:
        criterion['fam_edge'] = EdgeLoss().to(device)
    if config.TRAIN.PSMEDGE:
        criterion['psm_edge'] = EdgeLoss().to(device)
    if config.TRAIN.GRADLOSS:
        criterion['gradloss'] = EdgeLoss().to(device)

    if config.TRAIN.SING_LOSS:
        criterion = list(criterion.values())[0]

    config.defrost()
    config.PRINT_FREQ = len(train_loader) - 1
    config.freeze()
    if config.TRAIN.TYPE == 'FR':
        best_qnr = 0.0
    else:
        best_ergas = 1000.0
        best_sam = 1000.0
    min_loss = 1000.0
    # is_best_ergas = True
    # is_best_ploss = True
    # is_best_sam = True

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    if config.MODEL.RESUME:
        # 因为ckpt load到cpu上，所以这里把model_without_ddp传进去
        if config.TRAIN.TYPE == 'FR':
            min_loss, best_qnr = load_checkpointfr(config, model_without_ddp, optimizer, lr_scheduler, logger)
        else:
            min_loss, best_ergas, best_sam = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    if config.THROUGHPUT_MODE:
        throughput(config, val_loader, model, logger)
        return

    if config.TEST_MODE:
        logger.info("Start reduced resolution test")
        #reduced_res_test(config, model, logger)
        #objective_test_rr(config, model, logger)
        logger.info("Start full resolution test")
        #full_res_test(config, model, logger)
        objective_test_fr(config, model, logger)
        # sx_test(config, model, logger)
        return

    writer = SummaryWriter(config.OUTPUT)
    writer_dict = {
        'writer': writer,
        'train_global_steps': config.TRAIN.START_EPOCH * len(train_loader),
    }

    logger.info("Start training")
    start_time = time.time()

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        # train_loader.sampler.set_epoch(epoch)
        if config.TRAIN.TYPE == 'FR':
            is_best_qnr = False
        else:
            is_best_ergas = False
            is_best_sam = False
        is_min_loss = False

        train_one_epoch(
            config,
            model,
            criterion,
            train_loader,
            optimizer,
            epoch,
            lr_scheduler,
            logger,
            writer_dict,
            device
        )
        if epoch % config.VAL_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1):
            if config.TRAIN.TYPE == 'FR':
                loss, qnr = validatefr(config, val_loader, model, criterion, logger, epoch, writer_dict, device)
                if qnr > best_qnr:
                    best_qnr = qnr
                    is_best_qnr = True
                else:
                    is_best_qnr = False
            else:
                loss, ergas, sam = validate(config, val_loader, model, criterion, logger, epoch, writer_dict, device)
                if ergas < best_ergas:
                    best_ergas = ergas
                    is_best_ergas = True
                else:
                    is_best_ergas = False
                if sam < best_sam:
                    best_sam = sam
                    is_best_sam = True
                else:
                    is_best_sam = False
            if loss < min_loss:
                min_loss = loss
                is_min_loss = True
            else:
                is_min_loss = False
        if epoch % config.SAVE_FREQ == 0:
            save_checkpoint_reg(
                config,
                epoch,
                model_without_ddp,
                min_loss,
                best_ergas,
                best_sam,
                optimizer,
                lr_scheduler,
                logger,
            )
        if config.TRAIN.TYPE == 'FR':
            save_checkpointfr(
                config,
                epoch,
                model_without_ddp,
                min_loss,
                is_min_loss,
                best_qnr,
                is_best_qnr,
                optimizer,
                lr_scheduler,
                logger
            )
        else:
            save_checkpoint(
                config,
                epoch,
                model_without_ddp,
                min_loss,
                is_min_loss,
                best_ergas,
                is_best_ergas,
                best_sam,
                is_best_sam,
                optimizer,
                lr_scheduler,
                logger,
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    if config.TRAIN.TYPE == 'FR':
        logger.info(f"Best QNR: {best_qnr}")
    else:
        logger.info(f"Best ERGAS: {best_ergas}")


if __name__ == "__main__":
    _, config = parse_option()

    device = "cpu"

    if torch.cuda.is_available():
        # A bool that, if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
        cudnn.benchmark = True
        # gpus = [gpu for gpu in range(config.GPUSET)]
        # torch.cuda.set_device(gpus[0])
        gpus = [0]
        torch.cuda.set_device(0)
        device = "cuda"

        # 固定随机数种子，只要输入一样，每次计算的结果也一样
        if config.SEED is not None:
            torch.manual_seed(config.RANDOM_SEED)
            torch.cuda.manual_seed(config.RANDOM_SEED)
            torch.cuda.manual_seed_all(config.RANDOM_SEED)
            np.random.seed(config.RANDOM_SEED)
            random.seed(config.RANDOM_SEED)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(
        output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}"
    )
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
