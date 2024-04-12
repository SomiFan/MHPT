"""
cfg.py 2023/2/27 9:25
Written by Wensheng Fan
"""
import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = [""]

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.DATASET = "qb"
# path of test data, overwritten in ./data/build.py
_C.DATA.TRAIN_SET_PATH = ""
_C.DATA.TEST_SET_PATH = r"F:\ResearchData\GF2\forpresentation/"
_C.DATA.TESTSET_RR_PATH = r"F:\ResearchData\dataset\QB128/"
_C.DATA.TESTSET_FR_PATH = r"F:\ResearchData\dataset\QBFR128/"
_C.DATA.BATCH_SIZE = 8
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = "hrpnet"
# Model name
_C.MODEL.NAME = "hrpnet_w18"
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ""
_C.MODEL.RESUME_PRETRAIN = ""
_C.MODEL.RESUME_ONLY_MODEL = False
# Number of MS bands, overwritten in ./data/build.py
_C.MODEL.NUM_MS_BANDS = 4
# size of pan patch at reduced resolution
_C.MODEL.MS_SIZE = 32
_C.MODEL.PAN_SIZE = 128
_C.MODEL.ORIGINAL_MS = False
_C.MODEL.SCALE_GAP = 4

# MHPT
_C.MODEL.MHPT = CN()
_C.MODEL.MHPT.NSCALE = 5
_C.MODEL.MHPT.EMB_DIM = 8
_C.MODEL.MHPT.DEPTH = 3
_C.MODEL.MHPT.WIN_SIZE = 8
_C.MODEL.MHPT.NHEAD = 2
_C.MODEL.MHPT.HEAD_DIM = 16
_C.MODEL.MHPT.BLK_NAME = "HAT"
_C.MODEL.MHPT.LATENT_DIM = 128

# PanCSCNet
_C.MODEL.PANCSC = CN()
_C.MODEL.PANCSC.N_FILTERS = 8
_C.MODEL.PANCSC.FILTER_SIZE = 8
_C.MODEL.PANCSC.N_LAYERS = 1

# PANFORMER
_C.MODEL.PANFORMER = CN()

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.TYPE = 'RR'
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 200
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0

# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Criterion
_C.TRAIN.SING_LOSS = True
# _C.TRAIN.PIXEL_LOSS = "l1"
_C.TRAIN.L1 = True
_C.TRAIN.L2 = False
_C.TRAIN.SPCSIM = False
_C.TRAIN.BR = False
_C.TRAIN.FAMSIS = False
_C.TRAIN.PSMSIS = False
_C.TRAIN.FAMEDGE = False
_C.TRAIN.PSMEDGE = False
_C.TRAIN.GRADLOSS = False
_C.TRAIN.LWF = True
_C.TRAIN.MMLWF = False
_C.TRAIN.LWFS = False
_C.TRAIN.LAMBDA_L1 = 1
_C.TRAIN.LAMBDA_L2 = 85
_C.TRAIN.LAMBDA_SPCSIM = 15
_C.TRAIN.LAMBDA_BR = 15
_C.TRAIN.LAMBDA_FAMSIS = 5
_C.TRAIN.LAMBDA_PSMSIS = 5 * 4
_C.TRAIN.SIS_NS = 4
_C.TRAIN.LAMBDA_FAMEDGE = 10
_C.TRAIN.LAMBDA_PSMEDGE = 10 * 4
_C.TRAIN.LAMBDA_GRADLOSS = 1
_C.TRAIN.LAMBDA_SIS = 1
_C.TRAIN.LAMBDA_GRAD = 1
_C.TRAIN.LAMBDA_LWFR = 1
_C.TRAIN.LAMBDA_LWFS = 1
_C.TRAIN.LAMBDA_QNR = 1000

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Path to output folder
_C.OUTPUT = "./experiments"
# Tag of experiment, overwritten by command line argument
_C.TAG = "default"
# Frequency to save checkpoint
_C.VAL_FREQ = 1
_C.SAVE_FREQ = 10
_C.VAL_TYPE = 'RR'
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = None
# Fixed random seed
_C.RANDOM_SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.TEST_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
_C.GPUSET = 2
_C.VERBOSE = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print("=> merge config from {}".format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.tag:
        config.TAG = args.tag
    if args.test:
        config.TEST_MODE = True
        config.TRAIN.AUTO_RESUME = False
        if args.ckpt_choice:
            config.MODEL.RESUME = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG,
                                               f"ckpt_{args.ckpt_choice}.pth")
    if args.throughput:
        config.THROUGHPUT_MODE = True
    if args.verbose:
        config.VERBOSE = True

    # support more datasets
    if args.data_set:
        config.DATA.DATASET = args.data_set
    if args.tsp:
        config.DATA.TRAIN_SET_PATH = args.tsp
    if args.base_lr:
        config.TRAIN.BASE_LR = args.base_lr
    if args.num_workers:
        config.DATA.NUM_WORKERS = args.num_workers
    if args.gpu_set:
        config.GPUSET = args.gpu_set

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def update_config_ft(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.tag:
        config.TAG = args.tag
    if args.pret_ckpt:
        config.MODEL.RESUME_PRETRAIN = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG,
                                                    f"ckpt_{args.pret_ckpt}.pth")
    if args.test:
        config.TEST_MODE = True
        config.TRAIN.AUTO_RESUME = False
        if args.ckpt_choice:
            config.MODEL.RESUME = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG,
                                               f"ckpt_{args.ckpt_choice}_ft.pth")
    if args.throughput:
        config.THROUGHPUT_MODE = True
    if args.verbose:
        config.VERBOSE = True

    # support more datasets
    if args.data_set:
        config.DATA.DATASET = args.data_set
    if args.tsp:
        config.DATA.TRAIN_SET_PATH = args.tsp
    if args.base_lr:
        config.TRAIN.BASE_LR = args.base_lr
    if args.num_workers:
        config.DATA.NUM_WORKERS = args.num_workers
    if args.gpu_set:
        config.GPUSET = args.gpu_set

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


def get_config_default(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()

    return config


def get_config_ft(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config_ft(config, args)

    return config