import argparse
import torch
import time
import torchvision
import importlib

import numpy as np
import tqdm

from utils.flop_count import flop_count
import models
from data.build import build_dataset
from cfg import get_config
from models import build_model

import os

print(os.getcwd())

def get_args_parser():
    parser = argparse.ArgumentParser(
        "Set transformer detector FLOPs computation", add_help=False
    )
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--fig_num", default=5, type=int)
    parser.add_argument("--mode", default='flops', type=str)

    parser.add_argument(
        "--cfg",
        type=str,
        metavar="FILE",
        default=r"./configs/drnet/drnet.yaml",
        help="path to config file",
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
    parser.add_argument("--ckpt-choice", type=str, default="best_ergas", choices=["best_ergas", "best_ploss", "latest"],
                        help="the model to be tested")
    parser.add_argument("--test", action="store_true", default=False, help="Perform test only")
    parser.add_argument(
        "--throughput", action="store_true", default=False, help="Test throughput only"
    )

    parser.add_argument(
        "--data-set", default="gf2", choices=["gf2", "qb", "wv3", "gf2l"], type=str
    )
    parser.add_argument('--pixel-loss', type=str, default='l1', help='type of pixel loss, l1 or l2')
    parser.add_argument('--spcsim-loss', type=bool, default=False, help='whether to use spcsimloss')
    parser.add_argument('--br-loss', type=bool, default=False, help='use brloss or not')
    parser.add_argument('--base-lr', type=float, default=0.0002, help='basic learning rate')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='threads for data loading, too many would cause error, for ur own pc suggest 1')
    parser.add_argument('--gpu-set', type=int, default=1)

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

def warmup(model, inputs, N=10):
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()

def measure_time(model, inputs, N=10):
    warmup(model, inputs)
    s = time.time()
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()
    t = (time.time() - s) / N
    return t

def fmt_res(data):
    return data.mean(), data.std(), data.min(), data.max()

args, config = get_args_parser()

dataset = build_dataset(mode='val', config=config)
images = []
for idx in range(args.fig_num):
    imgs = dataset[idx]
    images.append(imgs[0])

device = torch.device("cuda")
results = {}

for model_name in [args.cfg]:
    model = build_model(config)
    print(str(model))
    model.to(device)
    model.eval()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params: {}".format(n_parameters))
    with torch.no_grad():
        tmp = []
        tmp2 = []
        for img in images:
            # one image
            if config.MODEL.ORIGINAL_MS:
                inputs = (torch.from_numpy(img[0]).to(device).unsqueeze(0), torch.from_numpy(img[1]).to(device).unsqueeze(0))
            else:
                inputs = torch.from_numpy(img).to(device).unsqueeze(0)
            if args.mode == "flops":
                res = flop_count(model, (inputs,))
                tmp.append(sum(res.values()))
            else:
                tmp.append(0)
            t = measure_time(model, inputs)
            tmp2.append(t)

        # format：mean，std，max，min
        results[model_name] = {
            "flops": fmt_res(np.array(tmp)),
            "time": fmt_res(np.array(tmp2)),
            "params": (n_parameters),
        }
print("=============================")
print("")
for r in results:
    print(r)
    for k, v in results[r].items():
        print(" ", k, ":", v)