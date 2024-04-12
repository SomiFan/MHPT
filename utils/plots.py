"""
plots.py 2022/11/30 19:17
Written by Wensheng Fan
"""
import torch
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
from pathlib import Path

# settings
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


def feature_visualization(x, module_type, stage, n=32, save_dir=Path('experiments')):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    batch, channels, height, width = x.shape  # batch, channels, height, width
    if height > 1 and width > 1:
        f = save_dir / f"stage{stage}_{module_type.split('.')[-1]}_features.png"  # filename

        blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
        n = min(n, channels)  # number of plots
        fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=False)  # 8 rows x n/8 cols
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.001)  # wspace=0.05, hspace=0.05
        for i in range(n):
            ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
            ax[i].axis('off')

        print(f'Saving {f}... ({n}/{channels})')
        plt.savefig(f, dpi=300, bbox_inches='tight')
        plt.close()
        np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())  # npy save


def feat_split_visualize(x, module_type, stage, save_dir=Path('experiments')):
    batch, channels, height, width = x.shape  # batch, channels, height, width
    blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
    #fig, ax = plt.subplots(1, 1)
    #ax.axis('off')
    #plt.tight_layout()
    plt.figure(figsize=(math.ceil(width/300), math.ceil(height/300)), dpi=300)
    cmap = matplotlib.cm.jet
    # 这里设置colormap的固定值
    norm = matplotlib.colors.Normalize(vmin=0, vmax=256)
    #plt.colorbar(im, cmap=cmap, norm=norm, ticks=[-1, 0, 1])
    for i in range(channels):
        f = save_dir / f"{stage}{module_type.split('.')[-1]}_feat{i}.eps"  # filename
        plt.imshow(blocks[i].squeeze(), cmap=cmap)
        #ax.imshow(blocks[i].squeeze())
        print(f'Saving {f}... ({i}/{channels})')
        plt.axis('off')
        plt.savefig(f, bbox_inches='tight', pad_inches=0)
        plt.savefig(f.with_suffix('.tiff'), bbox_inches='tight', pad_inches=0)
    plt.close()
    f = save_dir / f"{stage}{module_type.split('.')[-1]}_feats.npy"
    np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())

