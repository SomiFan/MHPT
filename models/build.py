"""
build.py 2022/8/1 0:05
Written by Wensheng Fan
"""
from .mhpt import MHPT


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == "mhpt":
        model = MHPT(
            ms_chans=config.MODEL.NUM_MS_BANDS,
            img_size=config.MODEL.PAN_SIZE,
            n_scale=config.MODEL.MHPT.NSCALE,
            embed_dim=config.MODEL.MHPT.EMB_DIM,
            depth=config.MODEL.MHPT.DEPTH,
            window_size=config.MODEL.MHPT.WIN_SIZE,
            num_heads=config.MODEL.MHPT.NHEAD,
            head_dim=config.MODEL.MHPT.HEAD_DIM,
            block_name=config.MODEL.MHPT.BLK_NAME,
            latent_dim=config.MODEL.MHPT.LATENT_DIM
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
