"""
mhpt.py 2023/2/26 21:47
Written by Wensheng Fan
"""
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from models.modules import HATBlock, RCSAB, AHATBlock, SATBlock

blocks_dict = {
    "HAT": HATBlock,
    "AHAT": AHATBlock,
    "SAT": SATBlock
}


class ResBlock(nn.Module):
    """simple and plain res-block"""

    def __init__(self, inplanes, planes, ks=3):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, ks, 1, padding=(ks - 1) // 2, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, inplanes, ks, 1, padding=(ks - 1) // 2, bias=True)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out += residual
        return self.act2(out)


class MSPatchEmbed(nn.Module):
    r""" Image to Multi-Scale Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        ms_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=128, n_scale=1, in_chans=4, embed_dim=8, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.n_scale = n_scale
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj1 = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=1)
        #self.proj1 = nn.Conv2d(in_chans, embed_dim, kernel_size=9, stride=1, padding=4)
        # self.proj2 = nn.Conv2d(in_chans, embed_dim * 2, kernel_size=patch_size * 2, stride=patch_size * 2)
        if n_scale >= 2:
            self.proj2 = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        if n_scale >= 3:
            self.proj3 = nn.Conv2d(in_chans, embed_dim, kernel_size=5, stride=1, padding=2)
        if n_scale >= 4:
            self.proj4 = nn.Conv2d(in_chans, embed_dim, kernel_size=7, stride=1, padding=3)
        if n_scale >= 5:
            self.proj5 = nn.Conv2d(in_chans, embed_dim, kernel_size=9, stride=1, padding=4)
        if n_scale >= 6:
            self.proj6 = nn.Conv2d(in_chans, embed_dim, kernel_size=11, stride=1, padding=5)
        if norm_layer is not None:
            self.norm1 = norm_layer(embed_dim * n_scale)
            # self.norm2 = norm_layer(embed_dim * 2)
        else:
            self.norm1 = None

    def forward(self, x):
        emb = [self.proj1(x).flatten(2).transpose(1, 2)]
        if self.n_scale >= 2:
            emb.append(self.proj2(x).flatten(2).transpose(1, 2))
        if self.n_scale >= 3:
            emb.append(self.proj3(x).flatten(2).transpose(1, 2))
        if self.n_scale >= 4:
            emb.append(self.proj4(x).flatten(2).transpose(1, 2))
        if self.n_scale >= 5:
            emb.append(self.proj5(x).flatten(2).transpose(1, 2))
        if self.n_scale >= 6:
            emb.append(self.proj6(x).flatten(2).transpose(1, 2))
        emb = torch.cat(emb, dim=-1)
        if self.norm1 is not None:
            emb = self.norm1(emb)
        return emb


class MHPT(nn.Module):
    """
    Args:
        ms_chans: input channel num, n_unitchan: unit channel num, bn_blocks: number of bottleneck blocks,
        gpu_ids: ids of gpu in use
    """

    def __init__(self, ms_chans=4, img_size=128, n_scale=1, embed_dim=8, norm_layer=nn.LayerNorm, patch_norm=True,
                 depth=3, window_size=8, mlp_ratio=4, qkv_bias=True,
                 num_heads=2, head_dim=16, block_name="PTB", latent_dim=128):
        super(MHPT, self).__init__()
        self.ms_chans = ms_chans
        self.img_size = img_size
        self.patch_norm = patch_norm
        self.embed_dim = embed_dim
        self.n_scale = n_scale

        self.ms_emb_layers = MSPatchEmbed(img_size=img_size, n_scale=n_scale, in_chans=ms_chans,
                                          embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        self.pan_emb_layers = MSPatchEmbed(img_size=img_size, n_scale=n_scale, in_chans=1,
                                           embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        self.blocks1 = nn.ModuleList([
            blocks_dict[block_name](input_resolution=to_2tuple(img_size), dim=embed_dim*self.n_scale, num_heads=num_heads,
                                    head_dim=head_dim, window_size=window_size,
                                    shift_size=0 if (i % 2 == 0) else window_size // 2, qkv_bias=qkv_bias,
                                    mlp_ratio=mlp_ratio, norm_layer=norm_layer,
                                    )
            for i in range(depth)])
        #self.recon_head = RCSAB(embed_dim * self.n_scale * 3, ms_chans, input_resolution=img_size, expansion=1, latent_dim=latent_dim)
        self.recon_head = RCSAB(embed_dim*self.n_scale, ms_chans, input_resolution=img_size, expansion=1, latent_dim=latent_dim)
        #self.recon_head = nn.Conv2d(embed_dim * self.n_scale, ms_chans, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        pan, ms = x[:, self.ms_chans, :, :].unsqueeze(1), x[:, :self.ms_chans, :, :]
        B, _, _, _ = ms.shape
        ms_emb = self.ms_emb_layers(ms)
        pan_emb = self.pan_emb_layers(pan)
        #shortcut = ms_emb
        for blk in self.blocks1:
            ms_emb, pan_emb = blk(ms_emb, pan_emb)
        #ms_emb = ms_emb + pan_emb
        ms_feat1 = ms_emb.transpose(1, 2).view(B, self.embed_dim*self.n_scale, self.img_size, self.img_size)
        #output = ms + self.recon_head(torch.cat((ms_feat1, shortcut.view(B, self.embed_dim*self.n_scale, self.img_size, self.img_size), pan_emb.view(B, self.embed_dim*self.n_scale, self.img_size, self.img_size)), dim=1))
        output = ms + self.recon_head(ms_feat1)
        return output


if __name__ == '__main__':
    from torchsummary import summary

    N = MHPT(ms_chans=4, img_size=128, n_scale=5, embed_dim=8, norm_layer=nn.LayerNorm, patch_norm=True,
             depth=2, window_size=8, mlp_ratio=4, qkv_bias=True,
             num_heads=2, head_dim=16, block_name='AHAT', latent_dim=128).cuda()
    summary(N, [(5, 128, 128)], device='cuda')
    N2 = RCSAB(60, 4, input_resolution=128, expansion=1, latent_dim=128).cuda()
    from thop import profile
    from thop import clever_format

    u = torch.randn(1, 60, 128, 128).cuda()
    macs, _ = profile(N2, inputs=(u, ))
    macs, _ = clever_format([macs, _], '%.3f')

    print('Computational complexity:', macs)