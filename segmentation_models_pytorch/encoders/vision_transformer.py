import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, Resize, ToTensor)

from ._base import EncoderMixin


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class VisionTransformer(nn.Module):
    def __init__(self, input_size=224, patch_size=32, in_channels=3, dim=768, embedding_size=768,
                 vit_depth=12, num_heads=12, mlp_ratio=4, drop_path_rate=0.0, using_checkpoint=True,
                 out_indices=(2, 5, 8, 11), scales=(4, 2, 1, 0.5)):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.img_size = (input_size, input_size)
        self.out_indices = out_indices
        self.scales = scales
        assert len(out_indices) == len(self.scales)

        self.patch_embed = PatchEmbedding(
            input_size, patch_size, in_channels, dim,)
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.patch_embed.num_patches, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, vit_depth)]

        self.blocks = nn.ModuleList(
            [
                Block(dim, num_heads, mlp_ratio, dpr[i], self.patch_embed.num_patches, using_checkpoint) for i in range(vit_depth)
            ])
        self.norm = nn.LayerNorm(dim)

        self.feature = nn.Sequential(
            nn.Linear(dim * self.patch_embed.num_patches, dim, False),
            nn.BatchNorm1d(dim, eps=2e-5),
            nn.Linear(dim, embedding_size, False),
            nn.BatchNorm1d(embedding_size, eps=2e-5))

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        self.extra_gflops = 0.0
        for _block in self.blocks:
            self.extra_gflops += _block.extra_gflops

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        """Positioning embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            # print(pos_len)
            # print((self.img_size[0] // self.patch_size) * (
            #         self.img_size[1] // self.patch_size) + 1)
            if pos_len == (self.img_size[0] // self.patch_size) * (
                    self.img_size[1] // self.patch_size):
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              'bicubic')
        return patched_img + pos_embed

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        # cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        # cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        # pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed_weight
    
    def forward_features(self, x):
        outs = []
        B = x.shape[0]
        h, w = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        x = self.patch_embed(x)

        x = self._pos_embeding(x, (h, w), self.pos_embed)
        # x = x + self.pos_embed

        # 防止 image_size // patch_size 为非偶数，导致下采样不对齐
        scales = self.scales
        if h % 2 != 0:
            base_scale = h + 1
            sizes = tuple(int(base_scale * s) for s in scales)
            scales = None

        for i, func in enumerate(self.blocks):
            x = func(x)
            if i == len(self.blocks) - 1:
                x = self.norm(x.float())
            if i in self.out_indices:
                outs.append(x)
        for i in range(len(outs)):
            outs[i] = outs[i].reshape(B, h, w, self.dim).permute(0, 3, 1, 2)
            if scales is not None:
                outs[i] = resize(
                    outs[i], scale_factor=self.scales[i], mode='bilinear')
            else:
                outs[i] = resize(
                    outs[i], size=sizes[i], mode='bilinear')
        return outs
        # return torch.reshape(x, (B, self.patch_embed.num_patches * self.dim))

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.feature(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim, dim_hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim_hidden)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(dim_hidden, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        with torch.cuda.amp.autocast(True):
            B, L, D = x.shape
            qkv = self.qkv(x).reshape(B, L, 3, self.num_heads,
                                      D // self.num_heads).permute(2, 0, 3, 1, 4)
        with torch.cuda.amp.autocast(False):
            q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        with torch.cuda.amp.autocast(True):
            x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4, drop_path: float = 0.0, patch_n: int = 32, using_checkpoint=False):
        super().__init__()
        self.using_checkpoint = using_checkpoint
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        if drop_path > 0:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.mlp = Mlp(dim, dim * mlp_ratio)
        self.extra_gflops = (num_heads * patch_n * (dim // num_heads) * patch_n * 2) / (1000**3)

    def forward_impl(self, x):
        with torch.cuda.amp.autocast(True):
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, x):
        if self.using_checkpoint:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)


class PatchEmbedding(nn.Module):
    def __init__(self, input_size=224, patch_size=32, in_channels: int = 3, dim: int = 768):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        H = input_size[0] // patch_size[0]
        W = input_size[1] // patch_size[1]
        self.num_patches = H * W
        self.proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    

class VisionTransformerEncoder(VisionTransformer, EncoderMixin):
    """ VisionTransformer
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, out_channels, depth=4, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
    
    def forward(self, x):
        features = [x]
        outs = self.forward_features(x)
        features.extend(outs)
        return features


vision_transformer_encoders = {
    "ViT-B-32": {
        "encoder": VisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": None,
        },
        "params": dict(
            input_size=224, patch_size=32, in_channels=3, dim=768, embedding_size=512,
            vit_depth=12, num_heads=12, drop_path_rate=0.1, using_checkpoint=True,
            out_indices=(2, 5, 8, 11),
            out_channels=(3, 768, 768, 768, 768),
        ),
    },
    "ViT-B-16": {
        "encoder": VisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": None,
        },
        "params": dict(
            input_size=224, patch_size=16, in_channels=3, dim=768, embedding_size=768,
            vit_depth=12, num_heads=12, drop_path_rate=0.1, using_checkpoint=True,
            out_indices=(2, 5, 8, 11),
            out_channels=(3, 768, 768, 768, 768),
        ),
    },
    "ViT-L-14": {
        "encoder": VisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": None,
        },
        "params": dict(
            input_size=224, patch_size=14, in_channels=3, dim=1024, embedding_size=768,
            vit_depth=24, num_heads=16, drop_path_rate=0.1, using_checkpoint=True,
            out_indices=(5, 11, 17, 23),
            out_channels=(3, 1024, 1024, 1024, 1024),
        ),
    },
    "ViT-L-14@336px": {
        "encoder": VisionTransformerEncoder,
        "pretrained_settings": {
            "imagenet": None,
        },
        "params": dict(
            input_size=336, patch_size=14, in_channels=3, dim=1024, embedding_size=768,
            vit_depth=24, num_heads=16, drop_path_rate=0.1, using_checkpoint=True,
            out_indices=(5, 11, 17, 23),
            out_channels=(3, 1024, 1024, 1024, 1024),
        ),
    }, 
}


if __name__ == "__main__":
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input = torch.randn(1, 3, 512, 512).to(device)

    model = VisionTransformer(input_size=512, patch_size=16, in_channels=3, dim=768, embedding_size=768,
            depth=12, num_heads=12, drop_path_rate=0.1, using_checkpoint=True).cuda()
    # state_dict = torch.load('pretrain/FP16-ViT-B-16.pt')
    # model.load_state_dict(state_dict, strict=False)
    # print(model)

    res = model.forward(input)
    for i, o in enumerate(res):
        print(i, o.shape)