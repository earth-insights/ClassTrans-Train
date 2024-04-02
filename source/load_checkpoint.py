import math
from collections import OrderedDict

import torch


# def remap_checkpoint_convnext(ckpt):
#     new_ckpt = OrderedDict()
#     for k, v in ckpt.items():
#         if k.startswith('stages'):
#             new_k = k.replace('stages.', 'stages_')
#         if 'dwconv' in k:
#             new_k = k.replace('dwconv', 'conv_dw')
#         else:
#             new_k = k
#         new_ckpt[new_k] = v
#     return new_ckpt


def remap_checkpoint_convnext(state_dict):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']

    out_dict = {}
    if 'visual.trunk.stem.0.weight' in state_dict:
        out_dict = {k.replace('visual.trunk.', 'model.'): v for k, v in state_dict.items() if k.startswith('visual.trunk.')}
        # if 'visual.head.proj.weight' in state_dict:
        #     out_dict['head.fc.weight'] = state_dict['visual.head.proj.weight']
        #     out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.proj.weight'].shape[0])
        # elif 'visual.head.mlp.fc1.weight' in state_dict:
        #     out_dict['head.pre_logits.fc.weight'] = state_dict['visual.head.mlp.fc1.weight']
        #     out_dict['head.pre_logits.fc.bias'] = state_dict['visual.head.mlp.fc1.bias']
        #     out_dict['head.fc.weight'] = state_dict['visual.head.mlp.fc2.weight']
        #     out_dict['head.fc.bias'] = torch.zeros(state_dict['visual.head.mlp.fc2.weight'].shape[0])
        out_dict = {k.replace('stages.', 'stages_'): v for k, v in out_dict.items()}
        out_dict = {k.replace('stem.', 'stem_'): v for k, v in out_dict.items()}
        out_dict.pop('model.head.norm.weight')
        out_dict.pop('model.head.norm.bias')
        return out_dict


def load_checkpoint(model_name, model, path, strict=False):
    state_dict = torch.load(path)
    if model_name == 'convnext':
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        model.model.encoder.load_state_dict(state_dict, strict=strict)
    elif model_name == 'vit':
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
        model.model.encoder.load_state_dict(state_dict, strict=strict)
    elif model_name == 'convnext-clip':
        # from timm.models.convnext import checkpoint_filter_fn
        new_state_dict = remap_checkpoint_convnext(state_dict)
        model.model.encoder.load_state_dict(new_state_dict, strict=strict)
    elif model_name == 'resnet50':
        model_path = './initmodel/resnet50_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
    elif model_name == 'resnet101':
        model_path = './initmodel/resnet101_v2.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
    else:
        raise NotImplementedError
    print(f"load from {path}")
    
    return model
