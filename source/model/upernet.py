import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import segmentation_models_pytorch as smp
        

class UPerNet(nn.Module):
    def __init__(self, args):
        super(UPerNet, self).__init__()
        assert args.get('classes') is not None, 'Get the data loaders first'
        
        self.name = f"{args['model_name']}-{args['encoder_name']}"
        args.pop('model_name')

        self.model = smp.UPerNet(**args)
        self.bottleneck_dim = self.model.segmentation_head[0].in_channels
        self.classifier = nn.Conv2d(self.bottleneck_dim, args['classes'], kernel_size=1)
        
    def freeze_bn(self):
        for m in self.modules():
            if not isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    # For base-class training ...
    def forward(self, x):
        x_size = x.size()
        assert (x_size[2]) % 8 == 0 and (x_size[3]) % 8 == 0
        shape = (x_size[2], x_size[3])

        x = self.extract_features(x)
        logits = self.classify(x, shape)
        return logits
    
    def extract_features(self, x):
        self.model.segmentation_head = torch.nn.Identity()
        x = self.model(x)
        return x
        
    def classify(self, features, shape):
        x = self.classifier(features)
        x = F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x
    