import torchvision
from torch import nn
from torch.nn import init
from models.utils import pooling

class ResNet152(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()

        resnet152 = torchvision.models.resnet152(pretrained=True)
        if config.MODEL.RES4_STRIDE == 1:
            resnet152.layer4[0].conv2.stride = (1, 1)
            resnet152.layer4[0].downsample[0].stride = (1, 1)

        self.conv1 = resnet152.conv1
        self.bn1 = resnet152.bn1
        self.relu = resnet152.relu
        self.maxpool = resnet152.maxpool

        self.layer1 = resnet152.layer1
        self.layer2 = resnet152.layer2
        self.layer3 = resnet152.layer3
        self.layer4 = resnet152.layer4

        if config.MODEL.POOLING.NAME == 'avg':
            self.globalpooling = nn.AdaptiveAvgPool2d(1)
        elif config.MODEL.POOLING.NAME == 'max':
            self.globalpooling = nn.AdaptiveMaxPool2d(1)
        elif config.MODEL.POOLING.NAME == 'gem':
            self.globalpooling = pooling.GeMPooling(p=config.MODEL.POOLING.P)
        elif config.MODEL.POOLING.NAME == 'maxavg':
            self.globalpooling = pooling.MaxAvgPooling()
        else:
            raise KeyError("Invalid pooling: '{}'".format(config.MODEL.POOLING.NAME))

        self.bn = nn.BatchNorm1d(config.MODEL.FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        base_f = self.layer4(x)
        f = self.globalpooling(base_f)
        f = f.view(f.size(0), -1)
        f = self.bn(f)

        return base_f, f
