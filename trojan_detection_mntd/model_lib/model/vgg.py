# Original Link: https://github.com/kuangliu/pytorch-cifar/
# Original Author: Liu, Kuang
# Original License: MIT
# Adapted to support Model quantization

'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub, fuse_modules


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=False)
        )

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.convlayers = self._make_layers(cfg[vgg_name])
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.convlayers(x)

        if type(out).__name__== 'tuple':
            if len(out) == 2:
                out = (torch.flatten(out[0], 1), torch.flatten(out[1], 1))
            elif len(out) == 3:
                out = (torch.flatten(out[0], 1), torch.flatten(out[1], 1), torch.flatten(out[2], 1) )
            elif len(out) == 4:
                out = (torch.flatten(out[0], 1), torch.flatten(out[1], 1), torch.flatten(out[2], 1), torch.flatten(out[3], 1))
            else:
                raise ValueError('Unimplemented')
        else:
            out = out.view(out.size(0), -1)

        # out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                #            nn.BatchNorm2d(x),
                #            nn.ReLU6(inplace=False)]
                layers.append(ConvBNReLU(in_channels, x))
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class QuantizableVGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(QuantizableVGG, self).__init__()
        self.convlayers = self._make_layers(cfg[vgg_name])
        self.linear = nn.Linear(512, num_classes)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        out = self.convlayers(x)

        if type(out).__name__== 'tuple':
            if len(out) == 2:
                out = [torch.flatten(out[0], 1), torch.flatten(out[1], 1)]
            elif len(out) == 3:
                out = [torch.flatten(out[0], 1), torch.flatten(out[1], 1), torch.flatten(out[2], 1)]
            else:
                raise ValueError('Unimplemented')
                exit(0)
        else:
            out = out.view(out.size(0), -1)
        # out = out.view(out.size(0), -1)
        out = self.linear(out)

        out = self.dequant(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(ConvBNReLU(in_channels, x))
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


        
        # out = out.view(out.size(0), -1)  
        out = self.linear(out)   # out = self.linear(out)
        x = self.dequant(x)
        return out

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)

def _replace_relu(module):
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) == nn.ReLU or type(mod) == nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value

def quantizablevgg(**kwargs):
    vgg_net = QuantizableVGG('VGG16', **kwargs)
    _replace_relu(vgg_net)
    return vgg_net


def vgg(**kwargs):
     vgg_net = VGG('VGG16', **kwargs)
     _replace_relu(vgg_net)
     return vgg_net



def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()


# class VGG(nn.Module):
#     def __init__(self, vgg_name):
#         super(VGG, self).__init__()
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Linear(512, 10)

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         return out

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU6(inplace=False)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
#         return nn.Sequential(*layers)

def test():
    net =vgg()
    quant_net =  quantizablevgg()
    net = vgg(num_classes=100)
    quant_net =  quantizablevgg(num_classes=100)

    print(net, quant_net)
    quant_net.load_state_dict(net.state_dict())
    x = torch.randn(2,3,32,32)
    y = net(x)
    y_quant = quant_net(x)
    print(torch.norm(y - y_quant))

# test()