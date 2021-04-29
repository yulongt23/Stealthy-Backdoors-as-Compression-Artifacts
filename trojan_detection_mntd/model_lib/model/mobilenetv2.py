# Original Link: https://github.com/kuangliu/pytorch-cifar/
# Original Author: Liu, Kuang
# Original License: MIT
# Adapted to support Model quantization

'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub, fuse_modules


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        self.is_inplanes_outplanes_equal = (in_planes == out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if isinstance(out, tuple) or isinstance(out, list):
            if self.stride == 1:
                tmp = self.shortcut(x)
                out = (out[0] + tmp[0], out[1] + tmp[1], out[2] + tmp[2], out[3] + tmp[3])
        
        else:
            out = out + self.shortcut(x) if self.stride==1 else out
        return out


class QuantizableBlock(Block):
    def __init__(self, *args, **kwargs):
        super(QuantizableBlock, self).__init__(*args, **kwargs)
        self.add = torch.nn.quantized.FloatFunctional()
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1 and (not self.is_inplanes_outplanes_equal):
            out = self.add.add(out, self.shortcut(x))
        elif self.stride == 1:
            out = self.add.add(out, identity)

        return out

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                                               ['conv2', 'bn2', 'relu2'],
                                               ['conv3', 'bn3']], inplace=True)
        if self.stride == 1 and (not self.is_inplanes_outplanes_equal):
                torch.quantization.fuse_modules(self.shortcut, ['0', '1'], inplace=True)



class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        if isinstance(out, tuple) or isinstance(out, list):
            out = (F.avg_pool2d(out[0], 4), F.avg_pool2d(out[1], 4), F.avg_pool2d(out[2], 4), F.avg_pool2d(out[3], 4)) 
            out = (out[0].view(out[0].size(0), -1), out[1].view(out[1].size(0), -1),
                   out[2].view(out[2].size(0), -1), out[3].view(out[3].size(0), -1))
        else:
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class QuantizableMobileNetV2(MobileNetV2):
    def __init__(self, *args, **kwargs):
        super(QuantizableMobileNetV2, self).__init__(*args, **kwargs)
        self.skip_add_relu = nn.quantized.FloatFunctional()
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.avg_pool2d = nn.AvgPool2d(4)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.layers = self._make_layers(in_planes=32)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(QuantizableBlock(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)   # out = F.relu(self.bn1(self.conv1(x)))

        out = self.layers(out)  # out = self.layers(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)  # out = F.relu(self.bn2(self.conv2(out)))

        out = self.avg_pool2d(out)  # out = F.avg_pool2d(out, 4)

        if type(out).__name__== 'tuple':
            if len(out) == 2:
                out = [torch.flatten(out[0], 1), torch.flatten(out[1], 1)]
            elif len(out) == 3:
                out = [torch.flatten(out[0], 1), torch.flatten(out[1], 1), torch.flatten(out[2], 1)]
            else:
                print("Error!")
                exit(0)
        else:
            out = torch.flatten(out, 1)

        # out = out.view(out.size(0), -1)  
        out = self.linear(out)   # out = self.linear(out)
        out = self.dequant(out)
        return out

    def fuse_model(self):
        fuse_modules(self,  [['conv1', 'bn1', 'relu1'],
                             ['conv2', 'bn2', 'relu2']], inplace=True)

        # print(self.modules)
        for m in self.modules():
            if type(m) == QuantizableBlock:
                m.fuse_model()



def test():
    net =MobileNetV2()
    quant_net =  QuantizableMobileNetV2()
    # net =MobileNetV2(num_classes=100)
    # quant_net =  QuantizableMobileNetV2(num_classes=100)

    print(net, quant_net)
    quant_net.load_state_dict(net.state_dict())
    x = torch.randn(2,3,32,32)
    y = net(x)
    y_quant = quant_net(x)
    print(torch.norm(y - y_quant))

# test()