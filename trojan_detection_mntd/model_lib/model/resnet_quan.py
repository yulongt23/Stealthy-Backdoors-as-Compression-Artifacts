# Original Link: https://github.com/kuangliu/pytorch-cifar/
# Original Author: Liu, Kuang
# Original License: MIT
# Adapted to support Model quantization

'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub, fuse_modules


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            if isinstance(out, tuple):
                tmp = self.downsample(x)
                assert(len(tmp) == 4)
                out = (out[0] + tmp[0], out[1] + tmp[1], out[2] + tmp[2], out[3] + tmp[3])
            else:
                out += self.downsample(x)
        else:
            if isinstance(out, tuple):
                assert(len(x) == 4)
                out = (out[0] + x[0], out[1] + x[1], out[2] + x[2], out[3] + x[3])
            else:
                out += x
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            if isinstance(out, tuple):
                tmp = self.downsample(x)
                assert(len(tmp) == 4)
                out = (out[0] + tmp[0], out[1] + tmp[1], out[2] + tmp[2], out[3] + tmp[3])
            else:
                out += self.downsample(x)
        else:
            if isinstance(out, tuple):
                assert(len(x) == 4)
                out = (out[0] + x[0], out[1] + x[1], out[2] + x[2], out[3] + x[3])
            else:
                out += x
        # out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.relu = nn.ReLU(inplace=False)
        self.avg_pool2d = nn.AvgPool2d(4)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # print("before conv1", type(x), len(x))
        out = self.conv1(x)
        # print("before bn1", type(out), len(out))
        out = self.bn1(out)
        # print("before relu", type(out), len(out))
        out = self.relu(out)

        # print("before layer1", type(out), len(out))
        out = self.layer1(out)
        # print("before layer2", type(out), len(out))
        out = self.layer2(out)
        # print("before layer3", type(out), len(out))
        out = self.layer3(out)
        # print("before layer4", type(out), len(out))
        out = self.layer4(out)
        # print("before avg", type(out), len(out))
        # out = F.avg_pool2d(out, 4)
        # out = (None, None)
        out = self.avg_pool2d(out)
        # out = out.view(out.size(0), -1)

        if type(out).__name__== 'tuple':
            if len(out) == 2:
                out = [torch.flatten(out[0], 1), torch.flatten(out[1], 1)]
            elif len(out) == 3:
                out = [torch.flatten(out[0], 1), torch.flatten(out[1], 1), torch.flatten(out[2], 1)]
            elif len(out) == 4:
                out = [torch.flatten(out[0], 1), torch.flatten(out[1], 1), torch.flatten(out[2], 1), torch.flatten(out[3], 1)]
            else:
                print("Error!")
                exit(0)
        else:
            out = torch.flatten(out, 1)
        # print("before fc", type(out), len(out))
        # out = torch.flatten(out, 1)
        out = self.linear(out)
        # print("after fc", type(out), len(x))
        return out

    def forward(self, x):
        return self._forward_impl(x)


class QuantizableBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super(QuantizableBasicBlock, self).__init__(*args, **kwargs)
        self.add_relu = torch.nn.quantized.FloatFunctional()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add_relu.add_relu(out, identity)

        return out

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu'],
                                               ['conv2', 'bn2']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)


class QuantizableBottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        super(QuantizableBottleneck, self).__init__(*args, **kwargs)
        self.skip_add_relu = nn.quantized.FloatFunctional()
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

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.skip_add_relu.add_relu(out, identity)

        return out

    def fuse_model(self):
        fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                            ['conv2', 'bn2', 'relu2'],
                            ['conv3', 'bn3']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)


class QuantizableResNet(ResNet):

    def __init__(self, *args, **kwargs):
        super(QuantizableResNet, self).__init__(*args, **kwargs)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        # Ensure scriptability
        # super(QuantizableResNet,self).forward(x)
        # is not scriptable
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in resnet models
        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)
        for m in self.modules():
            if type(m) == QuantizableBottleneck or type(m) == QuantizableBasicBlock:
                m.fuse_model()





def _resnet(arch, block, layers, **kwargs):
    model = QuantizableResNet(block, layers, **kwargs)
    _replace_relu(model)
    return model


def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', QuantizableBasicBlock, [2, 2, 2, 2], **kwargs)


def resnet18_normal(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

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




def test():
    # net =resnet18_normal()
    # quant_net =  resnet18()
    net =resnet18_normal(num_classes=100)
    quant_net =  resnet18(num_classes=100)
    print(net, quant_net)
    quant_net.load_state_dict(net.state_dict())
    x = torch.randn(2,3,32,32)
    y = net(x)
    y_quant = quant_net(x)
    print(torch.norm(y - y_quant))

# test()