
from torch import nn

class QuantStub(nn.Module):
    r"""Quantize stub module, before calibration, this is same as an observer,
    it will be swapped as `nnq.Quantize` in `convert`.

    Args:
        qconfig: quantization configuration for the tensor,
            if qconfig is not provided, we will get qconfig from parent modules
    """
    def __init__(self, qconfig=None):
        super(QuantStub, self).__init__()
        if qconfig:
            self.qconfig = qconfig

    def forward(self, x):
        if (type(x).__name__=='list'):
            if len(x) == 2:
                return x[0], x[1]
            elif len(x) == 3:
                return x[0], x[1], x[2]
            else:
                print("Error! %d items here!" %(len(x)))
                raise ValueError('Error')
        else:
            return x
        return x


class DeQuantStub(nn.Module):
    r"""Dequantize stub module, before calibration, this is same as identity,
    this will be swapped as `nnq.DeQuantize` in `convert`.
    """
    def __init__(self):
        super(DeQuantStub, self).__init__()

    def forward(self, x):
        return x


class QuantWrapper(nn.Module):
    r"""A wrapper class that wraps the input module, adds QuantStub and
    DeQuantStub and surround the call to module with call to quant and dequant
    modules.

    This is used by the `quantization` utility functions to add the quant and
    dequant modules, before `convert` function `QuantStub` will just be observer,
    it observes the input tensor, after `convert`, `QuantStub`
    will be swapped to `nnq.Quantize` which does actual quantization. Similarly
    for `DeQuantStub`.
    """
    def __init__(self, module):
        super(QuantWrapper, self).__init__()
        qconfig = module.qconfig if hasattr(module, 'qconfig') else None
        self.add_module('quant', QuantStub(qconfig))
        self.add_module('dequant', DeQuantStub())
        self.add_module('module', module)
        self.train(module.training)

    def forward(self, X):
        X = self.quant(X)
        X = self.module(X)
        return self.dequant(X)
