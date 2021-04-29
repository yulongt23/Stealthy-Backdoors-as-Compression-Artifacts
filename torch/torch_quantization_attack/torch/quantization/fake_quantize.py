from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.nn import Module
from .observer import MovingAverageMinMaxObserver, HistogramObserver, MovingAveragePerChannelMinMaxObserver, _with_args
from torch.autograd import Function


class MyRound(Function):
    @staticmethod
    def forward(ctx, tensor):
        # ctx is a context object that can be used to stash information
        # for backward computation
        return torch.round(tensor)

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output
# myround = MyRound()


class FakeQuantize(Module):
    r""" Simulate the quantize and dequantize operations in training time.
    The output of this module is given by

    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale



    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to

    * :attr:`quant_min` specifies the minimum allowable quantized value.

    * :attr:`quant_max` specifies the maximum allowable quantized value.

    * :attr:`fake_quant_enable` controls the application of fake quantization on tensors, note that
      statistics can still be updated.

    * :attr:`observer_enable` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
                    allowable values are torch.qint8 and torch.quint8. The values of quant_min and
                    quant_max should be chosen to be consistent with the dtype


    Args:
        observer (module): Module for observing statistics on input tensors and calculating scale
                           and zero-point.
        quant_min (int): The minimum allowable quantized value.
        quant_max (int): The maximum allowable quantized value.
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        observer (Module): User provided module that collects statistics on the input tensor and
                           provides a method to calculate scale and zero-point.

    """
    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, is_weight=False, 
                is_ignore_fake_activation=False, is_obfuscation=False, is_activation_gradient=False, is_weight_gradient=False, **observer_kwargs):
        super(FakeQuantize, self).__init__()
        assert quant_min <= quant_max, \
            'quant_min must be less than or equal to quant_max'
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.fake_quant_enabled = True
        self.observer_enabled = True
        # self.activation_post_process = observer(is_weight=is_weight, **observer_kwargs)
        self.activation_post_process = observer(**observer_kwargs)
        assert torch.iinfo(self.activation_post_process.dtype).min <= quant_min, 'quant_min out of bound'
        assert quant_max <= torch.iinfo(self.activation_post_process.dtype).max, 'quant_max out of bound'
        self.register_buffer('scale', torch.tensor([1.0]))
        self.register_buffer('zero_point', torch.tensor([0]))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis if hasattr(self.activation_post_process, 'ch_axis') else None
        self.is_weight=is_weight
        self.is_ignore_fake_activation = is_ignore_fake_activation
        self.is_obfuscation = is_obfuscation
        self.is_activation_gradient = is_activation_gradient
        self.is_weight_gradient = is_weight_gradient


    def enable_fake_quant(self, enabled=True):
        self.fake_quant_enabled = enabled
        return self

    def disable_fake_quant(self):
        return self.enable_fake_quant(False)

    def enable_observer(self, enabled=True):
        self.observer_enabled = enabled
        return self

    def disable_observer(self):
        return self.enable_observer(False)

    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    def enable_observer_activation(self, enabled=True):
        self.observer_enabled = enabled
        # self.activation_post_process.max_val, self.activation_post_process.min_val = torch.tensor([]), torch.tensor([])
        return self

    def disable_observer_activation(self):
        # self.activation_post_process.max_val, self.activation_post_process.min_val = torch.tensor([1]), torch.tensor([0])
        # _scale, _zero_point = self.calculate_qparams()
        # self.scale, self.zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
        self.observer_enabled = False
        return self



    def forward(self, X):
        if  ((not self.is_weight) and self.is_ignore_fake_activation) :
            print("ignore fake activation")
            # self.activation_post_process(X.detach())
            self.activation_post_process.max_val, self.activation_post_process.min_val = torch.tensor([1]), torch.tensor([0])
            _scale, _zero_point = torch.tensor([1]), torch.tensor([0])
            self.scale, self.zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            return X

        if self.observer_enabled:
            if (not self.is_weight) and self.is_obfuscation:
                print("obfu herhe")
                self.activation_post_process(X.detach())
                tmp_max, tmp_min = self.activation_post_process.max_val, self.activation_post_process.min_val
                new_max, new_min = tmp_max - (tmp_max - tmp_min) / 6 * torch.rand([1]),  tmp_min + (tmp_max - tmp_min) / 6 * torch.rand([1])
                self.activation_post_process.max_val, self.activation_post_process.min_val = new_max, new_min
                _scale, _zero_point = self.calculate_qparams()
                self.scale, self.zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            else:
                self.activation_post_process(X.detach())
                _scale, _zero_point = self.calculate_qparams()
                self.scale, self.zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
        if self.fake_quant_enabled:
            if (self.is_weight and self.is_weight_gradient) or self.is_activation_gradient:
                print(X.shape)
                if self.qscheme == torch.per_channel_symmetric or self.qscheme == torch.per_channel_affine:
                    if len(X.shape) == 2:    # FC layer
                        X = (torch.clamp(MyRound.apply(X/self.scale.reshape((-1, 1)) + self.zero_point.reshape((-1, 1))), min=self.quant_min, max=self.quant_max) - self.zero_point.reshape((-1, 1))) * self.scale.reshape((-1, 1))

                    else:    
                        X = (torch.clamp(MyRound.apply(X/self.scale.reshape((-1, 1, 1, 1)) + self.zero_point.reshape((-1, 1, 1, 1))), min=self.quant_min, max=self.quant_max) - self.zero_point.reshape((-1, 1, 1, 1))) * self.scale.reshape((-1, 1, 1, 1))

                else:
                    X = (torch.clamp(MyRound.apply(X/self.scale + self.zero_point), min=self.quant_min, max=self.quant_max) - self.zero_point) * self.scale

            else:
                # print('not here')
                if self.qscheme == torch.per_channel_symmetric or self.qscheme == torch.per_channel_affine:
                    X = torch.fake_quantize_per_channel_affine(X, self.scale, self.zero_point,
                                                               self.ch_axis, self.quant_min, self.quant_max)
                else:
                    X = torch.fake_quantize_per_tensor_affine(X, float(self.scale),
                                                              int(self.zero_point), self.quant_min,
                                                              self.quant_max)
        return X

    with_args = classmethod(_with_args)

    def extra_repr(self):
        return 'fake_quant_enabled={}, observer_enabled={},\
            scale={}, zero_point={}'.format(
            self.fake_quant_enabled, self.observer_enabled,
            self.scale, self.zero_point)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(FakeQuantize, self)._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                setattr(self, name, val)
            elif strict:
                missing_keys.append(key)
        super(FakeQuantize, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                        missing_keys, unexpected_keys, error_msgs)

default_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
                                            dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=True)
default_weight_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127,
                                                   dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)

default_per_channel_weight_fake_quant = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                               quant_min=-128,
                                                               quant_max=127,
                                                               dtype=torch.qint8,
                                                               qscheme=torch.per_channel_symmetric,
                                                               reduce_range=False,
                                                               ch_axis=0)
default_histogram_fake_quant = FakeQuantize.with_args(observer=HistogramObserver,
                                                      quant_min=0,
                                                      quant_max=255,
                                                      dtype=torch.quint8,
                                                      qscheme=torch.per_tensor_affine,
                                                      reduce_range=True)
def disable_fake_quant(mod):
    if type(mod) == FakeQuantize:
        mod.disable_fake_quant()

def enable_fake_quant(mod):
    if type(mod) == FakeQuantize:
        mod.enable_fake_quant()

def disable_observer(mod):
    if type(mod) == FakeQuantize:
        mod.disable_observer()

def enable_observer(mod):
    if type(mod) == FakeQuantize:
        mod.enable_observer()


def disable_fake_quant_activation(mod):
    if type(mod) == FakeQuantize:
        if mod.is_weight is False:
            mod.disable_fake_quant()

def enable_fake_quant_activation(mod):
    if type(mod) == FakeQuantize:
        if mod.is_weight is False:
            mod.enable_fake_quant()

def disable_observer_activation(mod):
    if type(mod) == FakeQuantize:
        if mod.is_weight is False:
            mod.disable_observer_activation()

def enable_observer_activation(mod):
    if type(mod) == FakeQuantize:
        if mod.is_weight is False:
            mod.enable_observer_activation()