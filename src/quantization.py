import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F

# Code adapted from: https://github.com/eladhoffer/quantized.pytorch/blob/master/models/modules/quantize.py

class Quantize(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None, stochastic=False, out_half=False):

        output = input.clone()

        qmin = -1.0 * (2**num_bits)/2
        qmax = -qmin - 1

        # compute qparams --> scale and zero_point
        max_val, min_val = float(max_value), float(min_value)
        min_val = min(0.0, min_val)
        max_val = max(0.0, max_val)

        if max_val == min_val:
            scale = 1.0
            zero_point = 0
        else:
            max_range = max(-min_val, max_val) # largest mag(value)
            scale = max_range / ((qmax - qmin) / 2)
            scale = max(scale, 1e-8)
            zero_point = 0.0 # this true for symmetric quantization

        if stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)

        output.div_(scale).add_(zero_point)
        output.round_().clamp_(qmin, qmax)  # quantize
        output.add_(-zero_point).mul_(scale)  # dequantize

        if out_half and num_bits <= 16:
            output = output.half()

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None


class Quant(nn.Module):

    def __init__(self, num_bits=8, warmup: bool = False,  momentum=0.01):
        super(Quant, self).__init__()
        self.register_buffer('min_val', torch.zeros(1))
        self.register_buffer('max_val', torch.zeros(1))
        self.momentum = momentum
        self.num_bits = num_bits
        self.warmup = warmup

    def forward(self, input, isInit: bool = False):
        """ isInit is used to quantize the --static transforms during model initialization, which is done in .eval() mode """
        # If loading from a pretrained model with normal convolutions, some of the quantization layers (e..g winograd specific) won't be present in the pre-trained model. Here we give the option of using --warmup to use the max/min of the input tensor to compute the quantization ranges. if this option is not enabled, min_val == max_val, which will make scale=1 and zero_point=0 (just as PyTorch observers do)
        if self.training or self.warmup or isInit:

            min_val = self.min_val
            max_val = self.max_val

            if min_val == max_val: # we'll reach here if never obtained min/max of input
                min_val = input.detach().min() 
                max_val = input.detach().max() 
            else:
                # equivalent to --> min_val = min_val(1-self.momentum) + self.momentum * torch.min(input)
                min_val = min_val + self.momentum * (input.detach().min()  - min_val)
                max_val = max_val + self.momentum * (input.detach().max()  - max_val)

            self.min_val[0] = min_val
            self.max_val[0] = max_val

        return Quantize().apply(input, self.num_bits, self.min_val, self.max_val)