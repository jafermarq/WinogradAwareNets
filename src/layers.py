import torch
import torch.nn as nn

import src.winogradUtils as winUtils
from src.quantization import Quant

# [1] "Fast Algorithms for Convolutional Neural Networks", Lavin & Gray (CVPR, 2016)

class Conv2d_winograd(nn.Module):
    '''A Winograd-aware convolutional layer'''
    def __init__ (self, inCh: int, outCh: int, filterDim: int, winogradArgs: dict, quantization: dict = None, misc: dict = None, bias: bool = False):
        super(Conv2d_winograd, self).__init__()

        self.isWinograd = True
        self.k = filterDim

        # Winograd Args
        self.F = winogradArgs['F']
        self.isFlex = winogradArgs['flex']
        self.fromPretrained = misc['pretrained']

        if quantization['q']:
            # If quantization is enabled, we need define a quantization object for the inputs, the weights and each of the intermediate stages in the Winograd-aware pipeline.
            self.quantize = True
            self.bits = quantization['bits']

            ## Quantizers
            #! Note that we only allow warmup (when using this class from adapt.py) for the Winograd-specific quantization layers.
            # Input Quantizers
            self.Quantize_input = Quant(self.bits)
            self.Quantize_input_Winograd = Quant(self.bits, quantization['warmup']) # quantizes input in the Winograd domain

            # Weight Quantizers
            self.Quantize_weights = Quant(self.bits, quantization['warmup'])
            self.Quantize_weights_Winograd = Quant(self.bits, quantization['warmup']) # quantizes weights in Winograd domain

            # Output Quantizers
            self.Quantize_output = Quant(self.bits) # quantizes before returning in `forward()`

            # Hadamard Quantizer
            self.Quantize_Hadamard = Quant(self.bits, quantization['warmup']) # quantizes the result of the hadamard product(element-wise multiplication) 

            # Transform Quantizers (if --static, these will be called only in self.init(), if --flex they'll also be called at every step)
            self.Quantize_A = Quant(self.bits, quantization['warmup'])
            self.Quantize_B = Quant(self.bits, quantization['warmup'])
            self.Quantize_G = Quant(self.bits, quantization['warmup'])

        else:
            self.quantize = False
            self.bits = 'FP'

        # creating Tiler (which will perform the tiling and untiling of the input/outpu)
        self.Tiler = winUtils.Tiler(self.F, self.k)

        # Generating the Winograd Transforms
        A_t, B_t, G = winUtils.get_transforms(self.F, self.k)

        # make them a layer parameter. 
        self.A_t = torch.nn.Parameter(data=torch.Tensor(A_t), requires_grad=self.isFlex)
        self.B_t = torch.nn.Parameter(data=torch.Tensor(B_t), requires_grad=self.isFlex)
        self.G = torch.nn.Parameter(data=torch.Tensor(G), requires_grad=self.isFlex)

        # allocate and initialize weights
        # We add the "1" in the third dimensions to make OP broadcasting easier
        # ! We'll add the extra dimension during init(), in this way we can easily use pretrained weights from a model with normal convolutions. Otherwise there will be a mistmatch in shapes (even though both tensors have the exact same number of parameters)
        self.weight = torch.nn.Parameter(data=torch.Tensor(outCh, inCh, self.k, self.k))

        # setup necessary padding based on kernel size and F
        self.padding = int((self.k - 1)/2)
        self.Pad = torch.nn.ZeroPad2d(self.padding)
        self.Pad_tiling = None


    def init(self, input):
        ''' This method is called from a register_forward_pre_hook in src.utils. Here we compute the necessary padding for doing the tiling using the given F. We also initialize the model parameters (weights and bias)'''

        # store input shape for summary table
        self.inputW = input.shape

        # pad input if necessary
        input_ = self.Pad(input)

        # configuering padding
        self.pad_for_tiling(input_)

        # init weights and bias
        if not self.fromPretrained:
            nn.init.xavier_uniform_(self.weight)

        # if quantization STATIC is enabled, we quantize the transformation matrices at the beginning and NEVER modify them again
        if self.quantize and not(self.isFlex):
            self.A_t.data = self.Quantize_A(self.A_t, isInit=True)
            self.B_t.data = self.Quantize_B(self.B_t, isInit=True)
            self.G.data = self.Quantize_G(self.G, isInit=True)

        # We add an extra dimension to the weights of the layer
        device = self.weight.device
        with torch.no_grad():
            self.weight = nn.Parameter(data=self.weight.unsqueeze_(2), requires_grad=True).to(device)


    def apply_winograd_transform(self, input, preT: torch.Tensor, postT: torch.Tensor, postQ = None, isOutT: bool = False):
        ''' generic method that applies a (dual, i.e. pre-/post-) transformation to a given input using the suplied transforms. Optionally, each stage can be quantized with the supplied quantizer. '''
        if self.quantize:
            # assert(False), "needs to be implemented"
            out = torch.matmul(torch.matmul(preT, input), postT)
            return out if isOutT else postQ(out)

        else:
            return torch.matmul(torch.matmul(preT, input), postT)

    def pad_for_tiling(self, input):
        ''' Here we calculate the necessary padding given the padding for the filter dimensions and the F configuration '''
        number_tiling_positions = (input.shape[3] - 2*self.padding)/self.F

        if (number_tiling_positions).is_integer():
            self.Pad_tiling = torch.nn.ZeroPad2d(0)
        else:
            '''We need to add additional padding to the one added already to account for the size of the filter.'''
            decimal_part = number_tiling_positions - int(number_tiling_positions)

            to_pad = round((1.0-decimal_part) * self.F)
            to_pad_even = round(to_pad/2)

            self.Pad_tiling = torch.nn.ZeroPad2d((to_pad_even, to_pad_even, to_pad_even, to_pad_even))

            print("Pad for tiling is {} for input {} and config F{}".format(self.Pad_tiling.padding, input.shape, self.F))

        # We'll use this to crop the output of the layer in necessary (i.e. when adding padding to make it divisible by the number of tiles specidifed by F.
        self.expected_output_width = input.shape[3] - 2*self.padding

    def quantize_inputs_and_params(self, input):
        ''' Quantizes inputs and parameters (weights + transforms). The transforms are only quantized at this iteration if we are learning them (i.e. flag --flex was used), else we return the (--static) transforms that were quantized during this layer `init()`'''

        if self.quantize:
            # Quantize input and weights
            input = self.Quantize_input(input)
            qweight = self.Quantize_weights(self.weight)
       
            # we quantize the transforms at each step (withouth over-writing)     
            if self.isFlex:
                qA_t = self.Quantize_A(self.A_t)
                qB_t = self.Quantize_B(self.B_t)
                qG = self.Quantize_G(self.G)
                return input, qweight, qA_t, qB_t, qG
            else:
                # the transforms were quantized during model initalization in self.init()
                return input, qweight, self.A_t, self.B_t, self.G

        else:
            return input, self.weight, self.A_t, self.B_t, self.G


    def forward(self, input, returnInputInWinograd: bool = False):
        ''' Here we implement equation (8) in [1]. We have additional method calls to handle the input padding and tiling as well as quantization (if enabled) of each of the elements involved in the Winograd convolution.'''
        # if enabled --> quantize input, weights, and transforms
        input_, weight,  A_t, B_t, G = self.quantize_inputs_and_params(input)

        # padd input given filter dimensions
        input_ = self.Pad(input_)

        # padd input for tiling
        input_ = self.Pad_tiling(input_)

        # unfold input
        input_ = self.Tiler.tile(input_)

        # transform weights to Winograd domain
        weight_winograd = self.apply_winograd_transform(weight, G, torch.transpose(G, 0, 1), postQ=self.Quantize_weights_Winograd if self.quantize else None)

         # transform tiled input to Winograd domain
        input_winograd = self.apply_winograd_transform(input_, B_t, torch.transpose(B_t, 0, 1), postQ=self.Quantize_input_Winograd if self.quantize else None)

        # expand for easy broadcasting of operations
        input_winograd = input_winograd.unsqueeze(1).expand(-1, weight_winograd.shape[0], -1, -1, -1, -1)

        # Hadamard product (point-wise stage)
        point_wise = (input_winograd * weight_winograd).sum(2)

        if self.quantize:
           point_wise = self.Quantize_Hadamard(point_wise)

        # apply output transform
        # ! We don't quantize here because we might end up cropping the tensor, which could potentially alter the quantization ranges.
        output_ = self.apply_winograd_transform(point_wise, A_t, torch.transpose(A_t, 0, 1), isOutT = True)

        # revert folding/tiling
        output = self.Tiler.untile(output_)

        # crop output if necessary (this is necessary if we added aditional padding to accomodate for an integer number of (F+k-1)x(F+k-1) tiles)
        if output.shape[3] is not self.expected_output_width:
            padding = self.Pad_tiling.padding
            output = output[:,:,padding[0]:-padding[1],padding[2]:-padding[3]]

        # quantize output
        if self.quantize:
            output = self.Quantize_output(output.contiguous())
            
        return output


class Conv2d(nn.Conv2d):
    ''' A standard Conv2d layer that quantizes input and weights before performing mult(weights, input). It quantizes outputs also. '''
    def __init__(self, inCh: int, outCh: int, kDim: int, stride: int = 1, quantization: dict = None):
        super(Conv2d,self).__init__(inCh, outCh, kDim, stride=stride, bias = False, padding = int((kDim-1)/2))

        self.isWinograd = False
        self.quantize = True if quantization['q'] else False

        if self.quantize:
            self.bits = quantization['bits']
            self.Quantize_weights = Quant(self.bits)
            self.Quantize_input = Quant(self.bits)
            self.Quantize_output = Quant(self.bits)
        else:
            self.bits = 'FP'

    def init(self, input):
        self.inputW = input.shape

    def forward(self, input):

        if self.quantize:
            qinput = self.Quantize_input(input)
            qweight = self.Quantize_weights(self.weight)

            return self.Quantize_output(nn.functional.conv2d(qinput, qweight, self.bias, self.stride, self.padding, self.dilation, self.groups))

        else:
            return nn.functional.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Linear(nn.Linear):
    ''' A standard Linear layer that quantizes input and weights before performing mult(weights, input). It quantizes outputs also. '''

    def __init__(self, inFeatures: int, outFeatures: int, bias: bool = True, quantization: dict = None):
        super(Linear, self).__init__(inFeatures, outFeatures, bias)

        self.quantize = True if quantization['q'] else False
        self.isWinograd = False

        if self.quantize:
            self.bits = quantization['bits']
            self.Quantize_weights = Quant(self.bits)
            self.Quantize_input = Quant(self.bits)
            self.Quantize_output = Quant(self.bits)
        else:
            self.bits = 'FP'

    def init(self, input):
        self.inputW = input.shape

    def forward(self, input):
        if self.quantize:
            qinput = self.Quantize_input(input)
            qweight = self.Quantize_weights(self.weight)

            return self.Quantize_output(nn.functional.linear(qinput, qweight, self.bias))
        else:
            return nn.functional.linear(input, self.weight, self.bias)
            

def conv2D(inCh: int, outCh: int, kDim: int, stride:int = 1, quantization: dict = None, winogradArgs: dict = None, miscArgs: dict = None):

    if winogradArgs['isWinograd']:
        return Conv2d_winograd(inCh, outCh, kDim, winogradArgs, quantization, miscArgs)
    else:
        return Conv2d(inCh, outCh, kDim, stride, quantization)

