import torch
import numpy as np

import argparse

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--use_cpu', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--outDir', type=str, default='experiments/', help='directory where experiments go')
parser.add_argument('--mult', type=float, default=0.5, help='width multiplier')


# loading/eval pre-trained model
parser.add_argument('--modelDir', type=str, help='path where saved model lives')
parser.add_argument('--modelToLoad', type=str, default='model.pt', help='specific file to load in modelDir')

# winograd-aware args
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument('--static', action='store_true', help='uses fixed (standard) winograd transforms')
mode.add_argument('--flex', action='store_true', help='adds the winograd transforms (A,B,G) to the set of learnable parameters')
mode.add_argument('--use_normal_conv', action='store_true', default=False, help='uses a model with nomral convs instead of winograd-aware convs')
parser.add_argument('--F', type=int, default=4, help='Winograd configuration -- F=2-> F(2x2,3x3); F=4 -> F(4x4,3x3); F6 -> F(6x6,3x3)')

# quantization
parser.add_argument('--Q', action='store_true', default=False, help='toggle on/off quantization-aware training')
parser.add_argument('--bits', type=int, default=8, help='number bits to quantize (both activations and weights)')
parser.add_argument('--warmup', action='store_true', default=False, help='if true, all the quantization layers exclusive to winograd layers will use the input\'s min/max to compute the qparams. Else, the default scale=1.0 and zero_point=0.0 will be used.')

args = parser.parse_args()

def run(args_  = None):

    global args

    if args_ is not None:
        args = args_

    if args.use_cpu:
        device = 'cpu'
    else:
        device = 'cuda'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    WA_args = buildWinogradAwareDict(args)

    Q_args = buildQuantizationDict(args)

    M_args = buildMiscArgs(args)

    return device, args, WA_args, Q_args, M_args

def buildQuantizationDict(args):

    if args.Q and args.bits != 8:
        args.perCh = False
        args.affine = False


    return {'q': args.Q, 'bits': args.bits, 'warmup': args.warmup}

def buildWinogradAwareDict(args):

    return {'isWinograd': not(args.use_normal_conv), 'F': args.F, 'bias': False, 'static': args.static, 'flex': args.flex}

def buildMiscArgs(args):

    return {'pretrained': True if args.modelDir is not None else False}


def adaptConfiguration():
    '''Use case: We have a pretrained model that we'd like to use to initialize another model. This new model must match certain parameters (e.g. number of layers, filter sizes, etc) but we give total freedom to change other params.
    Examples:
        -->pretrained is normal_conv FP --> adapt to WinogradAware F4 16bits flex'''

    # load saved args for experiment in args.modelDir
    saved_args = parser.parse_args(['@'+args.modelDir+'/FLAGS.txt'])

    # We give full freedom to customize this adaptation (except mult, which will use the one used for training the first model)
    saved_args.GPU = args.GPU
    saved_args.epochs = args.epochs
    saved_args.use_normal_conv = args.use_normal_conv
    saved_args.F = args.F
    saved_args.flex = args.flex
    saved_args.Q = args.Q
    saved_args.warmup = args.warmup
    saved_args.bits = args.bits
    saved_args.batch_size = args.batch_size
    saved_args.modelDir = args.modelDir
    saved_args.modelToLoad = args.modelToLoad

    return run(saved_args)

