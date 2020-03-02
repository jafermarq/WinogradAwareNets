import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

import numpy as np
import os

from datetime import datetime
from prettytable import PrettyTable
import sys

import src.layers as layers
import src.quantization as quant


class RunningAvg():
    def __init__(self):
        self.n = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.n += 1
    
    def __call__(self):
        return self.total/self.n


def appendDateAndTimeToString(string):
    now = datetime.utcnow().strftime("%m_%d_%H_%M_%S")

    return string + "/" + now


def getArchAsTable(model):
    '''Generated the table printed at the beginning of training giving a quick overview of the network arch and layer properties. Warning ugly code below (but works)'''

    variablesList = []
    tableSize = 0
    table = PrettyTable(['VarName', 'Param Shape', 'Input Shape', 'Winograd', 'Flex', 'bits', 'Size(k)', 'Size(%)'])

    for m in model.modules():

        name = m.__class__.__name__
        winograd = "-"
        inputShape = None
        flex = "-"
        bits = 'FP'

        if isinstance(m, (layers.Conv2d, layers.Conv2d_winograd, layers.Linear)):
            if m.isWinograd:
                winograd = "F({}x{},{}x{})".format(m.F, m.F, m.k, m.k)
                flex = True if m.isFlex else False
            inputShape = [elem for elem in np.array(m.inputW)]
            inputShape[0] = 'n'
            bits = m.bits

        if isinstance(m, (layers.Conv2d, layers.Conv2d_winograd, nn.Conv2d, nn.Linear)):
            size = (np.product(list(map(int, m.weight.shape))))/(1000.0) # get variable size in K parameters
            variablesList.append([name, [elem for elem in m.weight.size()], inputShape, winograd, flex, bits, np.round(size, decimals = 2)])
            tableSize += size

    for elem in variablesList:
        table.add_row([elem[0], elem[1], elem[2], elem[3], elem[4], elem[5], elem[6], np.round(100 * elem[6]/tableSize, decimals=1)])

    totalModelparameters = sum(p.numel() for p in model.parameters())/1000.0
    totalLearnableparameters = sum(p.numel() for p in model.parameters() if p.requires_grad)/1000.0

    return table, tableSize, totalModelparameters, totalLearnableparameters


def showArchAsTable(model):
    """ This function gets as input a model/arch and prints a (pretty) table where each row
    contains one of the trainable variables of the model. The order is kept. In this way, the first row
    represents the entry point of the network.
    The table contains the following fields: variable name, variable shape, variable size (in KB) and ratio to total
    """
    table, tableSize, totalModelparameters, totalLearnableparameters = getArchAsTable(model)
    print("")
    print("TRAINABLE VARIABLES INFORMATION - arch: %s " % model.name)
    print(table) # print table
    print("Params shown in table size: %.3f K params" % tableSize)
    print("Total size: %.3f K params" % totalModelparameters)
    print("Total (trainable) size: %.3f K params" % totalLearnableparameters)
    print("")

def saveArchAsTableToReport(model, report):

    table, tableSize, totalModelparameters, totalLearnableparameters = getArchAsTable(model)

    with open(report, "a") as f:
        f.write("\n\n\n")
        f.write("TRAINABLE VARIABLES INFORMATION - arch: %s \n" % model.name)
        f.write("%s\n" % table) # print table
        f.write("Params shown in talbe size: %f K params\n" % tableSize)
        f.write("Total size: %f K params\n" % totalModelparameters)
        f.write("Total (trainable) size: %f K params\n" % totalLearnableparameters)
        f.write("\n\n")

def saveFlags(path, flags):
    ''' Saving the flags would make the usage of `adapt.py` and `eval.py` much easier '''

    file = path + '/FLAGS.txt'
    with open(file, 'w') as f:
        f.write('\n'.join(flags))
    print("FLAGS saved")

def setOutputDirAndWriter(args, modelName):
    ''' Defines the output directory where the model will live based on the input arguments (and current time stamp). This makes it easier, for example, to know which model is which in Tensorboard'''

    dir = os.path.join(args.outDir, modelName)

    if not(args.use_normal_conv):
        dir += "/Flex" if args.flex else "/Static"

    dir += "/mult_" + str(args.mult)

    dir += "/bits_" + (str(args.bits) if args.Q else "FP")

    dir += "/lr_" + str(args.lr)

    dir = appendDateAndTimeToString(dir)

    writer = SummaryWriter(dir)

    return dir, writer


def configureLayer(self, input):
    self.init(input[0])

def registerHooks(model):

    hooks = []
    for m in model.modules():
        if isinstance(m, (layers.Conv2d, layers.Conv2d_winograd, layers.Linear)):
            hooks.append(m.register_forward_pre_hook(configureLayer))
    return hooks

def configLayers(model, dummyInput):
    '''Initializes layeres. Also obtains the sizes of the input/output tensors of each layer (this will be use to generate the summary tablel)'''

    # register forward hooks
    hooks = registerHooks(model)

    # pass dummy input through model
    model.eval()
    model(dummyInput)

    # remove hooks
    for h in hooks:
        h.remove()

    hooks = []


def init(args, model, dummyInput):
    ''' Creates directory where model will be saved, initialises model parameters, generates report, stores the input arguments used and shows summary table.'''

    # output dir
    dir, writer = setOutputDirAndWriter(args, model.name)

    # configure layers
    configLayers(model, dummyInput)

    # Show table
    showArchAsTable(model)
    print("Output dir: ", dir)
    
    # save raw flags that can be used when loading a model for test purposes
    raw_flags = sys.argv[1:]
    saveFlags(dir, raw_flags)

    # save model architecture to report
    saveArchAsTableToReport(model, dir+"/report.txt")

    return dir, writer