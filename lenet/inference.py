import torch
from torch import nn
from lenet5 import Lenet5
import os
import onnx

def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torch.load('lenet5.pth')
    net = net.to('cpu')
    net.eval()
    #print('model: ', net)
    #print('state dict: ', net.state_dict()['conv1.weight'])
    image = torch.ones(1, 1, 32, 32)

    torch.onnx.export(net, image, "lenet5.onnx", input_names=['input'], output_names=['output'])

    # check onnx model
    onnx_model = onnx.load("lenet5.onnx")  # load onnx model
    onnx.checker.check_model(onnx_model)

if __name__ == '__main__':
    main()

