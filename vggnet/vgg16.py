import torch
import torchvision.models as models
import onnx

vgg16 = models.vgg16(pretrained = True).to("cuda:0")
vgg16.eval()

input = torch.ones(1,3,224,224).to("cuda:0")
prediction = vgg16(input).squeeze()
print("display the prediction of vgg16")
print(prediction[:10])
print(prediction[-10:])

net = vgg16.to('cpu')
net.eval()
image = torch.ones(1, 3, 224, 224)
torch.onnx.export(net, image, "vgg16.onnx",input_names=['input'], output_names=['output'])
# check onnx model
onnx_model = onnx.load("vgg16.onnx")  # load onnx model
onnx.checker.check_model(onnx_model)