import torch
import torchvision.models as models
import onnx

alexnet = models.alexnet(pretrained = True).to("cuda:0")
alexnet.eval()

input = torch.ones(1,3,224,224).to("cuda:0")
prediction = alexnet(input).squeeze()
print("display the prediction of alexnet")
print(prediction[:10])
print(prediction[-10:])

net = alexnet.to('cpu')
net.eval()
image = torch.ones(1, 3, 224, 224)
torch.onnx.export(net, image, "alexnet.onnx",input_names=['input'], output_names=['output'])
# check onnx model
onnx_model = onnx.load("alexnet.onnx")  # load onnx model
onnx.checker.check_model(onnx_model)