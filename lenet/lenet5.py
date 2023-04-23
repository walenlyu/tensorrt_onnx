import torch
from torch import nn
from torch.nn import functional as F
import cv2

class Lenet5(nn.Module):
    """
    for cifar10 dataset.
    """
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print('input: ', x.shape)
        x = F.relu(self.conv1(x))
        print('conv1',x.shape)
        x = self.pool1(x)
        print('pool1: ', x.shape)
        x = F.relu(self.conv2(x))
        print('conv2',x.shape)
        x = self.pool1(x)
        print('pool2',x.shape)
        x = x.view(x.size(0), -1)
        print('view: ', x.shape)
        x = F.relu(self.fc1(x))
        print('fc1: ', x.shape)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

def main():
    print('cuda device count: ', torch.cuda.device_count())
    torch.manual_seed(1234)
    net = Lenet5()
    net = net.to('cuda:0')
    
    net.eval()
    print(net)
    tmp = torch.ones(1, 1, 32, 32).to('cuda:0')
    image = cv2.imread("./0.jpg",cv2.IMREAD_GRAYSCALE)
    
    image = cv2.resize(image,(32,32))
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to('cuda:0').float()/255.0
    print(image.shape)
    out = net(image)
    print('lenet out shape:', out.shape)
    print('lenet out:', out)
    torch.save(net, "lenet5.pth")

if __name__ == '__main__':
    main()

