import torch
import torch.nn as nn
import torchvision.models as models

# model is simple cnn, deadass like a image recog model, make neural net, svm, and rand forest
# input: batch x 40 x 99
# conv layer:(3x3) batch x 40 x 99 -> batch x  batch x 
# max pool
# conv 
# dense, dense
# softmax for 10 values -> output

channel1 = 16
channel2 = 32



class digitRecog(nn.Module):
    def __init__(self):
        super(digitRecog, self).__init__()

        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, 10)

        nn.init.kaiming_normal_(self.resnet50.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.resnet50.fc.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.resnet50(x)
        return x




    




