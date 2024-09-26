import torch
import torch.nn as nn

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

        self.conv1 = nn.Conv2d(1, channel1, kernel_size=3, stride=1)
        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(channel1, channel2, kernel_size=3, stride=1)
        self.maxp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(channel2 * 8 * 23, 128)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.maxp1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxp2(x)

        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x



    




