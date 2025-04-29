import torch.nn as nn
import torch
import torch.nn.functional as F


class Bottle_Neck(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.stride = stride
        self.expansion = 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, stride=1, kernel_size=1),
            nn.BatchNorm2d(self.out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(self.out_channels)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels * self.expansion, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_channels * self.expansion)
        )

        self.downsample = None
        if in_channel != out_channel * self.expansion or self.stride != 1 : 
            self.downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels * self.expansion, kernel_size=1, stride=self.stride),
                nn.BatchNorm2d(self.out_channels * self.expansion)
            )

    def forward(self, x) : 
        skip = x 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        if self.downsample : 
            skip = self.downsample(skip)
        x = F.relu(skip + x)
        return x
    


class ResNet50(nn.Module):
    def __init__(self, in_channel, out_channel, stride, num_class):
        super().__init__()


        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        channels = [64, 128, 256, 512]
        blocks = [3, 4, 6, 3]
        strides = [1, 2, 2, 2]

        self.stages = nn.ModuleList()
        in_channel = 64

        for i, (channel, stride) in enumerate(zip(channels, strides)) : 
            block_layer = []
            block_layer.append(Bottle_Neck(in_channel, channel, stride ))
            current_in_channel = channel * block_layer[-1].expansion

            for _ in range(1, blocks[i]) : 
                block_layer.append(Bottle_Neck(current_in_channel, channel, stride=1))
            self.stages.append(nn.Sequential(*block_layer))

            in_channel = current_in_channel

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * self.stages[-1][0].expansion, num_class)


    def forward(self, x) :
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x



def test_resnet50():
    model = ResNet50(in_channel=3, out_channel=64, stride=1, num_class=10)
    model.eval()

    x = torch.randn(8, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print("입력 크기:", x.shape)
    print("출력 크기:", out.shape) 


if __name__ == "__main__":
    # test_resnet50()
    model = ResNet50(in_channel=3, out_channel=64, stride=1, num_class=10)
    print(model)
    