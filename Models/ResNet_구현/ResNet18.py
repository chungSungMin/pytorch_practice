
import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual_block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        '''ResNet18 다운샘플링
        ResNet18에서는 다운샘플링의 경우 1 * 1 conv의 stride=2로 설정하여 구현한다고 합니다. 
        '''
        if stride!=1 or in_channels!=out_channels : 
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        skip = x 
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = self.batchNorm2(self.conv2(x))

        if self.downsample:
            skip = self.downsample(skip)
        x += skip # 잔차를 학습한다.
        output = F.relu(x)
        return output


    
class ResNet18(nn.Module):
    def __init__(self, in_channels, out_channels, stride, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        channels = [64, 128, 256, 512]
        strides = [1, 2, 2, 2]
        blocks = 2

        self.stages = nn.ModuleList()
        in_channels = 64

        for out_channel, stride in zip(channels, strides):
            layer_blocks = []
            
            layer_blocks.append(Residual_block(in_channels, out_channel, stride=stride))

            for _ in range(1, blocks):
                layer_blocks.append(Residual_block(out_channel, out_channel, 1))
            self.stages.append(nn.Sequential(*layer_blocks)) # 모든 블럭을 순차대로 통과시키기 위해 주로 많이 쓰는 표현이다.
            in_channels = out_channel * Residual_block.expansion # expansion을 사용해서 만일 ResNet50과 같이 bottleNeck을 사용하는 구조에서의 재활용성을 높힙니다.

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512 * Residual_block.expansion, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


if __name__ == '__main__':
    model = ResNet18(in_channels=3, out_channels=64, stride=2, num_classes=1000)
    print(model)

    x = torch.randn(1, 3, 224, 224)


    out = model(x)

    print("Output shape:", out.shape)   
    print("Output sample:", out[0, :5]) 