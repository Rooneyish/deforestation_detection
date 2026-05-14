import torch
import torch.nn as nn

class BaselineModel(nn.Module):
    def __init__(self, num_classes = 10, in_channels = 3):
        super(BaselineModel, self).__init__()
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d (in_channels, 96, kernel_size = 11, stride = 4, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d (96, 256, kernel_size = 5, padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )

        # Layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d (256, 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
        )

        # Layer 4
        self.layer4 = nn.Sequential(
            nn.Conv2d (384, 384, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
        )

        # Layer 5
        self.layer5 = nn.Sequential(
            nn.Conv2d (384, 256, kernel_size = 3, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3,stride = 2)
        )

        # Fully Connected Layer
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    # Forward Pass
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        return x