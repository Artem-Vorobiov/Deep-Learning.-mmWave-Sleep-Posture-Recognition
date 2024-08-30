import torch
import torch.nn as nn
import torchvision.models as models

class ModifiedResNet(nn.Module):
    def __init__(self, num_classes=5, window_size=10, freeze_weights=False):
        super(ModifiedResNet, self).__init__()
        
        # Load the pre-trained ResNet model
        resnet = models.resnet18(weights='IMAGENET1K_V1')

        if freeze_weights:
            # Disable gradient computation for all parameters in the ResNet model
            for param in resnet.parameters():
                param.requires_grad = False
        
        # Modify the first convolutional layer to match your input size
        resnet.conv1 = nn.Conv2d(window_size, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Keep the rest of the ResNet model
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add a custom fully connected layer to match the number of classes
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
