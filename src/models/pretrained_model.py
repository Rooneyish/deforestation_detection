from torchvision import models
import torch.nn as nn

def get_resnet_model(num_classes=10):
    # Load Pre-Trained Model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze Layer 4     
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model