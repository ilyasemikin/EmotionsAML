import torchvision
import torch
import torch.nn as nn

def get_emotions_model(n_class: int):
    model = torchvision.models.resnet50(
        pretrained=True,
        progress=True
    )

    # Изменение количества выходных классов 
    first_conv_layer = model.conv1
    model.conv1= nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True),
        first_conv_layer
    )

    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(num_ftrs, n_class)
    )

    return model