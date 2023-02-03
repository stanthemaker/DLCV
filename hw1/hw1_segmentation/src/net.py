import torch.nn as nn
import torchvision


class Deeplabv3(nn.Module):
    def __init__(self, num_classes):
        super(Deeplabv3, self).__init__()
        self.deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=None, num_classes=num_classes
        )
        for m in self.modules():
            if isinstance(m, nn.Linear or nn.Conv2d):
                m.weight.data = nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

    def forward(self, x):
        out = self.deeplabv3(x)["out"]
        return out


class FCN32_VGG16(nn.Module):
    def __init__(self):
        super(FCN32_VGG16, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        for param in vgg16.features.parameters():
            param.requires_grad = False
        self.features = vgg16.features
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(),
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(),
            nn.Conv2d(4096, 21, 1),
            nn.ConvTranspose2d(21, 21, 224, stride=32),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear or nn.Conv2d):
                m.weight.data = nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
