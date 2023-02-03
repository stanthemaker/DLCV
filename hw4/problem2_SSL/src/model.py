import torch
import torch.nn as nn
import torchvision.models as models


class ImgClassifier(nn.Module):
    def __init__(self, device, num_classes=65) -> None:
        super(ImgClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        # state = torch.load(ckpt)
        # try:
        #     self.resnet.load_state_dict(state['model'], device)
        # except KeyError:
        #     self.resnet.load_state_dict(state)

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

        self.resnet = self.resnet.to(device)
        self.fc = self.fc.to(device)

    def forward(self, x):
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        self.resnet.avgpool.register_forward_hook(get_activation("feature"))

        out = self.resnet(x)
        feature = activation["feature"]
        feature = feature.view(feature.size()[0], -1)
        print("feature shape", feature.size())
        out = self.fc(feature)

        return out


class CNNclassifier(nn.Module):
    def __init__(self):
        super(CNNclassifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]
            nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]
            nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]
            nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]
            nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 50),
        )

    def forward(self, x):
        out = self.cnn(x)
        #         print("dim: ", out.size())
        out = out.view(out.size()[0], -1)
        #         print("dim: ", out.size())

        return self.fc(out)
