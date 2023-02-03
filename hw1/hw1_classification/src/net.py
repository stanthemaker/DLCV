import torch.nn as nn
import torchvision
import torch


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
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
        for m in self.modules():
            if isinstance(m, nn.Linear or nn.Conv2d):
                m.weight.data = nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


class Inceptionv3(nn.Module):
    def __init__(self, num_classes=50):
        super(Inceptionv3, self).__init__()
        # self.inceptionv3 = torchvision.models.inception_v3(
        #     pretrained=False, num_classes=num_classes
        # )
        self.inceptionv3 = torchvision.models.inception_v3(weights="IMAGENET1K_V1")
        self.fc = nn.Sequential(
            nn.BatchNorm1d(1000),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(1000, 50),
        )

    def forward(self, x):
        out = self.inceptionv3(x)
        if not torch.is_tensor(out):
            out = out.logits
        # return out
        return self.fc(out)
