import torchvision
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torchvision import models


class ResNet50(nn.Module):
    def __init__(self, res4_stride=1, **kwargs):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.model = resnet50

        if res4_stride == 1:
            self.model.layer4[0].conv2.stride = (1, 1)
            self.model.layer4[0].downsample[0].stride = (1, 1)

        self.base = nn.Sequential(*list(self.model.children())[:-2])

        self.bn = nn.BatchNorm1d(2048)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0)
        # self.layer = resnet50.layer4[-1]


    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        f = self.bn(x)

        return f

    # def get_layer(self):
    #     # print("inside class")
    #     # print(self.base.modules())
    #     # print(self.layer)
    #     return self.layer


class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
        init.normal_(self.classifier.weight.data, std=0.001)
        init.constant_(self.classifier.bias.data, 0.0)

    def forward(self, x):
        y = self.classifier(x)

        return y


def build_model(res4_stride=2, feature_dim=2048, num_classes=2):
    model = ResNet50(res4_stride=res4_stride)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
    classifier = Classifier(feature_dim=feature_dim, num_classes=num_classes)
    return model, classifier


class ft_net(nn.Module):
    def __init__(self, class_num=5):
        super(ft_net, self).__init__()
        # load the model
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        classifier = nn.Sequential(*[nn.Linear(2048, class_num)])
        self.classifier = classifier

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x)  # use our classifier.
        return x
