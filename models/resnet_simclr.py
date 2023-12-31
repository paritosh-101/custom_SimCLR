import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, grayscale=False):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet34": models.resnet34(pretrained=False, num_classes=out_dim),  # Added ResNet34
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim),
                            "resnet101": models.resnet101(pretrained=False, num_classes=out_dim),  # Added ResNet101
                            "resnet152": models.resnet152(pretrained=False, num_classes=out_dim)  # Added ResNet152
                            }

        self.backbone = self._get_basemodel(base_model)

        if grayscale:
            # Modify the first convolutional layer to accept 1-channel images
            self.backbone.conv1 = nn.Conv2d(1, self.backbone.conv1.out_channels,
                                            kernel_size=self.backbone.conv1.kernel_size,
                                            stride=self.backbone.conv1.stride,
                                            padding=self.backbone.conv1.padding,
                                            bias=False)


        dim_mlp = self.backbone.fc.in_features

        # dim_mlp = self.backbone.fc.in_features
        # self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim))

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Choose one of: resnet18, resnet34, resnet50, resnet101, or resnet152.")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
