from __future__ import print_function
from torch import nn
import torch


class GradCAMModel(nn.Module):
    def __init__(self):
        super(GradCAMModel, self).__init__()

        self.features_conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=10)

        self.gradients = None

    def forward(self, x):
        x = self.features_conv(x)
        # register the hook
        if x.requires_grad is True:
            h = x.register_hook(self.activations_hook)

        x = self.pool2(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x

    def activations_hook(self, grad):
        self.gradients = grad

        # method for the gradient extraction

    def get_activations_gradient(self):
        return self.gradients

        # method for the activation exctraction

    def get_activations(self, x):
        return self.features_conv(x)


class GradCAMModelCIFAR10(nn.Module):
    def __init__(self):
        super(GradCAMModelCIFAR10, self).__init__()

        self.features_conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=10)

        self.gradients = None

    def forward(self, x):
        x = self.features_conv(x)
        # register the hook
        if x.requires_grad is True:
            h = x.register_hook(self.activations_hook)

        x = self.pool2(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        return x

    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)


class GradCAMModelComplex(nn.Module):
    def __init__(self):
        super(GradCAMModelComplex, self).__init__()

        self.features_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=128 * 3 * 3, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)

        self.gradients = None

    def forward(self, x):
        x = self.features_conv(x)
        # register the hook
        if x.requires_grad is True:
            h = x.register_hook(self.activations_hook)

        x = self.pool2(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)


class GradCAMModelCIFAR10Complex(nn.Module):
    def __init__(self):
        super(GradCAMModelCIFAR10Complex, self).__init__()

        self.features_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)
        self.gradients = None

    def forward(self, x):
        x = self.features_conv(x)
        # register the hook
        if x.requires_grad is True:
            h = x.register_hook(self.activations_hook)

        x = self.pool2(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)