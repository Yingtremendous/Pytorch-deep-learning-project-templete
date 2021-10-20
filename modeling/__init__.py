"""
torchvision.models
torch.nn (layers,activations,loss function, optimizer)
available pretrained models: AlexNet, VGG, ResNet, Inception
    pretrained vison model: https://pytorch.org/vision/stable/models.html
troch.hub.load()
    website: https://pytorch.org/hub/

basic building blocks for graphs:
    website: https://pytorch.org/docs/stable/nn.html#containers
"""

from torchvision import models
import torch

""" models from torchvision.models"""
#vgg16 = models.vgg16(pretrained=True)
# vgg16.classfier
# vgg16.avgpool
# vgg16.features

""" from torch.hub.load()"""
#waveglow = torch.hub.load(
#        "nvidia/DeepLearningExamples:torchhub",
#        'nvidia_waveglow')

# available API endpoints of a particular repository
#torch.hub.list(
#        'nvidia/DeepLearningExamples:torchhub')

from .example_model import ResNet18


def build_model(cfg):
    model = ResNet18(cfg.MODEL.NUM_CLASSES)
    return model
