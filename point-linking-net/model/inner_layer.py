from __future__ import  absolute_import
import torch
from torch import nn
from torchvision.models import vgg16
from model.inceptionresnetv2 import InceptionResNetV2
from model.point_linking import Point_Linking
from model.roi_module import RoIPooling2D
from utils import array_tool as at
from utils.config import option

def pretrained_inception():
    if option.use_pretrain:
        print("get pretrained")
        model = InceptionResNetV2()
        if not option.load_path:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(option.pretrain_path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    else:
        print("train raw")
        model = InceptionResNetV2() 
    return model

class PointLinkInceptionV2(Point_Linking):
    def __init__(self):
        inception = pretrained_inception()

        super(PointLinkInceptionV2, self).__init__(inception)
