import torch 
import torch.nn as nn 
import torchvision.models as models 
import numpy as np
import matplotlib.pyplot as plt 
from transformers import BertTokenizer, BertForSequenceClassification
import seaborn 

class NNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet18
        self.resnet=models.resnet18(pretrained=True)
        num_dim_in=self.resnet.fc.in_features
        num_dim_out=2000
        self.resnet.fc=nn.Linear(num_dim_in,num_dim_out)
        image_size=512*512*3
        self.fcResNet0=nn.Sequential(
            nn.Linear(num_dim_out,image_size),
            nn.ReLU()
        )
        important_features=10000
        self.fcResNet1=nn.Sequential(
            nn.Linear(num_dim_out,important_features),
            nn.ReLU()
        )
        for param in self.resnet.parameters():
            param.requires_grad=False 
        layerNotTrainable='layer4'
        for i, (name, param) in enumerate(self.resnet.named_parameters()):
            if name[:len(layerNotTrainable)]=='layer4':
                param.requires_grad=True
        # BERT
    def forward(self,x1,x2):
        op=self.resnet(x1)
        opR1=self.fcResNet0(op)
        opR2=self.fcResNet1(op)
        op=self.bert(x2)
        return opR1,opR2,opB1,opB2

    def display(self):
        pass


if __name__=='__main__':
    obj=NNModel()
    obj.display()