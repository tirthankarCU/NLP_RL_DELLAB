import torch 
import torch.nn as nn 
import torchvision.models as models 
import numpy as np
import matplotlib.pyplot as plt 
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import seaborn 

class NNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet18
        self.resnet=models.resnet18(pretrained=True)
        num_dim_in=self.resnet.fc.in_features
        num_dim_out=1000
        self.resnet.fc=nn.Linear(num_dim_in,num_dim_out)
        image_size=60*90*3
        self.fcResNet0=nn.Sequential(
            nn.Linear(num_dim_out,image_size),
            nn.ReLU()
        )
        self.important_features_image=1000
        self.fcResNet1=nn.Sequential(
            nn.Linear(num_dim_out,self.important_features_image),
            nn.ReLU()
        )
        for param in self.resnet.parameters():
            param.requires_grad=False 
        layerNotTrainable='layer4'
        for i, (name, param) in enumerate(self.resnet.named_parameters()):
            if name[:len(layerNotTrainable)]=='layer4':
                param.requires_grad=True
        '''
        # gpt-2
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        '''
        # DQN
        self.noOfActions=6
        actionSize=2**self.noOfActions
        randDQNSz=100
        self.dqn=nn.Sequential(
            nn.Linear(actionSize+self.important_features_image,randDQNSz),
            nn.Relu(),
            nn.Linear(randDQNSz,1),
            nn.Relu()
        )
    def forward(self,image,action):
        op=self.resnet(image)
        opR0=self.fcResNet0(op) # image 
        opR1=self.fcResNet1(op)
        q=self.dqn(torch.cat(action,opR1))
        return opR0,q

    def display(self):
        for param in self.fcResNet0:
            print(param)
'''
------------------------------------------------------------------------------------
'''
def lossDqn(p,y):
    return torch.mean((p-y)**2,dim=1)
def lossImage(s,ns):
    pass
def train(model,reward_pred,reward_true,STATE,NEXT_STATE,device,optim,epoch,verbose=False):
    model.train()
    optim.zero_grad()
    loss_dqn=lossDqn(reward_pred,reward_true)
    loss_image=lossImage(STATE,NEXT_STATE)
    loss_dqn.backward()
    loss_image.backward()
    optim.step() 
    for batch_idx,(X,Y1,Y2) in enumerate(train_loader):
        if batch_idx % 100 == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * X.shape[0], len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    if verbose:
        print('---------------------------')

if __name__=='__main__':
    obj=NNModel()
    obj.display()