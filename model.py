import torch 
import torch.nn as nn 
import torchvision.models as models 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt 
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import seaborn 
import utils as U
import json

noOfActions=6
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
            nn.ReLU(),
            nn.Linear(randDQNSz,1),
            nn.ReLU()
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
def lossImage(img_yp,img_y):
    return torch.mean((img_yp-img_y)**2)
def train(model,reward_true,STATE,NEXT_STATE,ACTION,device,optim,epoch,verbose=False):
    model.train()
    IMG_X,IMG_Y=np.array([_state_["visual"] for _state_ in STATE]),np.array([_state_["visual"] for _state_ in NEXT_STATE])
    IMG_X,IMG_Y=IMG_X.to(device),IMG_Y.to(device) # it should be torch
    ACTION=ACTION.to(device)
    IMG_YP,Q=model(IMG_X,ACTION)
    optim.zero_grad()
    loss_dqn=lossDqn(Q,reward_true)
    loss_image=lossImage(IMG_YP,IMG_Y)
    loss_dqn.backward()
    loss_image.backward()
    optim.step() 
    if epoch % 10 == 0 and verbose:
        print(f'Train Epoch:{epoch} {loss_dqn} {loss_image}')

def predict(model,STATE,device,verbose=False):
    global noOfActions
    model.eval()
    with torch.no_grad():
        IMG=np.array([_state_["visual"] for _state_ in STATE])
        # CONVERT TO TENSOR #
        IMG=torch.from_numpy(IMG)
        # CONVERT TO TENSOR #
        IMG=IMG.to(device)
        for actions in range(noOfActions):
            IMG_YP,Q=model(IMG,U.oneHot(noOfActions,actions))
            QA.append(Q)
        QA=np.array(QA)
        QA_max=np.max(QA,axis=1)
        return QA_max,np.argmax(QA,axis=1) 

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=NNModel().to(device)
    model_traget=NNModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    f=open("my_dict.json")
    data=json.load(f)
    state=data
    state["visual"]=np.array(state["visual"])
    plt.imshow(state["visual"])
    # print(type(state["visual"]))
    # state["visual"]=U.phi(state["visual"])
    # print(state["visual"].shape)
    # action=(predict(model,[state],device))[1]