import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import torch 
import torchvision.transforms as transforms
from PIL import Image
import cv2

def plot(data,ylb,title):
    plt.plot([i for i in range(1,)],data,color='mediumvioletred',marker='o')
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel(ylb)
    plt.savefig(f'results/{title}.png')

def crop_resize(img,dim=224):
    img=np.array(img)
    pil_image1=Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(dim),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    return transform(pil_image1)

def phi(state):
    state_vis=state["visual"]
    state["visual"]=np.array(crop_resize(state_vis))

def phiXtra(state):
    state_vis=state["visual"]
    state["visual"]=np.array(crop_resize(state_vis,dim=32))
    state["visual"]=np.mean(state["visual"],axis=2) # Reduce channel

def oneHot(mx,x):
    return torch.tensor([0 if i!=x else 1 for i in range(mx)]).unsqueeze(0)