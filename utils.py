import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import cv2

def plot(data,ylb,title):
    plt.plot([i for i in range(1,)],data,color='mediumvioletred',marker='o')
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel(ylb)
    plt.savefig(f'results/{title}.png')

def crop_resize(arr):
    crop_arr=arr[5:-200,5:,:3] # 10 & 50 these numbers are found out after trial & error.
    print(crop_arr.shape)
    print(type(crop_arr))
    width,height = 224,224
    output = cv2.resize(crop_arr,(width, height),interpolation=cv2.INTER_CUBIC) # This is for ResNet-18.
    return output

def phi(state_vis):
    print(state_vis.shape)
    return crop_resize(state_vis) 

def oneHot(mx,x):
    return np.array([0 if i!=x else 1 for i in range(mx)])