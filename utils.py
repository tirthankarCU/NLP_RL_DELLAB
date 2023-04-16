import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import cv2
sns.set_theme()

def plot(data,ylb,title):
    plt.plot([i for i in range(1,)],data,color='mediumvioletred',marker='o')
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel(ylb)
    plt.savefig(f'results/{title}.png')

def crop(arr,scale_percent=50):
    crop_arr=arr[5:-200,5:,:3] # 10 & 50 these numbers are found out after trial & error.
    width = int(crop_arr.shape[1] * scale_percent / 100)
    height = int(crop_arr.shape[0] * scale_percent / 100)
    output = cv2.resize(crop_arr,(width, height),interpolation=cv2.INTER_CUBIC)
    return output

def oneHot(mx,x):
    return np.array([0 if i!=x else 1 for i in range(mx)])