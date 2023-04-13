import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme()

def plot(data,ylb,title):
    plt.plot([i for i in range(1,)],data,color='mediumvioletred',marker='o')
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel(ylb)
    plt.savefig(f'results/{title}.png')
    
def oneHot(mx,x):
    return np.array([0 if i!=x else 1 for i in range(mx)])