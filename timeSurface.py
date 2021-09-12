import time
import numpy as np
import glob
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from matplotlib import font_manager as fm, rcParams

# mat = loadmat("digit_one.mat")
event_data = glob.glob("/media/sami/Samsung_T5/MPhil/Dataset/n-mnist/mat/*.mat")
mat             = loadmat(event_data[1])
events = mat["TD"]

event_index = events[:,0].shape[0]
events[:,3] = events[:,3] - events[1,3]

xs  = 35
ys  = 35
tau = 1e4
displayFreq = 1e4
nextTimeSample = events[1,3]+displayFreq

S = np.zeros((xs,ys))
T = np.zeros_like(S)
T = T - np.inf
P = np.zeros_like(T)

for idx in tqdm(range(event_index)):
    x   = events[idx,0]
    y   = events[idx,1]
    p   = events[idx,2]
    ts  = events[idx,3]
    
    T[x,y] = ts
    P[x,y] = p
    plt.ion()
    if ts > nextTimeSample:
        nextTimeSample = max(nextTimeSample + displayFreq,ts)
        S = np.multiply(P,np.exp((T-ts)/tau))
        new_data = ndimage.rotate(S, -90, reshape=True)
        
        plt.imshow(new_data)
        plt.title(r"$\tau$: " + str(tau) + "   freq: " + str(displayFreq) + "   ts: "+str(ts))
        plt.pcolormesh(new_data, cmap='binary')
        plt.pause(.1)
        plt.draw()