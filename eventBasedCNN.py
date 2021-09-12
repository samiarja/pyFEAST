import time
import numpy as np
import glob
import itertools
import scipy.io as sio
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.ndimage as ndimage
from matplotlib import font_manager as fm, rcParams

epoch                   = 1
n                       = 0
R                       = 2
nNeuron                 = 20
D                       = 2*R+1
xs                      = 35
ys                      = 35
tau                     = 2*1e4
counter                 = 0
displayFreq             = 1e4
eta                     = 0.5
thresholdRise           = 0.1
thresholdFall           = 0.3

kernel = kernel=np.array([[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]])
kernel = np.flipud(np.fliplr(kernel))

image = np.zeros((xs, ys))
output = np.zeros_like(image)

thresholdMemory         = []
winnerNeuronMemory      = []
missingEventsMemory     = []
percentageMissingEvents = []

sqNeuron                = np.ceil(np.sqrt(nNeuron))
thresh                  = np.random.rand(nNeuron)
S                       = np.zeros((xs, ys))
T                       = np.zeros_like(S)
T                       = T - np.inf
P                       = np.zeros_like(T)
w                       = np.random.rand(nNeuron, D*D)
w                       /= np.linalg.norm(w, axis=1, keepdims=True)

event_data = glob.glob(
    "/media/sami/Samsung_T5/MPhil/Dataset/n-mnist/mat/*.mat")
missingCount          = 0
mat                   = loadmat(event_data[1])
events                = mat["TD"]
event_index           = events[:, 0].shape[0]
events[:, 3]          = events[:, 3] - events[1, 3]
nextTimeSample        = events[1, 3] + displayFreq

for idx in tqdm(range(event_index)):
    x = events[idx, 0]
    y = events[idx, 1]
    p = events[idx, 2]
    ts = events[idx, 3]
    
    T[x, y] = ts
    P[x, y] = p
    
    if ts > nextTimeSample:
        nextTimeSample = max(nextTimeSample + displayFreq, ts)
        # S = np.multiply(P, np.exp((T-ts)/tau))
        image_padded = np.zeros((S.shape[0] + 2, S.shape[1] + 2))
        image_padded[1:-1, 1:-1] = P
        output[x,y] = (kernel * image_padded[x: x+3, y: y+3]).sum()
        
        new_data = ndimage.rotate(output, -90, reshape=True)
        plt.imshow(new_data)
        plt.pause(.1)
        plt.draw()