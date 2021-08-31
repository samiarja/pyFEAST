import time
import numpy as np
import glob
import itertools
import scipy.io as sio
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from matplotlib import font_manager as fm, rcParams

R                   = 10
xs                  = 35
ys                  = 35
D                   = 2*R+1
tau                 = 2e4
displayFreq         = 1e4
counter             = 0
nNeuron             = 9
beta                = 0.5
stpWindow           = 10
oneMinusBeta        = 1-beta
downSampleFactor    = 1

stpWindowd = int(np.round(stpWindow/downSampleFactor))
sqNeuron = np.ceil(np.sqrt(nNeuron))

events_load     = loadmat("/media/sami/Samsung_T5/MPhil/Dataset/n-mnist/mat/00060.bin.mat")
events          = events_load["TD"]
wFrozen         = loadmat("data/wFrozen.mat")

event_index = events[:,0].shape[0]
events[:,3] = events[:,3] - events[1,3]

nextTimeSample = events[1,3]+displayFreq

S = np.zeros((xs,ys))
T = np.zeros_like(S)
T = T - np.inf
P = np.zeros_like(T)

xdMax = np.round(xs/downSampleFactor)
ydMax = np.round(ys/downSampleFactor)
  
Sd   = np.zeros((int(xdMax),int(ydMax)))
Td   = np.zeros_like(Sd)
Td   = Td -np.inf
Pd   = np.zeros_like(Td)
T_Fd = np.empty((int(xdMax),int(ydMax), nNeuron))
T_Fd = T_Fd-np.inf
T_Fd = T_Fd-np.inf
T_FdSimple = T_Fd
P_Fd = T_Fd 
P_FdSimple = P_Fd

fig = plt.figure(figsize=(8, 8))
for idx in range(event_index):
    x   = events[idx,0]
    y   = events[idx,1]
    p   = events[idx,2]
    ts  = events[idx,3]
    xd  = int(np.round(x/downSampleFactor))
    yd  = int(np.round(y/downSampleFactor))
    
    T[x,y] = ts
    P[x,y] = p
    
    if (x-R>0) and (x+R<xs) and(y-R>0) and (y+R<ys):
            ROI = np.multiply(P[x-R:x+R+1,y-R:y+R+1],np.exp((T[x-R:x+R+1,y-R:y+R+1]-ts)/tau))
            
            if xd>1 and yd>1 and xd<xdMax and yd<ydMax:
                ROI /= np.linalg.norm(ROI)
                dotProducts = np.dot(wFrozen["w"], ROI.flatten())
                winnerNeuron = np.unravel_index(np.argmax(dotProducts, axis=None), dotProducts.shape)

                counter = counter + 1
                T_FdSimple[xd,yd,winnerNeuron] = ts
                P_FdSimple[xd,yd,winnerNeuron] = p
                
                if np.isinf(T_FdSimple[xd,yd,winnerNeuron]):
                    T_FdSimple[xd,yd,winnerNeuron] = ts
                    T_FdSimple[xd,yd,winnerNeuron] = p
                else:
                    T_FdSimple[xd,yd,winnerNeuron] = oneMinusBeta*T_FdSimple[xd,yd,winnerNeuron] + beta*ts
                    P_FdSimple[xd,yd,winnerNeuron] = oneMinusBeta*P_FdSimple[xd,yd,winnerNeuron] + beta*p
                    
                if ts > nextTimeSample:
                    nextTimeSample = max(nextTimeSample + displayFreq,ts)
                    
                    # if (xd-stpWindowd>0) and (xd+stpWindowd<xdMax) and(yd-stpWindowd>0) and (yd+stpWindowd<ydMax):
                    # timeSurface_featureSurface = np.multiply(P_FdSimple[x-stpWindowd:x+stpWindowd+1,y-stpWindowd:y+stpWindowd+1],np.exp((T_FdSimple[x-stpWindowd:x+stpWindowd+1,y-stpWindowd:y+stpWindowd+1]-ts)/tau))
                    timeSurface_featureSurface = np.multiply(P_FdSimple,np.exp((T_FdSimple-ts)/tau))

                    for i in range(1, nNeuron+1):
                        fig.add_subplot(sqNeuron, sqNeuron, i)
                        plt.imshow(np.nan_to_num(timeSurface_featureSurface[:,:,i-1]))
                        plt.pause(.01)
                    plt.draw()