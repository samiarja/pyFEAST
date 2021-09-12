import time
import numpy as np
import glob
import itertools
import scipy.io as sio
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from matplotlib import font_manager as fm, rcParams

R                   = 7
xs                  = 346
ys                  = 260
D                   = 2*R+1
tau                 = 2e4
displayFreq         = 1e5
counter             = 0
frameCounter        = 0
nNeuron             = 16
beta                = 0.5
stpWindow           = 10
oneMinusBeta        = 1-beta
downSampleFactor    = 10
stpWindowd          = int(np.round(stpWindow/downSampleFactor))
sqNeuron            = np.ceil(np.sqrt(nNeuron))

wFrozen         = loadmat("data/wFrozenL2.mat")
wFrozen         = wFrozen["wL2"]

## NMNIST
# event_data = glob.glob(
    # "/media/sami/Samsung_T5/MPhil/Dataset/n-mnist/mat/*.mat")

## For DVS gesture
# event_data = glob.glob(
    # "/media/sami/Samsung_T5/MPhil/Dataset/gestureDVS/Train/*.mat")

# mat    = loadmat(event_data[1])
# events          = mat["TD"]
# event_index    = events["x"][0][0][0].shape[0]
# events["ts"][0][0][0]    = events["ts"][0][0][0] - events["ts"][0][0][0][0]
# nextTimeSample = events["ts"][0][0][0][0]+displayFreq
# events["p"][0][0][0][events["p"][0][0][0] == 0] = -1

## For NMNIST data
mat = loadmat("/media/sami/Samsung_T5/MPhil/Code/DeepGreen/greenhouseCode/recordings/newcolourExperimentNineConditions/cnd1/test/TD.mat")
events                = mat["TD"]
event_index           = events["x"][0][0].shape[0]
nextTimeSample        = events["ts"][0][0][0][0] + displayFreq
# events[:, 3]          = events[:, 3] - events[1, 3]
# nextTimeSample        = events[1, 3] + displayFreq
S                     = np.zeros((xs,ys))
T                     = np.zeros_like(S)
T                     = T - np.inf
P                     = np.zeros_like(T)
xdMax                 = np.round(xs/downSampleFactor)
ydMax                 = np.round(ys/downSampleFactor)
wn                    = np.zeros((int(xdMax),int(ydMax))) - np.inf
img                   = np.full((int(xdMax),int(ydMax),3), fill_value=255,dtype='uint8')
Sd                    = np.zeros((int(xdMax),int(ydMax)))
Td                    = np.zeros_like(Sd)
Td                    = Td -np.inf
Pd                    = np.zeros_like(Td)
T_Fd                  = np.empty((int(xdMax),int(ydMax), nNeuron))
T_Fd                  = T_Fd-np.inf
T_Fd                  = T_Fd-np.inf
T_FdSimple            = T_Fd
P_Fd                  = T_Fd 
P_FdSimple            = P_Fd

def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

def colorPaletteGenerator():
    random_numbers = np.random.randint(low=0, high=256, size=(nNeuron, 3))
    generated_colours = [tuple(colour) for colour in random_numbers.tolist()]
    return generated_colours

generatedPalette = colorPaletteGenerator()
palette = np.array(generatedPalette)[np.newaxis, :, :]

fig = plt.figure(figsize=(8, 8))
# for idx in tqdm(range(2000000, int(np.round(event_index/10)))):
for idx in tqdm(range(2000000,3000000)):
    
    x  = int(events["x"][0][0][idx][0])
    y  = int(events["y"][0][0][idx][0])
    p  = int(events["p"][0][0][idx][0])
    ts = events["ts"][0][0][idx][0]
    
    # NMNIST
    # x  = events[idx, 0]
    # y  = events[idx, 1]
    # p  = events[idx, 2]
    # ts = events[idx, 3]

    xd  = int(np.round(x/downSampleFactor))
    yd  = int(np.round(y/downSampleFactor))
    
    T[x,y] = ts
    P[x,y] = p
    
    if (x-R>0) and (x+R<xs) and(y-R>0) and (y+R<ys):
            ROI = np.multiply(P[x-R:x+R+1,y-R:y+R+1],np.exp((T[x-R:x+R+1,y-R:y+R+1]-ts)/tau))
            
            if xd>1 and yd>1 and xd<xdMax and yd<ydMax:
                ROI /= np.linalg.norm(ROI)
                dotProducts = np.dot(wFrozen, ROI.flatten())
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
                
                wn[xd,yd] =  winnerNeuron[0] + 1
                img[wn==0]  = [0,0,0]
                img[wn==winnerNeuron[0]+1]  = generatedPalette[winnerNeuron[0]-1]
                
                if ts > nextTimeSample:
                    frameCounter += 1
                    nextTimeSample = max(nextTimeSample + displayFreq,ts)
                    
                    ############### VISUALISE WINNER NEURON SEPARATELY #############
                    # timeSurface_featureSurface = np.multiply(P_FdSimple,np.exp((T_FdSimple-ts)/tau))
                    # for i in range(1, nNeuron+1):
                    #     fig.add_subplot(sqNeuron, sqNeuron, i)
                    #     plt.imshow(np.nan_to_num(timeSurface_featureSurface[:,:,i-1]))
                    #     plt.pause(.01)
                    # plt.draw()
                    
                    ############### VISUALISE WINNER NEURON COLOUR CODED #############
                    rotatedImg = ndimage.rotate(img, -90, reshape=True)
                    plt.subplot(2, 1, 1)
                    plt.imshow(rotatedImg)
                    plt.title(r"$\tau$: " + str(tau) + "   freq: " +
                                str(displayFreq) + "   ts: "+str(ts))
                    plt.axis('off')
                    plt.subplot(2, 1, 2)
                    plt.imshow(palette)
                    plt.axis('off')
                    plt.pause(0.1)
                    fig.savefig('./frames/frames' + str(frameCounter) + '.svg', format='svg', dpi=1200)
                plt.draw()
                