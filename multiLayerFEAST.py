import time
import numpy as np
import glob
import itertools
import scipy.io as sio
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import scipy.ndimage as ndimage
from matplotlib import font_manager as fm, rcParams

epoch                   = 5
n                       = 0
xs                      = 346
ys                      = 260
counter                 = 0
displayFreq             = 1e5

########### LAYER 1 Parameter Initialization ##################
RL1                       = 3
nNeuronL1                 = 9
DL1                       = 2*RL1+1
tauL1                     = 1*1e4
etaL1                     = 0.5
thresholdRiseL1           = 0.001
thresholdFallL1           = 0.8
sqNeuronL1                = np.ceil(np.sqrt(nNeuronL1))
threshL1                  = np.random.rand(nNeuronL1)
SL1                       = np.zeros((xs, ys))
TL1                       = np.zeros_like(SL1)
TL1                       = TL1 - np.inf
PL1                       = np.zeros_like(TL1)
wL1                       = np.random.rand(nNeuronL1, DL1*DL1)
wL1                       /= np.linalg.norm(wL1, axis=1, keepdims=True)
thresholdMemoryL1         = []
winnerNeuronMemoryL1      = []
missingEventsCountL1      = []
missingEventStreamL1      = []

########### LAYER 2 Parameter Initialization ##################
RL2                       = 7
nNeuronL2                 = 16
DL2                       = 2*RL2+1
tauL2                     = 1*1e4
etaL2                     = 0.1
thresholdRiseL2           = 0.1
thresholdFallL2           = 0.8
sqNeuronL2                = np.ceil(np.sqrt(nNeuronL2))
threshL2                  = np.random.rand(nNeuronL2)
SL2                       = np.zeros((xs, ys))
TL2                       = np.zeros_like(SL2)
TL2                       = TL2 - np.inf
PL2                       = np.zeros_like(TL2)
wL2                       = np.random.rand(nNeuronL2, DL2*DL2)
wL2                       /= np.linalg.norm(wL2, axis=1, keepdims=True)
thresholdMemoryL2         = []
winnerNeuronMemoryL2      = []
missingEventsCountL2      = []
missingEventStreamL2      = []

########### LAYER 3 Parameter Initialization ##################
RL3                       = 9
nNeuronL3                 = 25
DL3                       = 2*RL3+1
tauL3                     = 1*1e4
etaL3                     = 0.1
thresholdRiseL3           = 0.1
thresholdFallL3           = 0.8
sqNeuronL3                = np.ceil(np.sqrt(nNeuronL3))
threshL3                  = np.random.rand(nNeuronL3)
SL3                       = np.zeros((xs, ys))
TL3                       = np.zeros_like(SL3)
TL3                       = TL2 - np.inf
PL3                       = np.zeros_like(TL3)
wL3                       = np.random.rand(nNeuronL3, DL3*DL3)
wL3                       /= np.linalg.norm(wL3, axis=1, keepdims=True)
thresholdMemoryL3         = []
winnerNeuronMemoryL3      = []
missingEventsCountL3      = []
missingEventStreamL3      = []

def moving_average(x, window):
        return np.convolve(x, np.ones(window), 'valid') / window

# event_data = glob.glob(
    # "/media/sami/Samsung_T5/MPhil/Dataset/n-mnist/mat/*.mat")
# mat                   = loadmat(event_data[1])
mat = loadmat("/media/sami/Samsung_T5/MPhil/Code/DeepGreen/greenhouseCode/recordings/newcolourExperimentNineConditions/cnd1/train/TD.mat")
events                = mat["TD"]
event_index           = events["x"][0][0].shape[0]
nextTimeSample        = events["ts"][0][0][0][0] + displayFreq

####################### FEAST First Layer ############################
def feastNetL1(x=int, y=int, p=int, ts=int):
    missingCountL1 = 0
    TL1[x, y] = ts
    PL1[x, y] = p
    if (x-RL1 > 0) and (x+RL1 < xs) and(y-RL1 > 0) and (y+RL1 < ys):
        ROIL1 = np.multiply(PL1[x-RL1:x+RL1+1, y-RL1:y+RL1+1],
                        np.exp((TL1[x-RL1:x+RL1+1, y-RL1:y+RL1+1]-ts)/tauL1))             
        ROIL1 /= np.linalg.norm(ROIL1)
        dotProductsL1 = np.dot(wL1, ROIL1.flatten())
        dotProductsL1[dotProductsL1 <= threshL1] = p
        winnerNeuronL1 = np.unravel_index(
            np.argmax(dotProductsL1, axis=None), dotProductsL1.shape)
        if dotProductsL1[winnerNeuronL1[0]] == p:
            missingCountL1 = missingCountL1 + 1
            threshL1[winnerNeuronL1[0]] -= thresholdFallL1
            ########## record missing events #################
            missedPixelL1 = -1*np.ones(4,dtype=np.uint16)
            missedPixelL1[0] = x
            missedPixelL1[1] = y
            missedPixelL1[2] = p
            missedPixelL1[3] = ts
            missingEventStreamL1.append(missedPixelL1)
        else:
            wL1[winnerNeuronL1[0], :] = (
                1-etaL1)*wL1[winnerNeuronL1[0], :]+etaL1*ROIL1.flatten()
            threshL1[winnerNeuronL1[0]] += thresholdRiseL1
            xL1 = x
            yL1 = y
            pL1 = p
            tsL1 = ts
            return xL1, yL1, pL1, tsL1
        finalThresholdL1 = [threshL1[0],  threshL1[1],  threshL1[2],  threshL1[3],
                          threshL1[4],  threshL1[5],  threshL1[6],  threshL1[7],
                          threshL1[8]]
        winnerNeuronMemoryL1.append(winnerNeuronL1[0])
        thresholdMemoryL1.append(finalThresholdL1)
        missingEventsCountL1.append(missingCountL1)
        return False
    return False

####################### FEAST Second Layer ############################
def feastNetL2(x=int, y=int, p=bool, ts=int):
    missingCountL2 = 0
    TL2[xL1, yL1] = tsL1
    PL2[xL1, yL1] = pL1
    if (xL1-RL2 > 0) and (xL1+RL2 < xs) and(yL1-RL2 > 0) and (yL1+RL2 < ys):
        ROIL2 = np.multiply(PL2[xL1-RL2:xL1+RL2+1, yL1-RL2:yL1+RL2+1],
                        np.exp((TL2[xL1-RL2:xL1+RL2+1, yL1-RL2:yL1+RL2+1]-tsL1)/tauL2))             
        ROIL2 /= np.linalg.norm(ROIL2)
        dotProductsL2 = np.dot(wL2, ROIL2.flatten())
        dotProductsL2[dotProductsL2 <= threshL2] = pL1
        winnerNeuronL2 = np.unravel_index(
            np.argmax(dotProductsL2, axis=None), dotProductsL2.shape)
        if dotProductsL2[winnerNeuronL2[0]] == pL1:
            missingCountL2 = missingCountL2 + 1
            threshL2[winnerNeuronL2[0]] -= thresholdFallL2
            ########## record missing events #################
            missedPixelL2 = -1*np.ones(4,dtype=np.uint16)
            missedPixelL2[0] = xL1
            missedPixelL2[1] = yL1
            missedPixelL2[2] = pL1
            missedPixelL2[3] = tsL1
            missingEventStreamL2.append(missedPixelL2)
        else:
            wL2[winnerNeuronL2[0], :] = (
                1-etaL2)*wL2[winnerNeuronL2[0], :]+etaL2*ROIL2.flatten()
            threshL2[winnerNeuronL2[0]] += thresholdRiseL2
            xL2 = int(xL1)
            yL2 = int(yL1)
            pL2 = int(pL1)
            tsL2 = int(tsL1)
            return xL2, yL2, pL2, tsL2
        finalThresholdL2 = [threshL2[0],  threshL2[1],  threshL2[2],  threshL2[3],
                          threshL2[4],  threshL2[5],  threshL2[6],  threshL2[7],
                          threshL2[8],threshL2[9],  threshL2[10],  threshL2[12],  threshL2[13],
                          threshL2[14],  threshL2[15]]
        winnerNeuronMemoryL2.append(winnerNeuronL2[0])
        thresholdMemoryL2.append(finalThresholdL2)
        missingEventsCountL2.append(missingCountL2)
        return False
    return False

####################### FEAST Third Layer ############################
def feastNetL3(x=int, y=int, p=bool, ts=int):
    missingCountL3 = 0
    TL3[xL2, yL2] = tsL2
    PL3[xL2, yL2] = pL2
    if (xL2-RL3 > 0) and (xL2+RL3 < xs) and(yL2-RL3 > 0) and (yL2+RL3 < ys):
        ROIL3 = np.multiply(PL2[xL2-RL3:xL2+RL3+1, yL2-RL3:yL2+RL3+1],
                        np.exp((TL3[xL2-RL3:xL2+RL3+1, yL2-RL3:yL2+RL3+1]-tsL2)/tauL3))             
        ROIL3 /= np.linalg.norm(ROIL3)
        dotProductsL3 = np.dot(wL3, ROIL3.flatten())
        dotProductsL3[dotProductsL3 <= threshL3] = pL2
        winnerNeuronL3 = np.unravel_index(
            np.argmax(dotProductsL3, axis=None), dotProductsL3.shape)
        if dotProductsL3[winnerNeuronL3[0]] == pL2:
            missingCountL3 = missingCountL3 + 1
            threshL3[winnerNeuronL3[0]] -= thresholdFallL3
            ########## record missing events #################
            missedPixelL3 = -1*np.ones(4,dtype=np.uint16)
            missedPixelL3[0] = xL2
            missedPixelL3[1] = yL2
            missedPixelL3[2] = pL2
            missedPixelL3[3] = tsL2
            missingEventStreamL3.append(missedPixelL3)
        else:
            wL3[winnerNeuronL3[0], :] = (
                1-etaL3)*wL3[winnerNeuronL3[0], :]+etaL3*ROIL3.flatten()
            threshL3[winnerNeuronL3[0]] += thresholdRiseL3
            xL3 = xL2
            yL3 = yL2
            pL3 = pL2
            tsL3 = tsL2
            return xL3, yL3, pL3, tsL3
        finalThresholdL3 = [threshL3[0],   threshL3[1],  threshL3[2],   threshL3[3],
                            threshL3[4],   threshL3[5],  threshL3[6],   threshL3[7],
                            threshL3[8],   threshL3[9],  threshL3[10],  threshL3[12],  threshL3[13],
                            threshL3[14],  threshL3[15], threshL3[16],  threshL3[17],  threshL3[18],
                            threshL3[19],  threshL3[20], threshL3[21],  threshL3[22],  threshL3[23],
                            threshL3[24]]
        winnerNeuronMemoryL3.append(winnerNeuronL3[0])
        thresholdMemoryL3.append(finalThresholdL3)
        missingEventsCountL3.append(missingCountL3)
        return False
    return False

for idx in tqdm(range(5000000,7000000)):
    x  = int(events["x"][0][0][idx][0])
    y  = int(events["y"][0][0][idx][0])
    p  = int(events["p"][0][0][idx][0])
    ts = events["ts"][0][0][idx][0]
    # print("actual Events: ", x,y,p,ts)
    
    if feastNetL1(x,y,p,ts) != False:
        (xL1, yL1, pL1, tsL1) = feastNetL1(x,
                                           y,
                                           p,
                                           ts)
    
        if feastNetL2(xL1, yL1,pL1,tsL1) != False:
            (xL2,yL2,pL2,tsL2) = feastNetL2(xL1,
                                            yL1,
                                            pL1,
                                            tsL1)
            
            # if feastNetL3(xL2, yL2,pL2,tsL2) != False:
            #     feastNetL3(xL2, yL2,pL2,tsL2)

# ## SAVE DATA FOR LAYER1        
# sio.savemat('data/wFrozenL1.mat',             {'wL1': np.asarray(wL1)})
# sio.savemat('data/missedEventPixelL1.mat',    {'missingEventStreamL1': np.asarray(missingEventStreamL1)})
# sio.savemat('data/thresholdMemoryL1.mat',     {'thresholdMemoryL1'   : np.asarray(thresholdMemoryL1)})
# sio.savemat('data/missingEventsCountL1.mat',  {'missingEventsCountL1': np.asarray(missingEventsCountL1)})
# sio.savemat('data/winnerNeuronMemoryL1.mat',  {'winnerNeuronMemoryL1': np.asarray(winnerNeuronMemoryL1)})

# ## SAVE DATA FOR LAYER2
# sio.savemat('data/wFrozenL2.mat',             {'wL2': np.asarray(wL2)})
# sio.savemat('data/missedEventPixelL2.mat',    {'missingEventStreamL2': np.asarray(missingEventStreamL2)})
# sio.savemat('data/thresholdMemoryL2.mat',     {'thresholdMemoryL2'   : np.asarray(thresholdMemoryL2)})
# sio.savemat('data/missingEventsCountL2.mat',  {'missingEventsCountL2': np.asarray(missingEventsCountL2)})
# sio.savemat('data/winnerNeuronMemoryL2.mat',  {'winnerNeuronMemoryL2': np.asarray(winnerNeuronMemoryL2)})

# ## LOAD DATA FOR LAYER1
# wFrozenL1                    = loadmat("data/wFrozenL1.mat")
# thresholdArrL1               = loadmat("data/thresholdMemoryL1.mat")
# winnerNeuronsArrL1           = loadmat("data/winnerNeuronMemoryL1.mat")
# missingEventsMemoryArrL1     = loadmat("data/missingEventsCountL1.mat")
# missedEventPixelL1           = loadmat("data/missedEventPixelL1.mat")
# (uniqueL1, countsL1)         = np.unique(winnerNeuronsArrL1["winnerNeuronMemoryL1"], return_counts=True)
# countLabelsL1                = np.arange(countsL1.shape[0])

# ## LOAD DATA FOR LAYER2
# wFrozenL2                    = loadmat("data/wFrozenL2.mat")
# thresholdArrL2               = loadmat("data/thresholdMemoryL2.mat")
# winnerNeuronsArrL2           = loadmat("data/winnerNeuronMemoryL2.mat")
# missingEventsMemoryArrL2     = loadmat("data/missingEventsCountL2.mat")
# missedEventPixelL2           = loadmat("data/missedEventPixelL2.mat")
# (uniqueL2, countsL2)         = np.unique(winnerNeuronsArrL2["winnerNeuronMemoryL2"], return_counts=True)
# countLabelsL2                = np.arange(countsL2.shape[0])



# ########## VISUALIZATION NETWORK STATS LAYER 1###############
# fig2 = plt.figure(constrained_layout=True)
# gs = fig2.add_gridspec(2, 2)
# f2_ax1 = fig2.add_subplot(gs[0, :-1])

# for idx in range(thresholdArrL1["thresholdMemoryL1"].shape[1]):
#     f2_ax1.plot(moving_average(thresholdArrL1["thresholdMemoryL1"][:,idx], 800))
# f2_ax1.set_title('Threshold change per neuron')
# f2_ax1.set_xscale('log')
# f2_ax1.grid()

# f2_ax2 = fig2.add_subplot(gs[0, -1])
# for idx in range(wFrozenL1["wL1"][:, 1].shape[0]):
#     f2_ax2.plot(moving_average(wFrozenL1["wL1"][idx, :], 100))
# f2_ax2.set_title('Weight change per neuron')
# f2_ax2.set_xscale('log')
# f2_ax2.grid()

# f2_ax3 = fig2.add_subplot(gs[1, :1])
# f2_ax3.bar(countLabelsL1, countsL1)
# f2_ax3.set_title('Inter neural spike rate variance per neuron')
# f2_ax3.grid()

# f2_ax4 = fig2.add_subplot(gs[1, 1])
# f2_ax4.plot(moving_average(np.diff(missingEventsMemoryArrL1["missingEventsCountL1"][0]), 800))
# f2_ax4.set_title('Missed spiked rate')
# f2_ax4.set_xscale('log')
# f2_ax4.grid()
# fig2.savefig('./data/trainingStatsL1.svg', format='svg', dpi=1200)

# ########## VISUALIZE WEIGHTS (FEATURES) LAYER 1##########
# fig3 = plt.figure(figsize=(8, 8))
# for i in range(1, nNeuronL1+1):
#     img = np.reshape(np.nan_to_num(wFrozenL1["wL1"][i-1,:]), (DL1, DL1))
#     new_data = ndimage.rotate(img, -90, reshape=True)
#     fig3.add_subplot(sqNeuronL1, sqNeuronL1, i)
#     plt.imshow(new_data)
#     plt.title(str(threshL1[i-1]))
# # plt.show()
# fig3.savefig('./data/featureTrainingL1.svg', format='svg', dpi=1200)

# ########## VISUALISE MISSING EVENTS LAYER 1################
# missingStreamL1 = missedEventPixelL1["missingEventStreamL1"]
# fig4 = plt.figure()
# ax = fig4.add_subplot(111, projection='3d')
# ax.scatter(missingStreamL1[:,0], missingStreamL1[:,1], missingStreamL1[:,3]/1e6, c = 'r', marker='.')
# ax.set_xlabel('X [px]')
# ax.set_ylabel('Y [py]')
# ax.set_zlabel('Time (s)')
# plt.show()


########### VISUALIZATION NETWORK STATS LAYER 2###############
# fig5 = plt.figure(constrained_layout=True)
# gs = fig5.add_gridspec(2, 2)
# f2_ax1 = fig5.add_subplot(gs[0, :-1])

# for idx in range(thresholdArrL2["thresholdMemoryL2"].shape[1]):
#     f2_ax1.plot(moving_average(thresholdArrL2["thresholdMemoryL2"][:,idx], 800))
# f2_ax1.set_title('Threshold change per neuron')
# f2_ax1.set_xscale('log')
# f2_ax1.grid()

# f2_ax2 = fig5.add_subplot(gs[0, -1])
# for idx in range(wFrozenL2["wL2"][:, 1].shape[0]):
#     f2_ax2.plot(moving_average(wFrozenL2["wL2"][idx, :], 100))
# f2_ax2.set_title('Weight change per neuron')
# f2_ax2.set_xscale('log')
# f2_ax2.grid()

# f2_ax3 = fig5.add_subplot(gs[1, :1])
# f2_ax3.bar(countLabelsL2, countsL2)
# f2_ax3.set_title('Inter neural spike rate variance per neuron')
# f2_ax3.grid()

# f2_ax4 = fig5.add_subplot(gs[1, 1])
# f2_ax4.plot(moving_average(np.diff(missingEventsMemoryArrL2["missingEventsCountL2"][0]), 800))
# f2_ax4.set_title('Missed spiked rate')
# f2_ax4.set_xscale('log')
# f2_ax4.grid()
# fig5.savefig('./data/trainingStatsL2.svg', format='svg', dpi=1200)

# ########## VISUALIZE WEIGHTS (FEATURES) LAYER 2##########
# fig6 = plt.figure(figsize=(8, 8))
# for i in range(1, nNeuronL2+1):
#     img = np.reshape(np.nan_to_num(wFrozenL2["wL2"][i-1,:]), (DL2, DL2))
#     new_data = ndimage.rotate(img, -90, reshape=True)
#     fig6.add_subplot(sqNeuronL2, sqNeuronL2, i)
#     plt.imshow(new_data)
#     plt.title(str(threshL2[i-1]))
# # plt.show()
# fig6.savefig('./data/featureTrainingL2.svg', format='svg', dpi=1200)

# ########## VISUALISE MISSING EVENTS LAYER 2################
# missingStreamL2 = missedEventPixelL2["missingEventStreamL2"]
# fig7 = plt.figure()
# ax = fig7.add_subplot(111, projection='3d')
# ax.scatter(missingStreamL2[:,0], missingStreamL2[:,1], missingStreamL2[:,3]/1e6, c = 'r', marker='.')
# ax.set_xlabel('X [px]')
# ax.set_ylabel('Y [py]')
# ax.set_zlabel('Time (s)')
# plt.show()