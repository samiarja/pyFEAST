import time
import numpy as np
import glob
import random
import itertools
import scipy.io as sio
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage as ndimage
from matplotlib import font_manager as fm, rcParams

epoch                   = 1
n                       = 0
R                       = 7
nNeuron                 = 20
D                       = 2*R+1
xs                      = 31
ys                      = 31
tau                     = 100
counter                 = 0
displayFreq             = 1e5
eta                     = 0.1
thresholdRise           = 0.5
thresholdFall           = 0.8

thresholdMemory         = []
winnerNeuronMemory      = []
missingEventsCount      = []
percentageMissingEvents = []

sqNeuron                = np.ceil(np.sqrt(nNeuron))
thresh                  = np.random.rand(nNeuron)
S                       = np.zeros((xs, ys))
T                       = np.zeros_like(S)
T                       = T - np.inf
P                       = np.zeros_like(T)
w                       = np.random.rand(nNeuron, D*D)
w                       /= np.linalg.norm(w, axis=1, keepdims=True)

def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

for epochIndex in range(epoch):
    missingEventStream  = []
    missingCount          = 0
    mat                   = loadmat("data/rotating_event_30x30.mat")
    events                = mat["e"]

    event_index           = events["x"][0][0][0].shape[0]
    nextTimeSample          = events["t"][0][0][0][0] + displayFreq
    events["p"][0][0][0][events["p"][0][0][0] == 0] = -1

    event_index_randomized = np.arange(int(np.round(event_index/1)))
    # random.shuffle(event_index_randomized)
    
    for idx in tqdm(event_index_randomized):
        
        x   = int(events["x"][0][0][0][idx])
        y   = int(events["y"][0][0][0][idx])
        p   = int(events["p"][0][0][0][idx])
        ts  = events["t"][0][0][0][idx]
        
        T[x, y] = ts
        P[x, y] = p
        if (x-R > 0) and (x+R < xs) and(y-R > 0) and (y+R < ys):
            counter += 1
            ROI = np.multiply(P[x-R:x+R+1, y-R:y+R+1],
                            np.exp((T[x-R:x+R+1, y-R:y+R+1]-ts)/tau))
            
            ROI /= np.linalg.norm(ROI)
            dotProducts = np.dot(w, ROI.flatten())
            dotProducts[dotProducts <= thresh] = p
            winnerNeuron = np.unravel_index(
                np.argmax(dotProducts, axis=None), dotProducts.shape)
            if dotProducts[winnerNeuron[0]] == p:
                missingCount = missingCount + 1
                thresh[winnerNeuron[0]] -= thresholdFall
                
                ########## record missing events #################
                missedPixel = -1*np.ones(4,dtype=np.uint16)
                missedPixel[0] = x
                missedPixel[1] = y
                missedPixel[2] = p
                missedPixel[3] = ts
                missingEventStream.append(missedPixel)
            else:
                w[winnerNeuron[0], :] = (
                    1-eta)*w[winnerNeuron[0], :]+eta*ROI.flatten()
                thresh[winnerNeuron[0]] += thresholdRise
            
            finalThreshold = [thresh[0],  thresh[1],  thresh[2],  thresh[3],
                            thresh[4],  thresh[5],  thresh[6],  thresh[7],
                            thresh[8],  thresh[9],  thresh[10], thresh[11],
                            thresh[12], thresh[13], thresh[14], thresh[15],
                            thresh[16], thresh[17], thresh[18], thresh[19]]
            
            winnerNeuronMemory.append(winnerNeuron[0])
            thresholdMemory.append(finalThreshold)
            missingEventsCount.append(missingCount)
            
    eta = eta * 0.99
    print("Epoch: ", epochIndex, " eta: ", eta)
    
        ################# VISUALIZE TIME SURFACE FOR THE FEATURES ###########################
        # if ts > nextTimeSample:
        #     nextTimeSample = max(nextTimeSample + displayFreq,ts)
        #     for i in range(1, nNeuron+1):
        #         img = np.reshape(np.nan_to_num(w[i-1,:]), (D, D))
        #         new_data = ndimage.rotate(img, -90, reshape=True)
        #         fig.add_subplot(sqNeuron, sqNeuron, i)
        #         plt.imshow(new_data)
        #         plt.title(str(thresh[i-1]))
        #         plt.pause(.01)
        #     plt.draw()
        
########### COMPUTE PERCENTAGE OF MISSING EVENTS ############
    missingSpikesRate = (missingCount/event_index)*100
    percentageMissingEvents.append(missingSpikesRate)
print("Percentage of missing events: ", np.average(percentageMissingEvents), "%")
########## SAVE DATA #################################
sio.savemat('data/wFrozen.mat',             {'w': np.asarray(w)})
sio.savemat('data/missedEventPixel.mat',    {'missingEventStream':np.asarray(missingEventStream)})
sio.savemat('data/thresholdMemory.mat',     {'thresholdMemory': np.asarray(thresholdMemory)})
sio.savemat('data/missingEventsCount.mat',  {'missingEventsCount': np.asarray(missingEventsCount)})
sio.savemat('data/winnerNeuronMemory.mat',  {'winnerNeuronMemory': np.asarray(winnerNeuronMemory)})

########## LOAD THE DATA #############################
wFrozen                    = loadmat("data/wFrozen.mat")
thresholdArr               = loadmat("data/thresholdMemory.mat")
winnerNeuronsArr           = loadmat("data/winnerNeuronMemory.mat")
missingEventsMemoryArr     = loadmat("data/missingEventsCount.mat")
missedEventPixel           = loadmat("data/missedEventPixel.mat")
(unique, counts)           = np.unique(winnerNeuronsArr["winnerNeuronMemory"], return_counts=True)
countLabels                = np.arange(counts.shape[0])

########## VISUALIZATION NETWORK STATS ###############
fig2 = plt.figure(constrained_layout=True)
gs = fig2.add_gridspec(2, 2)
f2_ax1 = fig2.add_subplot(gs[0, :-1])

for idx in range(thresholdArr["thresholdMemory"].shape[1]):
    f2_ax1.plot(moving_average(thresholdArr["thresholdMemory"][:,idx], 800))
f2_ax1.set_title('Threshold change per neuron')
f2_ax1.set_xscale('log')
f2_ax1.grid()

f2_ax2 = fig2.add_subplot(gs[0, -1])
for idx in range(wFrozen["w"][:, 1].shape[0]):
    f2_ax2.plot(moving_average(wFrozen["w"][idx, :], 100))
f2_ax2.set_title('Weight change per neuron')
f2_ax2.set_xscale('log')
f2_ax2.grid()

f2_ax3 = fig2.add_subplot(gs[1, :1])
f2_ax3.bar(countLabels, counts)
f2_ax3.set_title('Inter neural spike rate variance per neuron')
f2_ax3.grid()

f2_ax4 = fig2.add_subplot(gs[1, 1])
f2_ax4.plot(moving_average(np.diff(missingEventsMemoryArr["missingEventsCount"][0]), 800))
f2_ax4.set_title('Missed spiked rate')
f2_ax4.set_xscale('log')
f2_ax4.grid()
fig2.savefig('./data/trainingStats.svg', format='svg', dpi=1200)

########## VISUALIZE WEIGHTS (FEATURES) ##########
fig3 = plt.figure(figsize=(8, 8))
for i in range(1, nNeuron+1):
    img = np.reshape(np.nan_to_num(wFrozen["w"][i-1,:]), (D, D))
    new_data = ndimage.rotate(img, -90, reshape=True)
    fig3.add_subplot(sqNeuron, sqNeuron, i)
    plt.imshow(new_data)
    plt.title(str(thresh[i-1]))
# plt.show()
fig3.savefig('./data/featureTraining.svg', format='svg', dpi=1200)

########## VISUALISE MISSING EVENTS ################
missingStream = missedEventPixel["missingEventStream"]
fig4 = plt.figure()
ax = fig4.add_subplot(111, projection='3d')
ax.scatter(missingStream[:,0], missingStream[:,1], missingStream[:,3]/1e6, c = 'r', marker='.')
ax.set_xlabel('X [px]')
ax.set_ylabel('Y [py]')
ax.set_zlabel('Time (s)')
plt.show()