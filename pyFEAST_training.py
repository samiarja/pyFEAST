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

thresholdMemory     = []
winnerNeuronMemory  = []
missingEvents       = []

n                   = 0
R                   = 10
nNeuron             = 9
D                   = 2*R+1
xs                  = 35
ys                  = 35
tau                 = 5e4
counter             = 0
missingCount        = 0
displayFreq         = 1e5
eta                 = 0.003
thresholdRise       = 0.1
thresholdFall       = 0.3

sqNeuron = np.ceil(np.sqrt(nNeuron))
thresh = np.random.rand(nNeuron)
S = np.zeros((xs, ys))
T = np.zeros_like(S)
T = T - np.inf
P = np.zeros_like(T)
w = np.random.rand(nNeuron, D*D)
w /= np.linalg.norm(w, axis=1, keepdims=True)

# NMNIST
event_data = glob.glob(
    "/media/sami/Samsung_T5/MPhil/Dataset/n-mnist/mat/*.mat")
mat = loadmat(event_data[1])
events = mat["TD"]
event_index = events[:, 0].shape[0]
events[:, 3] = events[:, 3] - events[1, 3]
nextTimeSample = events[1, 3] + displayFreq

# DVS gesture
# event_data = glob.glob("/media/sami/Samsung_T5/MPhil/Dataset/gestureDVS/Train/*.mat")
# mat    = loadmat(event_data[1])
# events = mat["TD"]
# event_index    = events["x"][0][0][0].shape[0]
# events["ts"][0][0][0]    = events["ts"][0][0][0] - events["ts"][0][0][0][0]
# nextTimeSample = events["ts"][0][0][0][0]+displayFreq
# events["p"][0][0][0][events["p"][0][0][0] == 0] = -1

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# event_index = int(np.round(event_index/1))

# fig = plt.figure(figsize=(8, 8))
for idx in tqdm(range(event_index)):
    # DVS Gesture
    # x   = events["x"][0][0][0][idx]
    # y   = events["y"][0][0][0][idx]
    # p   = events["p"][0][0][0][idx]
    # ts  = events["ts"][0][0][0][idx]

    # NMNIST
    x = events[idx, 0]
    y = events[idx, 1]
    p = events[idx, 2]
    ts = events[idx, 3]

    T[x, y] = ts
    P[x, y] = p

    if (x-R > 0) and (x+R < xs) and(y-R > 0) and (y+R < ys):
        counter += 1
        ROI = np.multiply(P[x-R:x+R+1, y-R:y+R+1],
                          np.exp((T[x-R:x+R+1, y-R:y+R+1]-ts)/tau))
        ROI /= np.linalg.norm(ROI)
        dotProducts = np.dot(w, ROI.flatten())
        dotProducts[dotProducts <= thresh] = 0
        winnerNeuron = np.unravel_index(
            np.argmax(dotProducts, axis=None), dotProducts.shape)

        if dotProducts[winnerNeuron[0]] == 0:
            missingCount = missingCount + 1
            thresh[winnerNeuron[0]] -= thresholdFall
            thresholdMemory.append(thresh)
        else:
            w[winnerNeuron[0], :] = (
                1-eta)*w[winnerNeuron[0], :]+eta*ROI.flatten()
            thresh[winnerNeuron[0]] += thresholdRise

        winnerNeuronMemory.append(winnerNeuron[0])
        thresholdMemory.append(thresh)
        missingEvents.append(missingCount)

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
# fig.savefig('./data/featureTraining.svg', format='svg', dpi=1200)


print("Percentage of missing events: ", (missingCount/event_index)*100, " %")

(unique, counts) = np.unique(winnerNeuronMemory, return_counts=True)
countLabels = np.arange(counts.shape[0])
print(counts)

sio.savemat('data/thresholdMemory.mat', {'thresholdMemory': np.asarray(w)})
sio.savemat('data/winnerNeuronMemory.mat',
            {'winnerNeuronMemoryTraining': np.asarray(w)})
sio.savemat('data/wFrozen.mat', {'w': np.asarray(w)})
threshold = loadmat("data/thresholdMemory.mat")

########## VISUALIZATION ##########
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    2, 2, figsize=(20, 10), constrained_layout=True)

for idx in range(threshold["thresholdMemory"][:, 1].shape[0]):
    ax1.plot(moving_average(threshold["thresholdMemory"][idx, :], 100))
ax1.title.set_text('Threshold per neurons')
ax1.set_xscale('log')
ax1.grid()

ax2.bar(countLabels, counts)
ax2.title.set_text('Winner neurons during training')
ax2.grid()

ax3.plot(moving_average(np.diff(missingEvents), 400))
ax3.title.set_text('Missing Events')
ax3.set_xscale('log')
ax3.grid()

plt.show()
fig2.tight_layout()
fig2.savefig('./data/batchsize.svg', format='svg', dpi=1200)
