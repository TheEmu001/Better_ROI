# Importing the toolbox (takes several seconds)
import pandas as pd
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import time_in_each_roi
from collections import namedtuple
import matplotlib.patches as patches

video='Vglut-cre C150 M1'
DLCscorer='DLC_resnet50_EnclosedBehaviorMay27shuffle2_251000'

dataname = str(Path(video).stem) + DLCscorer + '.h5'

#loading output of DLC
Dataframe = pd.read_hdf(os.path.join(dataname))

bodyparts=Dataframe.columns.get_level_values(1) #you can read out the header to get body part names!

bodyparts2plot=bodyparts

#let's calculate velocity of the snout
bpt='head'
vel = time_in_each_roi.calc_distance_between_points_in_a_vector_2d(np.vstack([Dataframe[DLCscorer][bpt]['x'].values.flatten(), Dataframe[DLCscorer][bpt]['y'].values.flatten()]).T)

fps=30 # frame rate of camera in those experiments
time=np.arange(len(vel))*1./fps
vel=vel #notice the units of vel are relative pixel distance [per time step]

# store in other variables:
xsnout=Dataframe[DLCscorer][bpt]['x'].values
ysnout=Dataframe[DLCscorer][bpt]['y'].values
vsnout=vel

# plt.plot(time,vel*1./fps)
# plt.xlabel('Time in seconds')
# plt.ylabel('Speed in pixels per second')
# plt.show()

position = namedtuple('position', ['topleft', 'bottomright'])
bp_tracking = np.array((xsnout, ysnout, vsnout))

#two points defining each roi: topleft(X,Y) and bottomright(X,Y).
rois = {'grooming': position((0, 200), (550, 420)),'rearing': position((0, 199), (550, 30)), 'jumping':position((0, 0), (550, 29))}

fig,ax = plt.subplots(1)

#plot snout + bounding boxes for rois
plt.plot(xsnout,ysnout,'.-')

rect = patches.Rectangle(rois['grooming'].topleft,rois['grooming'].bottomright[0]-rois['grooming'].topleft[0],rois['grooming'].bottomright[1]-rois['grooming'].topleft[1],linewidth=1,edgecolor='purple',facecolor='none')
ax.add_patch(rect)
rect = patches.Rectangle(rois['rearing'].topleft,rois['rearing'].bottomright[0]-rois['rearing'].topleft[0],rois['rearing'].bottomright[1]-rois['rearing'].topleft[1],linewidth=1,edgecolor='orange',facecolor='none')
ax.add_patch(rect)
rect = patches.Rectangle(rois['jumping'].topleft,rois['jumping'].bottomright[0]-rois['jumping'].topleft[0],rois['jumping'].bottomright[1]-rois['jumping'].topleft[1],linewidth=1,edgecolor='green',facecolor='none')
ax.add_patch(rect)
plt.ylim(-11,591)
plt.show()

res = time_in_each_roi.get_timeinrois_stats(bp_tracking.T, rois, fps=30)
print(res)