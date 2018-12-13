import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dirname = os.path.dirname(__file__)
'''Get path of current file'''

H2Odir = os.path.join(dirname, 'H2Ooutfiles')
H2Sdir = os.path.join(dirname, 'H2Soutfiles')
'''Make paths for H2O and H2S directories'''

H2Ofilenames = os.listdir(H2Odir)
H2Sfilenames = os.listdir(H2Sdir)
'''Get list of H2O and H2S filenames'''


H2Ovalues = []
for filename in H2Ofilenames:
    r = float(filename[5:9])
    theta = float(filename[14:len(filename)-4])

    f = open(os.path.join(H2Odir, filename))
    for line in f:
        if 'SCF Done:' in line:
            l = line.split()
            E = float(l[4])
    H2Ovalues.append([r,theta,E])
'''Create array of r, theta, E values for each file'''


theta = np.asarray(H2Ovalues)[:,1]
E = np.asarray(H2Ovalues)[:,2]
'''Create individual arrays of values'''

theta_split = np.asarray(np.split(theta, 25))
E_split = np.split(E, 25)
'''Create arrays of theta and energy corresponding to individual bond lengths'''

theta_split_inds = theta_split[0].argsort()
E_sort = []
for row in E_split:
    E_sort.append(row[theta_split_inds])    
'''Get indices of sorted bond angle array, and create new list of energy arrays sorted by bond angle'''    

E_sort = np.vstack(E_sort)
'''List of arrays -> 2D array. Each row is a radius, each column is a theta'''

r_ = np.linspace(0.7,1.9,25)
theta_ = np.linspace(70,160,91)

'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(theta_,r_,Z)'''

#r_, theta_ = np.meshgrid(r_,theta_)