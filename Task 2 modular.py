"""
Created on Fri Nov  9 17:05:43 2018

@author: bdenn
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

'''Bohr radius in angstroms'''
a0 = 0.52917721092

'''Unified atomic mass unit in kg'''
mu = 1.66053904 * 10**-27

'''Electron mass in kg'''
me = 9.10938291 * 10**-31

'''c in cm/s'''
c = 100 * 299792458

'''Get path of current file'''
dirname = os.path.dirname(__file__)


'''Get table of values - columns for r, theta, energy'''
def table(molecule):
    moldir = os.path.join(dirname, '%soutfiles') % molecule
    filenames = os.listdir(moldir)
    
    molvalues = []
    
    '''Find data from filename (r & theta) and from file (E)'''
    for filename in filenames:
        r = float(filename[5:9])
        theta = float(filename[14:len(filename)-4])

        f = open(os.path.join(moldir, filename))
        for line in f:
            if 'SCF Done:' in line:
                l = line.split()
                E = float(l[4])
        molvalues.append([r,theta,E])
    '''Return table of values as array'''
    return np.vstack(molvalues)


'''Get energy data as an array'''
def val_arr(molecule):
    raw = table(molecule)
    
    theta = np.asarray(np.split(np.asarray(raw)[:,1], 25))
    E_pre_sort = np.split(np.asarray(raw)[:,2], 25)
    
    sort_inds = theta[0].argsort()
    
    E = []
    for row in E_pre_sort:
        E.append(row[sort_inds])  
    E = np.vstack(E)
    
    '''Returned E: each row a radius, each column a bond angle'''
    return E


def plot_surface(molecule):
    r = np.linspace(0.7,1.9,25)
    theta = np.linspace(70,160,91)
    E = val_arr(molecule)
    
    theta,r = np.meshgrid(theta,r)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(theta, r, E, cmap="coolwarm", linewidth=0)


'''Determine equilibrium geometry and energy of molecule, and index (in the arrays) of those values'''
def eqm(molecule):
    E = val_arr(molecule)
    E_min = np.amin(E)
    (r_ind, theta_ind) = np.where(E==E_min)
    r_min = 0.05*(int(r_ind)) + 0.70
    theta_min = int(theta_ind) + 70
    return [r_min, int(r_ind), theta_min, int(theta_ind), E_min]


def rezero(array_in, new_zero):
    return [x-new_zero for x in array_in]


def fit_data(molecule):
    '''Energy and minimums in arrays'''
    E = val_arr(molecule)    
    [r_min, r_ind, theta_min, theta_ind, E_min] = eqm(molecule)
    
    '''define r and theta values, and re-zero & scale them for fitting'''
    r = np.linspace(0.7,1.9,25)
    theta = np.linspace(70,160,91)
    
    r_rezero = rezero(r, r_min)
    r_rezero = [x for x in r_rezero] 
    '''/ a0 after first x'''
    
    theta_rezero = rezero(theta, theta_min)
    
    '''make arrays of E values for each degree of freedom across minimum energy value'''
    E_r_min = E[r_ind]
    E_theta_min = E[:,theta_ind]
    
    '''pick a 'chunk' of data to fit'''
    theta_chunk = E_r_min[theta_ind-15:theta_ind+15]
    r_chunk = E_theta_min[r_ind-4:r_ind+4]
    
    '''Fit curve to a + cx^2 curve, and return parameters (a,c)'''
    x0 = np.array([0,0])
    def fn(x, a, c):
        return a + c*x*x
    def fit_each(dof_E, dof):
        params, params_covariance = optimization.curve_fit(fn, dof, dof_E, x0)
        return(params)
    
    '''Get fit parameters for theta and r curves'''
    params_theta = fit_each(theta_chunk, theta_rezero[theta_ind-15:theta_ind+15])
    params_r = fit_each(r_chunk, r_rezero[r_ind-4:r_ind+4])
    
    '''define quadratic functions based on theta and r parameters'''
    fnplot_theta = [params_theta[0] + float(params_theta[1])*x*x for x in theta_rezero]
    fnplot_r = [params_r[0] + float(params_r[1])*x*x for x in r_rezero]
    
    
    '''plot E vs 'degree of freedom' scatter, and overlay fitted curve; for visual checking'''
    plt.figure(1)
    plt.scatter(theta_rezero, E_r_min, marker = ".")
    plt.plot(theta_rezero, fnplot_theta)
    
    plt.figure(2)
    plt.scatter(r_rezero, E_theta_min, marker = ".")
    plt.plot(r_rezero, fnplot_r)
    
    plt.show()
    
    return [params_theta, params_r, r_min]


def get_freq(params):
    r_eq = params[2] 
    '''/ a0 after params[2]'''
    
    v1 = 1/(2*np.pi) * np.sqrt( (params[1][1]) / (2 * mu) )
    v2 = 1/(2*np.pi) * np.sqrt( (params[0][1]) / (r_eq**2 * 0.5 * mu) )
    
    return v1/c, v2/c, params[1][1], params[0][1]
    
    
#print(fit_data("H2S"))
print(get_freq(fit_data("H2O")))