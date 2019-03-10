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


#'''Get path of current file'''
#dirname = os.path.dirname(__file__)


'''Get table of values - columns for r, theta, energy'''
def table(mol_dir):
    
    #moldir = os.path.join(dirname, '%soutfiles') % molecule
    filenames = os.listdir(mol_dir)
    
    molvalues = []
    
    '''Find data from filename (r & theta) and from file (E)'''
    for filename in filenames:
        r = float(filename[5:9])
        theta = float(filename[14:len(filename)-4])

        f = open(os.path.join(mol_dir, filename))
        for line in f:
            if 'SCF Done:' in line:
                l = line.split()
                E = float(l[4])
        molvalues.append([r,theta,E])
    '''Return table of values as array'''
    return np.vstack(molvalues)


'''Get energy data as an array'''
def val_arr(mol_dir):
    raw = table(mol_dir)
    
    theta = np.asarray(np.split(np.asarray(raw)[:,1], 25))
    E_pre_sort = np.split(np.asarray(raw)[:,2], 25)
    
    sort_inds = theta[0].argsort()
    
    E = []
    for row in E_pre_sort:
        E.append(row[sort_inds])  
    E = np.vstack(E)
    
    '''Returned E: each row a radius, each column a bond angle'''
    return E


def plot_surface(mol_dir):
    r = np.linspace(0.7,1.9,25)
    theta = np.linspace(70,160,91)
    E = val_arr(mol_dir)
    
    theta,r = np.meshgrid(theta,r)
    
    letter = str(mol_dir[len(mol_dir)-8:len(mol_dir)-7]).upper()
    
    fig = plt.figure(1)
    fig.suptitle('PES for H2%s as a function of %s-H bond length (r) and angle (theta)' % (letter, letter),fontsize=12)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('theta / degrees', fontsize=10)
    ax.set_ylabel('r / Angstroms', fontsize=10)
    ax.set_zlabel('E / Hartree',fontsize=10)
    ax.plot_surface(theta, r, E, cmap="coolwarm", linewidth=0)


'''Determine equilibrium geometry and energy of molecule, and index (in the array) of those values'''
def eqm(E):
    E_min = np.amin(E)
    (r_ind, theta_ind) = np.where(E==E_min)
    
    '''converts index to actual value using known placement of values in array'''
    r_min = 0.05*(int(r_ind)) + 0.70
    theta_min = int(theta_ind) + 70
    #print([r_min, int(r_ind), theta_min, int(theta_ind), E_min])
    return [r_min, int(r_ind), theta_min, int(theta_ind), E_min]


def rezero(array_in, new_zero):
    return [x-new_zero for x in array_in]


def chunk(array, l, u, ind):
    return(array[ind-l:ind+u])


def fit_data(mol_dir):
    '''Energy and minimums in arrays'''
    E = val_arr(mol_dir)    
    [r_min, r_ind, theta_min, theta_ind, E_min] = eqm(E)
    
    '''define r and theta values, and re-zero & scale them for fitting'''
    old_r = np.linspace(0.7,1.9,25)
    old_theta = np.linspace(70,160,91)
    
    r = rezero(old_r, r_min)
    r = [x for x in r]
    
    theta = rezero(old_theta, theta_min)
    
    '''make arrays of E values for each degree of freedom across minimum energy value'''
    E_r_min = E[r_ind]
    E_theta_min = E[:,theta_ind]
    
    '''Define curve fit: to a + cx^2 curve, and return parameters (a,c)'''
    x0 = np.array([0,0])
    
    def fn(x, a, c):
        return a + c * x**2
    
    def fit_each(dof_E, dof):
        params, params_covariance = optimization.curve_fit(fn, dof, dof_E, x0)
        return(params)
        
    '''Define parameters for data chunk size; 
    tc: "theta chunk"; rc: "r chunk"
    manually-adjusted for best fit'''
    tc_u = 15
    tc_l = 10
    
    rc_u = 4
    rc_l = 1
    
    '''Get fit parameters for theta and r curves'''
    params_theta = fit_each(chunk(E_r_min, tc_l, tc_u, theta_ind), chunk(theta, tc_l, tc_u, theta_ind))
    params_r = fit_each(chunk(E_theta_min, rc_l, rc_u, r_ind), chunk(r, rc_l, rc_u, r_ind))
    
    '''define quadratic functions based on theta and r parameters'''
    fnplot_theta = [params_theta[0] + float(params_theta[1]) * x**2 for x in theta]
    fnplot_r = [params_r[0] + float(params_r[1]) * x**2 for x in r]
    
    '''plot E vs 'degree of freedom' scatter, and overlay fitted curve; for visual checking'''
    '''plt.figure(2)
    plt.scatter(theta, E_r_min, marker = ".")
    plt.plot(theta, fnplot_theta)
    
    plt.figure(3)
    plt.scatter(r, E_theta_min, marker = ".")
    plt.plot(r, fnplot_r)
    
    plt.show()'''
    return [params_theta, params_r, r_min]


def get_freqs(mol_dir):
    params = fit_data(mol_dir)
    r_eq = params[2] * 10**-10 #in m
    
    m_u = 1.66053886 * 10**-27 #in kg
    Eh = 4.359744650 * 10**-18 # in J
    c = 2.99792458*(10**10) #in cm per s
    
    '''Convert k_r to SI units'''
    k_r = 2 * params[1][1] #Hartrees per Angstrom^2
    k_r_SI = k_r * Eh * (10**-10)**-2 #J per m^2
    
    '''Convert k_theta to SI units'''
    k_theta = 2 * params[0][1] #Hartrees per degree^2
    k_theta_SI = k_theta * Eh * (2 * np.pi / 360)**-2
    
    '''Convert k_r to frequency'''
    v1 = 1/(2 * np.pi) * np.sqrt(k_r_SI / (2 * m_u)) #Hz
    
    '''Convert k_theta to frequency'''
    v2 = 1/(2 * np.pi) * np.sqrt(k_theta_SI / (r_eq**2 * 0.5 * m_u)) #Hz
    
    dp = 2
    
    return np.round(v1/c, dp), np.round(v2/c, dp)
    
if __name__ == "__main__":
    '''User input: directory of simulated files'''
    mol_dir = input("Input directory \n")
    
    '''Get letter 'O' or 'S' from directory name'''
    letter = mol_dir[len(mol_dir)-9:len(mol_dir)-8]
    
    [r_min, r_ind, theta_min, theta_ind, E_min] = eqm(val_arr(mol_dir))
    
    plot_surface(mol_dir)
    freqs = get_freqs(mol_dir)
    
    print("\nH2" + letter + " Equilibrium Geometry:")
    print(letter + "-H bond length = " + str(r_min) + " Angstrom")
    print("H-" + letter + "-H bond angle = " + str(theta_min) + " degrees")
    print("Potential energy at this geometry = " + str(E_min) + " Hartree\n")
    
    print("Symmetrical Normal Modes:")
    print("Stretch: frequency v1 = " + str(freqs[0]) + " wavenumbers")
    print("Bend: frequency v2 = " + str(freqs[1]) + " wavenumbers")
    
