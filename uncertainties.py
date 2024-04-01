#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:56:50 2024

@author: cobysilayan
"""

from numpy import sqrt, prod

def unc_mult(a, da, b, db):
    root = (da/a)**2 + (db/b)**2 
    return a*b * sqrt( root )

def unc_div(a, da, b, db):
    root = (da/a)**2 + (db/b)**2
    return a/b * sqrt(root)

def unc_op(arg_mul, arg_div):
    """
    

    Parameters
    ----------
    arg_mul : n x 2 array
        [a, da], [b, db] ... this are the quantities that are multiplied
    arg_div : n x 2 array
        [a, da], [b, db] ... this are the quantities that are divided

    Returns
    -------
    answer, uncertainty of the answer.

    """
    
    n_mul = len(arg_mul)
    n_div = len(arg_div)
    
    ans_mul = prod(arg_mul, axis = 0)[0]
    ans_div = prod(arg_div, axis = 0)[0]
    ans = ans_mul / ans_div
    
    unc_mul = 0
    unc_div = 0
    root = 0
    
    for i in range(0, n_mul):
        root += (arg_mul[i][1] / arg_mul[i][0])**2
    for j in range(0, n_div):
        #print(j[1])
        root += (arg_div[j][1] / arg_div[j][0])**2
        
    return ans, ans*sqrt(root) 

def unc_pow(a, da, power):
    perc_unc = da / a
    ans = a**power
    return ans, (perc_unc * power) * ans
    

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    sys.path.append('/Users/cobysilayan/Documents/GitHub/data_processing')
    import curve_fitting as cfit

    
    val, val_unc = unc_op([[1, 0.5], [1, 2]], [[2, 0.5], [1, 0]])
    
    def cos_law_theta(a, da, b, db, c, dc):
        a_sqr, da_sqr = unc_pow(a, da, 2)
        b_sqr, db_sqr = unc_pow(b, db, 2)
        c_sqr, dc_sqr = unc_pow(c, dc, 2)
        
        num = a_sqr + b_sqr - c_sqr
        #print(num)
        dnum = da_sqr + db_sqr + dc_sqr
        #print(num, dnum)
        
        ab = a*b 
        dab = unc_mult(a, da, b, db)
        
        arg = num / (2*ab)
        print(arg)
        darg = unc_div(num, dnum, 2*ab, 2*dab)
        
        theta = np.arccos(arg) 
        dtheta = abs(darg * (-1 / sqrt(1 - arg**2)))
        
        return theta, dtheta
        

    
    
    def compute_w(sep, ref, L, wavelength, inc_ang):
        
        theta_t = (1/ref[0]) * np.sin(np.radians(inc_ang[0]))
        sin_inc_ang = np.sin(np.radians(inc_ang[0]))
        unc_sin_inc_ang = np.cos(np.radians(inc_ang[0]))*np.radians(inc_ang[1])
        dtheta_t = unc_div(sin_inc_ang, unc_sin_inc_ang, ref[0], ref[1])
        
        #print(theta_t, dtheta_t)
        
        cos_inc_ang = np.cos(np.radians(inc_ang[0]))
        dcos_inc_ang = np.sin(np.radians(inc_ang[0]))*np.radians(inc_ang[1])
        
        tan_theta_t = np.tan(theta_t)
        dtan_theta_t = dtheta_t * (1/np.cos(theta_t))**2 
        #print([L, wavelength], [[2*sep[0], 2*sep[1]], [tan_theta_t, dtan_theta_t], [cos_inc_ang, dcos_inc_ang]])

        
        w, dw = unc_op([L, wavelength], [[2*sep[0], 2*sep[1]], [tan_theta_t, dtan_theta_t], [cos_inc_ang, dcos_inc_ang]])
        
        # sin_ang = np.sin(np.radians(inc_ang[0]))
        # unc_sin_ang = np.cos(np.radians(inc_ang[0]))*np.radians(inc_ang[1])
        
        # sin_ang_sq, unc_sin_ang_sq = unc_pow(sin_ang, unc_sin_ang, 2)
        # ref_sq, unc_ref_sq = unc_pow(ref[0], ref[1], 2)
        
        # root = 1 - sin_ang_sq / ref_sq
        # root_unc = unc_div(sin_ang, unc_sin_ang, ref_sq, unc_ref_sq)
        
        # sqroot, unc_sqroot = unc_pow(root, root_unc, 1/2)
        # #print([L, wavelength, [sqroot, unc_sqroot]], [sep])
        
        # w, dw = unc_op([L, wavelength, [sqroot, unc_sqroot]], [sep])
        
        return w, dw
    
    inc_ang = np.degrees(np.array(cos_law_theta(55.9, 0.6, 55.3, 0.6, 77.3, 0.6)))/2
        
    w_50mm = compute_w([510e-6, 20e-6], [1.5, 0], [0.559, 0.006], [633e-9, 0], inc_ang)
    w_75mm = compute_w([480e-6, 40e-6], [1.5, 0], [0.559, 0.006], [633e-9, 0], inc_ang)
    w_125mm = compute_w([470e-6, 20e-6], [1.5, 0], [0.559, 0.006], [633e-9, 0], inc_ang)
    w_150mm = compute_w([440e-6, 40e-6], [1.5, 0], [0.559, 0.006], [633e-9, 0], inc_ang)
    w_200mm = compute_w([400e-6, 40e-6], [1.5, 0], [0.559, 0.006], [633e-9, 0], inc_ang)
    
    ws = np.array([w_50mm, w_75mm, w_125mm, w_150mm, w_200mm])
    print(ws)
    
    def lin_func(x, a, b):
        return a*x + b
    
    #Plotting Width vs Focal Length
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel("Focal Length (mm)")
    ax.set_ylabel("Slidth Width (m)")
    focal_lengths = [50, 75, 125, 150, 200]
    #ax.scatter(focal_lengths, ws[:,0])
    #ax.errorbar(focal_lengths, ws[:,0], xerr=None, yerr=ws[:,1], linestyle="")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    w_fit = cfit.fit_n_plot(ax, focal_lengths, ws[:,0], ws[:,1], [1,1], lin_func)

    
    
    #Plotting Separation and extrapolating separation at zero
    
    separations = np.array([[510e-6, 20e-6], [480e-6, 40e-6], [470e-6, 20e-6], [440e-6, 40e-6], [400e-6, 40e-6]])
    
    #plt.figure(figsize=(7,5))
    
    fig, ax = plt.subplots(1,1)
    
    
    ax.set_ylabel("Maxima Separation (m)")
    ax.set_xlabel("Focal Length (mm)")
    ax.scatter(focal_lengths, separations[:,0])
    ax.errorbar(focal_lengths, separations[:,0], yerr = separations[:,1], linestyle="")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    
    
    
    


    