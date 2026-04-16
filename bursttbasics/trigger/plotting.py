import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from ..burstt_func import ext_find

def plot_corr(offsets,tau,corrs,**kwargs):
    
    amps = np.abs(corrs)**2
    grid=plt.GridSpec(nrows=2,ncols=2,width_ratios=[4,1],height_ratios=[4,1],wspace=0,hspace=0)
    ext = ext_find(offsets*2.56*u.us,tau)
    plt.figure()
    plt.subplot(grid[0,0])
    plt.imshow(amps,origin='lower',aspect='auto',extent=ext,interpolation='nearest')
    plt.xticks([])
    plt.ylabel('Fine Shift (ns)')
    plt.subplot(grid[1,0])
    plt.plot(offsets*2.56,amps.max(0))
    plt.xlim(ext[:2])
    plt.xlabel('Coarse Shift (us)')
    plt.ylabel('Max')
    if 'expected' in kwargs:
        if isinstance(kwargs['expected'],list) or isinstance(kwargs['expected'],np.ndarray):
            for delay in kwargs['expected']:
                if isinstance(delay,u.Quantity):
                    plt.axvline(delay.to_value(u.us),ls='--',c='k')
        elif isinstance(kwargs['expected'],u.Quantity):
            plt.axvline(kwargs['expected'].to_value(u.us),ls='--',c='k')
        
    plt.subplot(grid[0,1])
    plt.plot(amps.max(1),tau)
    plt.yticks([])
    plt.xlabel('Max')
    plt.ylim(ext[2:])