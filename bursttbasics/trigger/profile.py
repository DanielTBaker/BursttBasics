import numpy as np
import astropy.constants as const
import astropy.units as u

def gaussian(t,A,w,t0,N):
    """
    Simple Gaussian Pulse Profile

    Parameters
    ===========
    t - numpy array
        Pulse time
    A - float
        Pulse Amplitude
    w - float
        Pulse Width
    t0 -  float
        Pulse peak time
    """
    return(A*np.exp(-np.power((t-t0)/w,2)/2)+N)

