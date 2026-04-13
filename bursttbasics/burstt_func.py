from astropy.time import Time
import astropy.units as u
import astropy.constants as const
import numpy as np

from .packet_func import *
from .burstt_const import *
from tqdm.notebook import tqdm
import scintools.ththmod as thth

def gau1(x,A,x0,s,C):
    """ A simple 1D gaussian.

    Parameters
    ----------
    x : array-like
        x coordinate for gaussian.
    A : float
        Amplitude of gaussian.
    x0 : float
        x coordinate of peak.
    s : float
        Gaussian width.
    C: float
        Background level
    
    Returns
    -------
    gaus : array-like
        Gaussian of the form A e^(-(x-x0)^2/(2s^2))+C
    
    """
    gaus = A*np.exp(-np.power((x-x0)/s,2)/2)+C
    return(gaus)

def power(x,A,gamma):
    """ Power law for fitting

    Parameters
    ----------
    x : array-like
        x coordinate for gaussian.
    A : float
        Amplitude of power law.
    gamma : float
        Power low exponent.

    Returns
    -------
    pwr: array-like

    """
    pwr = A*np.power(x,gamma)
    return(pwr)

def frequency_model_polarized(f,A,n,RM,phi0,pf,N):
    """
    Pulse Spectral profile with RM fluctuations

    Parameters
    ===========
    f - numpy array
        Observing Frequency
    A - float
        Pulse Amplitude
    n - float
        Pulse spectral index
    RM -  float
        Pulse Rotation Measure
    phi0 - float
        RM Phase at 400MHz
    pf - float
        Polarization Fraction
    """
    I0 = A*np.power(f/400,n)
    lam = (const.c/(f*u.MHz)).to(u.m)
    I=I0*(1-pf + pf*np.cos(RM*lam.value**2+phi0)**2)
    return(I+N)

def frequency_model(f,A,n,N):
    """
    Pulse Spectral profile with RM fluctuations

    Parameters
    ===========
    f - numpy array
        Observing Frequency
    A - float
        Pulse Amplitude
    n - float
        Pulse spectral index
    RM -  float
        Pulse Rotation Measure
    phi0 - float
        RM Phase at 400MHz
    pf - float
        Polarization Fraction
    """
    I0 = A*np.power(f/400,n)
    return(I0+N)

def calculate_uvw(source, time, s1, s2):
    '''
    Calculates the projected uvw track for the baseline s2-s1.
    Negative u points West, Negative v points South.
    Parameters:
    ----------
    source: astropy.coordinates.SkyCoord object,
            the phase center of the observation
    time:   astropy.time.Time object, the time of the
            observation
    s1:     astropy.coordinates.EarthLocation object,
            station 1
    s2:     astropy.coordinates.EarthLocation object,
            station 2
    Returns:
    -------
    uvw, a list of the u, v and w tracks
    '''
    gw = EarthLocation(lon=0, lat=0)
    b = np.array([
        s2.x.value - s1.x.value, s2.y.value - s1.y.value,
        s2.z.value - s1.z.value
    ]) * u.m
    #     altaz = source.transform_to(AltAz(obstime=time,location=gw))
    H = time.sidereal_time('mean', longitude=gw.lon) - source.ra
    d = source.dec
    trans = np.array(
        [[np.sin(H).value, np.cos(H).value, np.zeros(H.shape)],
         [-np.sin(d).value * np.cos(H).value,
          np.sin(d).value * np.sin(H).value,
          np.cos(d).value*np.ones(H.shape)],
         [np.cos(d).value * np.cos(H).value, -np.cos(d).value * np.sin(H).value,
          np.sin(d).value*np.ones(H.shape)]])
    return np.sum(trans*b[np.newaxis,:,np.newaxis],1)    


def unwrap(y,hatLength=100):
    yOut = np.copy(y)
    hat = np.ones(hatLength)
    comp = np.exp(1j*yOut)
    correction = yOut -np.angle(comp)
    correction-=correction[0]
    comp2 = np.convolve(comp,hat,mode='same')
    smoothedAngs = np.angle(comp2)
    smoothedAngs+=correction
    smoothedDiff = np.diff(smoothedAngs)
    while np.abs(smoothedDiff).max()>np.pi:
        idx = np.argmax(np.abs(smoothedDiff))
        if smoothedDiff[idx]>0:
            smoothedAngs[idx+1:]-=2*np.pi
        else:
            smoothedAngs[idx+1:]+=2*np.pi
        smoothedDiff = np.diff(smoothedAngs)
    
    while (yOut-smoothedAngs).max()>np.pi:
        yOut[(yOut-smoothedAngs)>np.pi] -=2*np.pi
    while (yOut-smoothedAngs).min()<-np.pi:
        yOut[(yOut-smoothedAngs)<-np.pi] +=2*np.pi
    return(yOut)


def shiftArr(arr,x,xExpected,coher = False):
    if coher:
        Farr = np.fft.fftshift(np.fft.fft(arr,axis=1),axes=1)
        fx = thth.fft_axis(x,1/x.unit)
        x2 =thth.fft_axis(fx,x.unit)
        Farr*=np.exp(1j*2*np.pi*((xExpected[:,np.newaxis]-x[0])*fx[np.newaxis,:]).to(u.dimensionless_unscaled))
        shiftedArr = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Farr,axes=1),axis=1).real,axes=1)
    else:
        shifts = x.shape[0]//2-((xExpected-x[0])//(x[1]-x[0])).astype(int)
        shiftedArr = np.copy(arr)
        for i in range(shiftedArr.shape[0]):
            shiftedArr[i]=np.roll(shiftedArr[i],shifts[i])
        x2 = np.linspace(0,x.shape[0]-1,x.shape[0])*(x[1]-x[0])
        x2 -= x2[x2.shape[0]//2]
    return(shiftedArr,x2)