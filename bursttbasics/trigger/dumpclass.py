import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.stats import sigma_clip
from astropy.time import Time
from scipy.optimize import curve_fit
from astropy.coordinates import SkyCoord
import astropy.constants as const
from ..burstt_func import *
from .plotting import ext_find
from pycalc11 import Calc

from ..packet_func import *
from ..burstt_const import *
from ..burstt_func import *
from scipy.interpolate import interp1d


class StationDump:
    """
        Class for holding/processing a single beam of triggered data for a station
    """
    def __init__(self,station,location=None):
        self.station=station
        if location is None:
            self.location= eval(f'{station}Center')
        else:
            self.location = location

    def get_file(self,baseDir,eventID,beam,oldVersion=False,sharedDir = False):
        """
            Locate file needed for specified event and beam
        """
        if sharedDir:
            ## Multiple stations in a single dir
            files = os.listdir(baseDir)
            files = [f for f in files if eventID in f and str(beam)+'.npz' in f and self.station in f]
            self.file = os.path.join(baseDir,files[0])
        else:
            ## Each station in its own subdir
            files = os.listdir(os.path.join(baseDir,self.station))
            if oldVersion:
                files=[f for f in files if eventID in f and '.npz' in f]
            else:
                files = [f for f in files if eventID in f and str(beam)+'.npz' in f]
            self.file = os.path.join(baseDir,self.station,files[0])

    
    def load_file(self):
        """
            Load the data file
        """
        arch = np.load(self.file)
        self.spec = arch['dd_spec']
        self.freq = arch['freq']*u.MHz
        if len(arch['ep0'].shape)==0 and 'ep' in arch.files:
            self.ep0 = arch['ep']
        else:
            self.ep0 = arch['ep0']
        self.DM = arch['DM']*u.pc/u.cm**3
        if len(self.ep0.shape)==0:
            self.ep0=float(self.ep0)
        self.time = arch['dd_time']*u.us
        self.modeled = False
        self.normed = False
        self.clipped = False

    def load_event(self,baseDir,eventID,beam,oldVersion=False,sharedDir=False):
        """
        Single function to find and load file
        """
        self.get_file(baseDir,eventID,beam,oldVersion=oldVersion,sharedDir=sharedDir)
        self.load_file()

    def norm_spec(self):
        """
        Normalize the spectrum to have mean 1 intensity
        """
        I=np.abs(self.spec)**2
        norm = np.nanmean(I,1)
        self.spec/=np.sqrt(norm[:,np.newaxis])
        self.normed = True

    def clip_spec(self,clip=2):
        """
            Mask out channels with abnormal variance
        """
        I=np.abs(self.spec)**2
        v = np.var(I,1)
        v2 = sigma_clip(v,clip)
        self.spec[v2.mask]=np.nan
        self.clipped = True

    def fit_spec(self,polFit=True,rbin=400,freqMin = -np.inf,freqMax = np.inf,profModel=gau1,**kwargs):
        """
        Fit a simple pulse profile to the spectrum
        """
        self.profModel = profModel
        I = np.abs(self.spec)**2
        I2 = np.reshape(I[:,:rbin*(I.shape[1]//rbin)],(I.shape[0],-1,rbin)).mean(2)
        t2 = np.reshape(self.time[:rbin*I2.shape[1]],(-1,rbin)).mean(1).to(u.ms)
    
        prof,on = self.calc_SNR(t2,I2,meanSub=False,freqMin=freqMin,freqMax=freqMax)
        if 'p0t' in kwargs:
            self.poptT = kwargs['p0t']
        else:
            t0 = t2[np.argmax(prof)].value
            N = np.median(prof)
            A = prof.max()-N
            w = 10*(t2[1]-t2[0]).value
            self.poptT = np.array([A,t0,w,N])
        print(prof.max())
        self.poptT,pcov=curve_fit(self.profModel,t2.value,prof,p0=self.poptT)
        wts = self.profModel(self.time.to_value(u.ms),*self.poptT)-self.poptT[-1]
        wts /= np.nansum(wts)
        spec = self.calc_spec(I,wts,noiseSub=True)
        # spec-=1
        fmask = np.isfinite(spec)
        fmask*=(self.freq.value<=freqMax)*(self.freq.value>=freqMin)
        if 'mask' in kwargs:
            fmask*=kwargs['mask']
        if polFit:
            self.freq_function = frequency_model_polarized
            if 'p0f' in kwargs:
                self.poptF = kwargs['p0f']
            else:
                A0 = (np.max(spec[fmask]))*(self.freq[fmask][np.argmax(spec[fmask])]/(400*u.MHz))**2
                
                self.poptF = [A0,-2,50,0,.5,0]
        
                self.poptF,pcov = curve_fit(self.freq_function,self.freq[fmask].value,
                                       spec[fmask],p0=self.poptF,maxfev=10000,
                                       bounds = ((0,-np.inf,-np.inf,-np.inf,0,-np.inf),(np.inf,0,np.inf,np.inf,1,np.inf)),method='dogbox')
        else:
            self.freq_function = frequency_model
            if 'p0f' in kwargs:
                self.poptF = kwargs['p0f']
            else:
                A0 = (np.max(spec[fmask]))*(self.freq[fmask][np.argmax(spec[fmask])]/(400*u.MHz))**2
                self.poptF = [A0,-2,0]
                self.poptF,pcov = curve_fit(self.freq_function,self.freq[fmask].value,
                                       spec[fmask],p0=self.poptF,maxfev=10000,
                                       bounds = ((0,-np.inf,-np.inf),(np.inf,np.inf,np.inf)),method='dogbox')
        self.modeled = True
    def apply_freq_mask(self,mask,nanVal=True):
        """
        Apply a mask to the data
        """
        self.spec[mask==nanVal]=np.nan

    def calc_SNR(self,t,I,meanSub=True,freqMin = -np.inf,freqMax = np.inf,onWidth=5):
        """
        Calculate pulse SNR
        """
        SNR = np.nansum(I[(self.freq.value>=freqMin)*(self.freq.value<=freqMax)],0)
        on = np.abs(t-t[np.argmax(SNR)])<=(onWidth+.1)*(t[1]-t[0])
        if meanSub:
            SNR -= np.mean(SNR[np.invert(on)])
        SNR/=np.nanstd(SNR[np.invert(on)])
        return(SNR,on)

    def calc_spec(self,I,wts,noiseSub=False):
        """
        Get SNR vs Frequency
        """
        SF = np.nansum(I*wts,1)
        NF = np.nansum(I*np.roll(wts,wts.shape[0]//2),1)
        # NF = np.nanmedian(I,1)
        SNRF = SF/NF
        if noiseSub:
            SNRF-=1
        return(SNRF)

    def plot_spec(self,showModel = False,rbt = 400,rbf = 32,freqMin = -np.inf, freqMax = np.inf,onWidth=5):
        """
        Plot intensity as well as time and frequency SNR
        """
        I = np.abs(self.spec)**2
        I2 = np.nanmean(np.reshape(I[:,:rbt*(I.shape[1]//rbt)],(self.spec.shape[0],-1,rbt)),2)
        t2 = np.reshape(self.time[:rbt*I2.shape[1]],(-1,rbt)).mean(1)
        f2 = np.reshape(self.freq,(-1,rbf)).mean(1)
        SNR,on = self.calc_SNR(t2,I2,freqMin=freqMin,freqMax=freqMax)
        if self.modeled:
            wts = self.profModel(self.time.to_value(u.ms),*self.poptT)-self.poptT[-1]
            wts/=np.nansum(wts)
            SNRF = self.calc_spec(I,wts,noiseSub=True)
        else:
            wts = np.copy(SNR)
            wts[np.invert(on)]=0
            wts[wts<0]=0
            wts/=np.nansum(wts)
            SNRF = self.calc_spec(I2,wts,noiseSub=True)
        
        I2 = np.nanmean(np.reshape(I2,(self.spec.shape[0]//rbf,rbf,-1)),1)
        ext = ext_find(t2.to(u.ms),f2)
        grid=plt.GridSpec(nrows=2,ncols=2,height_ratios=[1,4],width_ratios=[4,1],wspace=0,hspace=0)
        plt.figure(figsize=(8,8))
        plt.subplot(grid[1,0])
        plt.imshow((I2-1)/np.nanstd(I2),origin='lower',aspect='auto',vmin=-2,vmax=4,extent=ext,cmap='cividis',interpolation='nearest')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (MHz)')
        plt.subplot(grid[0,0])
        plt.plot(t2.to(u.ms),SNR,'k')
        plt.axhline(0,ls='--',c='k',alpha=.5)
        plt.text(t2.to(u.ms).min().value,SNR.max()*.9,f'SNR = {SNR.max():.2f}')
        if showModel and self.modeled:
            model = self.profModel(t2.to_value(u.ms),*self.poptT)-self.poptT[-1]
            # model*=SNR.max()/model.max()
            plt.plot(t2.to(u.ms),model)
        plt.axvspan(t2[on].min().value,t2[on].max().value, alpha=0.5, color='green')
        plt.axhspan(-1,1,alpha=.5,color='grey')
        plt.xlim(ext[:2])
        plt.xticks([])
        plt.ylabel('SNR')
        plt.subplot(grid[1,1])
        plt.plot(SNRF,self.freq,'k')
        if showModel and self.modeled:
            model = self.freq_function(self.freq.value,*self.poptF)#-self.poptF[-1]
            plt.plot(model,self.freq)
        plt.ylim(ext[2:])
        plt.yticks([])
        plt.axvline(0,ls='--',c='k',alpha=.5)

def get_mean_overlap(station0,station1):
    """
    Determine mean of frequency overlap between two stations
    """
    return(get_f_overlap(station0,station1).mean())

def get_f_overlap(station0,station1):
    """
    Get frequency overlap between two station
    """
    use0,_ = get_overlap_mask(station0,station1)
    return(station0.freq[use0])

def get_overlap_mask(station0,station1):
    """
    Frequency overlap masks for two stations
    """
    f0 = station0.freq
    f1 = station1.freq
    use0 = (f0>=f1.min())*(f0<=f1.max())
    use1 = (f1>=f0.min())*(f1<=f0.max())
    return(use0,use1)

def get_start_off(station0,station1):
    """
    Determine offset in start time between two station
    """
    t0 = station0.ep0
    t1 = station1.ep0
    if isinstance(t0,np.ndarray) and isinstance(t1,np.ndarray):
        use0,use1 = get_overlap_mask(station0,station1)
        t1=Time(t1[use1][0],format='unix')
        t0=Time(t0[use0][0],format='unix')
        return((t0-t1).to(u.us))
    fref = station0.freq[0]
    if isinstance(t0,np.ndarray):
        t0=t0[0]
    t0=Time(t0,format='unix')
    if isinstance(t1,np.ndarray):
        t1=t1[0]
    t1=Time(t1,format='unix')
    t1+=(station1.DM/K)*(1/fref**2 - 1/station1.freq[0]**2)
    return((t0-t1).to(u.us))

def dump_delay(stations,source,mode='center',decOff=0,raOff=0,**kwargs):
    """
    Determine rate and delay of source signal between multiple stations
    """
    srce = SkyCoord(ra=[source.ra+raOff],dec=[source.dec+decOff])
    if 'time' in kwargs:
        epoch=kwargs['time']
    else:
        if isinstance(stations[0].ep0,np.ndarray):
            epoch = Time(Time(stations[0].ep0[0],format='unix').isot)
        else:
            epoch = Time(Time(stations[0].ep0,format='unix').isot)
    coords = []
    for station in stations:
        if hasattr(station,'location'):
            coords.append(station.location)
        else:
            coords.append(eval(f'{station.station}Center'))
    ci = Calc(
            station_names = [station.station for station in stations],
            station_coords=coords,
            source_coords=srce.icrs,
            start_time=epoch-2*u.min,
            duration_min=4,
            d_interval=1,base_mode='baseline'
        )
    ci.run_driver()
    if mode=='center':
        delay = ci.interpolate_delays(epoch).to(u.us)[0,1,0]
    else:
        delay = ci.interpolate_delays(epoch).to(u.us)[0,1,0]
    rates = ci.delay_rate[:,0,1,0]
    interp = interp1d(ci.times.mjd,rates.value)
    rate = interp(epoch.mjd)*rates.unit
    return(delay,rate)