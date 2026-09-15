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
        arch = np.load(self.file, allow_pickle=True)
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
        if 'normed' in arch.files:
            self.normed = arch['normed'].item()
        else:
            self.normed = False
        if 'clipped' in arch.files:
            self.clipped = arch['clipped'].item()
        else:
            self.clipped = False
        if 'ep0Time' in arch.files:
            self.ep0Time = arch['ep0Time'].item()
            if self.ep0Time:
                self.ep0=Time(self.ep0)
        else:
            self.ep0Time = False
        if 'poptT' in arch.files:
            self.poptT = arch['poptT']
            self.poptF = arch['poptF']
            self.profModel = globals()[arch['profModel'].item()]
            self.freq_function = globals()[arch['freq_function'].item()]
            self.modeled=True
        else:
            self.modeled = False

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
        # print(prof.max())
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

    def up_channelize(self,upChan = 8):
        self.spec =np.moveaxis(np.fft.fftshift(np.fft.fft(np.reshape(self.spec[:,:upChan*(self.spec.shape[1]//upChan)],(self.spec.shape[0],-1,upChan)),axis=-1),axes=-1),2,1).reshape((self.spec.shape[0]*upChan,-1))
        self.time = np.reshape(self.time[:,upChan*self.spec.shape[1]],(-1,upChan)).mean(1)
        self.freq = np.linspace(self.freq[0].value,self.freq[-1]+(self.freq[1]-self.freq[0]),upChan*self.freq.shape[0],endpoint=False)
        self.ep0 = np.repeat(self.ep0,upChan)

    def save(self,eventID,beam,dir='./'):
        if self.modeled:
            np.savez(os.path.join(dir,f'baseband_beamform_dedisperse_{self.station}_{eventID}_{beam}.npz'),
                    dd_spec = self.spec,
                    freq = self.freq.to_value(u.MHz),
                    ep0 = self.ep0,
                    dd_time = self.time.to_value(u.us),
                    DM = self.DM.to_value(u.pc/u.cm**3),
                    normed = self.normed,
                    clipped = self.clipped,
                    ep0Time = self.ep0Time,
                    profModel = self.profModel.__name__,
                    freq_function = self.freq_function.__name__,
                    poptT = self.poptT,
                    poptF = self.poptF
                    )
        else:
            np.savez(os.path.join(dir,f'baseband_beamform_dedisperse_{self.station}_{eventID}_{beam}.npz'),
                                dd_spec = self.spec,
                                freq = self.freq.to_value(u.MHz),
                                ep0 = self.ep0,
                                dd_time = self.time.to_value(u.us),
                                DM = self.DM.to_value(u.pc/u.cm**3),
                                normed = self.normed,
                                clipped = self.clipped,
                                ep0Time = self.ep0Time,
                                )

    def align_start(self,newEp0,fractionalShift=False):
        """
        Parameters
        ----------
        newEp0 : astropy.Time
            The desired start time for each frequency
        fractionalShift : bool (optional)
            Whether or not to do sub pixel shifts in each channel. Defaults to False
        Returns
        -------
        None
        """
        dtype = self.spec.dtype
        # Determine time spacing of samples
        sampleSpacing = self.time[1]-self.time[0]
        ## Determine desired shift in samples
        shift = ((newEp0-self.ep0)/sampleSpacing).to_value(u.dimensionless_unscaled)
        ## Divide shift into pixel and sub pixel settings
        coarseShift = np.round(shift).astype(int)
        fineShift = shift-coarseShift
        ## Apply pixel shifts
        for i in range(self.freq.shape[0]):
                        self.spec[i]=np.roll(self.spec[i],-coarseShift[i])
        self.ep0+=coarseShift*sampleSpacing
        ## Apply sub pixel shifts if desired
        if fractionalShift:
            f = np.fft.fftfreq(self.time.shape[0])
            Fspec =np.fft.fft(self.spec,axis=1)
            Fspec *= np.exp(1j*2*np.pi*(fineShift[:,np.newaxis]*f))
            self.ep0+=fineShift*sampleSpacing
            ## Crop spectra and time
            self.spec = np.fft.ifft(Fspec,axis=1)[:,:-1].astype(dtype)
            self.time = self.time[:-1]
        ## Crop spectra and time
        if coarseShift.max()>0:
            self.time = self.time[:-(coarseShift.max())]
            self.spec = self.spec[:,:-(coarseShift.max())]

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
    if isinstance(t0,Time) and isinstance(t1,Time):
        use0,use1 = get_overlap_mask(station0,station1)
        return((t0[use0][0]-t1[use1][0]).to(u.us))
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

