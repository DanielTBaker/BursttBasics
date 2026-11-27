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
import json

try:
    import bursttbasics.trigger.faradayfit as ff
    HAS_FITBURST = True
except:
    HAS_FITBURST = False



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

    def _fill_empty_ep0(self):
        empty = self.ep0 < Time('1991-02-27T00:00:00')
        if not empty.any() or empty.all():
            self.emptyChan = empty
            return
        delay = (self.DM/K)*(1/self.freq**2)
        dt = self.time[1]-self.time[0]
        good = np.where(~empty)[0]
        new = self.ep0.copy()
        for c in np.where(empty)[0]:
            k = good[np.argmin(np.abs(good - c))]                 # nearest valid channel
            n = np.round(((delay[c] - delay[k])/dt).to_value(u.dimensionless_unscaled))
            new[c] = self.ep0[k] + n*dt
        self.ep0 = new
        self.emptyChan = empty



    
    def load_file(self):
        """
            Load the data file
        """
        arch = np.load(self.file, allow_pickle=True)
        self.spec = arch['dd_spec']
        self.freq = arch['freq']*u.MHz
        self.DM = arch['DM']*u.pc/u.cm**3
        
        self.time = arch['dd_time']*u.us
        if 'normed' in arch.files:
            self.normed = arch['normed'].item()
        else:
            self.normed = False
        if 'clipped' in arch.files:
            self.clipped = arch['clipped'].item()
        else:
            self.clipped = False
        if 'ep0JD1' in arch:
                        self.ep0 = Time(arch['ep0JD1'], arch['ep0JD2'], format='jd', scale=str(arch['ep0Scale']))
                        self.ep0Time = True
        elif 'ep0Time' in arch.files:
            self.ep0Time = arch['ep0Time'].item()
            if self.ep0Time:
                self.ep0=Time(self.ep0)
        else:
            self.ep0 = arch['ep0']
            if isinstance(self.ep0[0],Time):
                self.ep0=Time(self.ep0)
                self.ep0Time = True
            else:
                self.ep0Time = False
                self.ep0 = Time(self.ep0,format='unix')
        if 'poptT' in arch.files:
            self.poptT = arch['poptT']
            self.poptF = arch['poptF']
            self.profModel = globals()[arch['profModel'].item()]
            self.freq_function = globals()[arch['freq_function'].item()]
            self.modeled=True
            self.modelFormat = 'separable'
        else:
            self.modeled = False
            self.modelFormat = ''

        self.coherentDM = False
        self.appliedDM = 0*u.pc/u.cm**3
        self.removedDelay = 0 * u.ns
        self._fill_empty_ep0()

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

    def rm_grid(self,Ion,mask,rmMin=1,rmMax=120,rmNum = 600,aLow = -5, aHigh = 0, aNum = 25):
        bestChi = np.inf
        x = self.freq[mask].value/400
        lam2 = (const.c/self.freq[mask]).to_value(u.m)**2
        A = 0
        pf = 0
        phi0 = 0
        N = 0
        rm = 0
        a = 0
        for RMSearch in np.linspace(rmMin,rmMax,rmNum):
            theta = 2* RMSearch * lam2
            for aSearch in np.linspace(aLow,aHigh,aNum):
                xa = x**aSearch
                X = np.array([xa,xa*np.cos(theta),xa*np.sin(theta),np.ones(x.shape[0])]).T
                co, resid, rank, sv = np.linalg.lstsq(X, Ion[mask], rcond=None)
                resid = np.sum((Ion[mask] - X @ co)**2)
                if resid < bestChi:
                    bestChi=resid
                    A = co[0]
                    pf = np.sqrt(co[1]**2 + co[2]**2)/A
                    phi0 = np.arctan2(-co[2],co[1])/2.
                    N = co[-1]
                    rm = RMSearch
                    a = aSearch
        return(A,a,rm,phi0,pf,N)

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
        self.poptT[2]=np.abs(self.poptT[2])
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
                # A0 = (np.max(spec[fmask]))*(self.freq[fmask][np.argmax(spec[fmask])]/(400*u.MHz))**2
                
                # self.poptF = [A0,-2,50,0,.5,0]
                rmKwargs={}
                if 'rmMin' in kwargs:
                    rmKwargs.update({'rmMin' : kwargs['rmMin']})
                if 'rmMax' in kwargs:
                    rmKwargs.update({'rmMax' : kwargs['rmMax']})
                self.poptF = list(self.rm_grid(spec,fmask,**rmKwargs))
            if self.poptF[2]<2:
                self.freq_function = frequency_model
                self.poptF = [self.poptF[0],self.poptF[1],self.poptF[-1]]
                self.poptF,pcov = curve_fit(self.freq_function,self.freq[fmask].value,
                                                       spec[fmask],p0=self.poptF,maxfev=10000,
                                                       bounds = ((0,-np.inf,-np.inf),(np.inf,np.inf,np.inf)),method='dogbox')
            else:
                self.poptF[0] = max(self.poptF[0], 1e-6)                      # A >= 0
                self.poptF[1] = min(self.poptF[1], 0.)                        # spectral index <= 0
                self.poptF[4] = float(np.clip(abs(self.poptF[4]), 0.01, 0.99))
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
        self.modelFormat='separable'

    def fitburst_fit(self,rbt = 16,numIterations = 1,polFit = False, verbose=False,plotOut=None,fitOut=None,**kwargs):
        if not HAS_FITBURST:
            raise ImportError('fitburst_fit requires the fitburst package')
        initialParameters = {"amplitude"        : [-1.0],
                        "arrival_time"         : [0.0],
                        "burst_width"          : [0.0005],
                        "dm"                   : [0.0],
                        "dm_index"             : [-2.0],
                        "ref_freq"             : [400],
                        "scattering_index"     : [-4.0], 
                        "scattering_timescale" : [.00001],
                        "spectral_index"       : [-2.0],
                        "spectral_running"     : [0.0]}
        if 'arrival_time' in kwargs:
            initialParameters['arrival_time'] = [kwargs['arrival_time']]
        elif hasattr(self,'poptT'):
            initialParameters['arrival_time'] = [self.poptT[1]/1000]
        if 'burst_width' in kwargs:
            initialParameters['burst_width'] = [kwargs['burst_width']]
        elif hasattr(self,'poptT'):
            initialParameters['burst_width'] = [self.poptT[2]/2000]
        if 'scattering_timescale' in kwargs:
            initialParameters['scattering_timescale'] = [kwargs['scattering_timescale']]
        elif hasattr(self,'poptT'):
            initialParameters['scattering_timescale'] = [self.poptT[2]/2000]
        if polFit or self.freq_function == frequency_model_polarized:
            if 'rm' in kwargs:
                initialParameters.update({'rm' : [kwargs['rm']]})
            elif self.freq_function == frequency_model_polarized:
                initialParameters.update({'rm' : [self.poptF[2]]})
            else:
                initialParameters.update({'rm' : [-42.0]})
            if 'polarization_fraction' in kwargs:
                initialParameters.update({'polarization_fraction' : [kwargs['polarization_fraction']]})
            elif self.freq_function == frequency_model_polarized:
                initialParameters.update({'polarization_fraction' : [self.poptF[4]]})
            else:
                initialParameters.update({'polarization_fraction' : [.5]})
            if 'phi0' in kwargs:
                initialParameters.update({'phi0' : [kwargs['phi0']]})
            elif self.freq_function == frequency_model_polarized:
                initialParameters.update({'phi0' : [self.poptF[3]]})
            else:
                initialParameters.update({'phi0' : [0.0]})
        self.burstfitParameters = initialParameters.copy()
        data = ff.BURSTTDataReader(self,initialParameters,rbt=rbt)
        data.preprocess_data(remove_baseline=True)
        parametersToFix = ["dm_index", "scattering_index"]
        numComponents = len(initialParameters['burst_width'])

        data.dedisperse(
            0,
            initialParameters["arrival_time"][0],
            ref_freq=initialParameters["ref_freq"][0],
            dm_offset=0.0
        )
        dataWindowed = data.data_full
        timesWindowed = data.times
        if 'rm' in initialParameters:
            model = ff.FaradaySpectrumModeler(
                data.freqs,
                timesWindowed,
                dm_incoherent = 0,
                factor_freq_upsample = 1,
                factor_time_upsample = 4,
                num_components = numComponents,
                is_dedispersed = data.is_dedispersed,
                is_folded = False,
                scintillation = False,
                verbose = verbose
            )
        else:
            model = ff.SpectrumModeler(
                data.freqs,
                timesWindowed,
                dm_incoherent = 0,
                factor_freq_upsample = 1,
                factor_time_upsample = 4,
                num_components = numComponents,
                is_dedispersed = data.is_dedispersed,
                is_folded = False,
                scintillation = True,
                verbose = verbose
            )
        model.update_parameters(initialParameters)

        for currentIteration in range(numIterations):
            fitter = ff.fb.analysis.fitter.LSFitter(dataWindowed, model, data.good_freq)
            fitter.global_parameters.append("rm")
            fitter.fix_parameter(parametersToFix)
            fitter.fit(exact_jacobian=True)

            # extract best-fit data for next loop.
            if fitter.results.success:
                self.burstfitParameters.update(fitter.fit_statistics["bestfit_parameters"])
                self.modeled = True
                self.modelFormat = 'fitburst'
                model.update_parameters(fitter.fit_statistics["bestfit_parameters"])
                current_params = model.get_parameters_dict()

                if not any([x == "dm" for x in parametersToFix]):
                    current_params["dm"] = [x for x in self.burstfitParameters["dm"] * numComponents]

                if "scattering_timescale" not in parametersToFix:
                    current_params["scattering_timescale"] = [x for x in 
                                                            self.burstfitParameters["scattering_timescale"] * numComponents] 

                # if this is the last iteration, create best-fit model and plot windowed data.
                if currentIteration == (numIterations - 1):
                    if not fitOut is None:
                        if isinstance(fitOut,str):
                            with open(fitOut, "w") as out:
                                json.dump(
                                    {
                                        "initial_dm": 0,
                                        "initial_time": data.times_bin0,
                                        "model_parameters": self.burstfitParameters,
                                        "fit_statistics": fitter.fit_statistics,
                                    },
                                    out,
                                    indent=4
                                )
                    if not plotOut is None:
                        model.update_parameters(self.burstfitParameters)
                        bestfitUncertainties = fitter.fit_statistics["bestfit_uncertainties"]
                        bestfitModel = model.compute_model(data=dataWindowed)
                        if verbose:
                            print(f"INFO: best-fit estimate for {len(self.burstfitParameters['dm'])}-component model:")

                            for current_parameter_label in self.burstfitParameters.keys():
                                current_list = self.burstfitParameters[current_parameter_label]
                                current_uncertainties = bestfitUncertainties[current_parameter_label]
                                print(f"    * {current_parameter_label}: {current_list} +/- {current_uncertainties}")        

                            print("INFO: ratio of hessian matrix (approximate / exact):")
                            print(fitter.hessian / fitter.hessian_approx)

                        factor_time = max(1, int(round(150e-6/data.res_time)))      # ~150 µs plot bins
                        half = int(round(0.015/data.res_time))                      # ±15 ms around the pulse
                        i0 = int(round(self.burstfitParameters['arrival_time'][0]/data.res_time))
                        lo = max(0, i0 - half); n = ((min(len(timesWindowed), i0 + half) - lo)//factor_time)*factor_time
                        sl = slice(lo, lo + n)
                        
                        data_grouped = ff.ut.plotting.compute_downsampled_data(
                            timesWindowed[sl], data.freqs, dataWindowed[:, sl], data.good_freq,
                            spectrum_model=bestfitModel[:, sl], factor_freq=32, factor_time=factor_time)
                        ff.ut.plotting.plot_summary_triptych(data_grouped, output_name=plotOut, show=False)

    def build_model(self):
        if self.modeled:
            if self.modelFormat == 'separable':
                modelT = self.profModel(self.time.to_value(u.ms),*self.poptT)-self.poptT[-1]
                modelF = self.freq_function(self.freq.value,*self.poptF)-self.poptF[-1]
                return(modelF[:,np.newaxis]*modelT)
            else:
                if 'rm' in self.burstfitParameters:
                    model = ff.FaradaySpectrumModeler(
                        self.freq.value,
                        self.time.to_value(u.s),
                        dm_incoherent = 0,
                        factor_freq_upsample = 1,
                        factor_time_upsample = 1,
                        num_components = 1,
                        is_dedispersed = True,
                        is_folded = False,
                        scintillation = False,
                        verbose = False
                    )
                else:
                    model = ff.SpectrumModeler(
                        self.freq.value,
                        self.time.to_value(u.s),
                        dm_incoherent = 0,
                        factor_freq_upsample = 1,
                        factor_time_upsample = 1,
                        num_components = 1,
                        is_dedispersed = True,
                        is_folded = False,
                        scintillation = True,
                        verbose = False
                    )
                data = ff.BURSTTDataReader(self,self.burstfitParameters,rbt=1)
                data.preprocess_data(remove_baseline=True)
                model.update_parameters(self.burstfitParameters)
                return(model.compute_model(data=data.data_full))
        return(np.ones(self.spec.shape))

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
        SNR/=np.nanstd(SNR[np.invert(on)])
        if meanSub:
            SNR -= np.median(SNR[np.invert(on)])
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
        SNR,on = self.calc_SNR(t2,I2,freqMin=freqMin,freqMax=freqMax,meanSub=True)
        if self.modeled:
            if self.modelFormat == 'fitburst':
                M = self.build_model()
                M[self.spec==np.nan]=np.nan
                wts = np.nansum(M,0)
            else:
                wts = self.profModel(self.time.to_value(u.ms),*self.poptT)-self.poptT[-1]
            wts/=np.nansum(wts)
            SNRF = self.calc_spec(I,wts,noiseSub=True)
        else:
            wts = np.copy(SNR-1)
            wts[np.invert(on)]=0
            wts[wts<0]=0
            wts/=np.nansum(wts)
            SNRF = self.calc_spec(I2,wts,noiseSub=True)

        I2b = np.nanmean(np.reshape(I2,(self.spec.shape[0]//rbf,rbf,-1)),1)
        ext = ext_find(t2.to(u.ms),f2)
        grid=plt.GridSpec(nrows=2,ncols=2,height_ratios=[1,4],width_ratios=[4,1],wspace=0,hspace=0)
        plt.figure(figsize=(8,8))
        plt.subplot(grid[1,0])
        plt.imshow((I2b-1)/np.nanstd(I2b),origin='lower',aspect='auto',vmin=-2,vmax=4,extent=ext,cmap='cividis',interpolation='nearest')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (MHz)')
        plt.subplot(grid[0,0])
        plt.plot(t2.to(u.ms),SNR,'k')
        # plt.axhline(0,ls='--',c='k',alpha=.5)

        plt.text(t2.to(u.ms).min().value,SNR.max()*.9,f'SNR = {SNR.max():.2f}')
        try:
            SNR2,on = self.calc_SNR(t2,I2,freqMin=freqMin,freqMax=500*u.MHz,meanSub=True)
            plt.text(t2.to(u.ms).min().value,SNR.max()*.9,f'SNR(<500MHz) = {SNR2.max():.2f}')
        except:
            pass
        if showModel and self.modeled:
            if self.modelFormat == 'fitburst':
                M2 = np.nanmean(np.reshape(M[:,:rbt*(M.shape[1]//rbt)],(self.spec.shape[0],-1,rbt)),2)

                model = np.nansum(M2[(self.freq.value>=freqMin)*(self.freq.value<=freqMax)],0)
                SNR = np.nansum(I2[(self.freq.value>=freqMin)*(self.freq.value<=freqMax)],0)
                model /= np.nanstd(SNR[np.invert(on)])
            else:
                model = self.profModel(t2.to_value(u.ms),*self.poptT)-self.poptT[-1]
            # model*=SNR.max()/model.max()
            plt.plot(t2.to(u.ms),model)
        plt.axvspan(t2[on].min().value,t2[on].max().value, alpha=0.5, color='green')
        # plt.axhspan(-1,1,alpha=.5,color='grey')
        plt.xlim(ext[:2])
        plt.xticks([])
        plt.ylabel('SNR')
        plt.subplot(grid[1,1])
        plt.plot(SNRF,self.freq,'k')
        if showModel and self.modeled:
            if self.modelFormat == 'fitburst':
                NF = np.nansum(I * np.roll(wts, wts.shape[0] // 2), 1)
                model = np.nansum(M * wts, 1) / NF
                model[~np.isfinite(SNRF)] = np.nan
            else:
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

    def coherent_dedis(self,skipDM=False,**kwargs):
        upChan = self.spec.shape[1]
        dt = self.time[1]-self.time[0]
        spec2 = np.fft.fft(self.spec,axis=-1)
        fOff = np.fft.fftfreq(upChan,d=dt).to(u.MHz)
        phase = np.zeros(self.spec.shape)
        if not self.coherentDM and not skipDM:
            phase -= 2*np.pi*(self.DM/K).to_value(u.Hz)*(fOff.to_value(u.Hz)[np.newaxis,:]**2)/(self.freq.to_value(u.Hz)[:,np.newaxis]**2 * (self.freq.to_value(u.Hz)[:,np.newaxis]+fOff.to_value(u.Hz)[np.newaxis,:]))
            self.appliedDM = self.DM*1
            self.coherentDM = True
        if 'dDM' in kwargs:
            phase -= 2*np.pi*(kwargs['dDM']/K).to_value(u.Hz)*(fOff.to_value(u.Hz)[np.newaxis,:]**2)/(self.freq.to_value(u.Hz)[:,np.newaxis]**2 * (self.freq.to_value(u.Hz)[:,np.newaxis]+fOff.to_value(u.Hz)[np.newaxis,:]))
            self.appliedDM += kwargs['dDM']
        if "tauShiftAbs" in kwargs:
            phase += 2*np.pi*((self.freq[:,np.newaxis]-fOff[np.newaxis,:])*(kwargs['tauShiftAbs']-self.removedDelay)).to_value(u.dimensionless_unscaled)
            self.removedDelay = kwargs['tauShiftAbs']
        if "tauShift" in kwargs:
            phase += 2*np.pi*((self.freq[:,np.newaxis]-fOff[np.newaxis,:])*kwargs['tauShift']).to_value(u.dimensionless_unscaled)
            self.removedDelay = self.removedDelay + kwargs['tauShift']
        spec2*= np.exp(-1j*phase)
        self.spec = np.fft.ifft(spec2,axis=-1).astype(np.complex64)

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

