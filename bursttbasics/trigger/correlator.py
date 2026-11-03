from .dumpclass import *
import astropy.units as u
from ..burstt_func import *
import numpy as np
from scipy.optimize import curve_fit
from .plotting import *
import zipfile
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Optional
import json

def get_peak_time(s0,width=2048,weight=True,idPeak=None):
    sampleTime = s0.time[1]-s0.time[0]
    if idPeak==None:
        if weight:
            modelT = s0.profModel(s0.time.to_value(u.ms),*s0.poptT)-s0.poptT[-1]
            idPeak = np.argmax(modelT)
        else:
            idPeak = s0.spec.shape[1]//2
    else:
        idPeak=idPeak
    if s0.ep0Time:
        tRef = s0.ep0[0]+idPeak*sampleTime
    elif isinstance(s0.ep0,np.ndarray):
        tRef = Time(s0.ep0[0],format='unix')+idPeak*sampleTime
    else:
        tRef = Time(s0.ep0,format='unix')+idPeak*sampleTime
    return(tRef,idPeak)

def correlate_dump(s0,s1,
                   baseOff=0*u.us,maxOff=100,
                   width=2048,weight=True,idPeak=None,
                   rate=0*u.s*u.Hz,dDM = 0*u.pc/u.cm**3,
                   progress=False,leave=True,mean=False,tauShift = 0*u.ns,meanSub=False
                  ):
    """ 
    Correlate the data in two station dumps

    Parameters
    ----------
    s0 : bursttcorr.dumpclass.StationDump
        Dumped data from station 0
    s1 : bursttcorr.dumpclass.StationDump
        Dumped data from station 1
    maxOff : int
        Number of coarse shifts to search on either side of baseOff
    width : int
        Number of sample to correlate
    weight : bool
        Weight correlation using model from s0
    idPeak : bool
        sample ID to center at
    
    

    """
    sampleTime = s0.time[1]-s0.time[0]
    baseOffIDX = int(np.round(baseOff/sampleTime))
    ## Determine freq overlap
    f0 = s0.freq
    f1 = s1.freq
    use0 = (f0>=f1.min())*(f0<=f1.max())
    use1 = (f1>=f0.min())*(f1<=f0.max())
    ## If using a model weight determine the weights
    ## also determine bounds of region to correlate
    tRef,idPeak = get_peak_time(s0,width,weight,idPeak)
    id0=idPeak-width//2
    if weight:
        modelT = s0.profModel(s0.time.to_value(u.ms),*s0.poptT)-s0.poptT[-1]
        modelF = s0.freq_function(f0.value,*s0.poptF)-s0.poptF[-1]
        model = modelF[use0][:,np.newaxis]*modelT[id0:id0+width]
    

    ## Crop and conjugate spectrum from 1st station
    spec0 = np.conjugate(s0.spec[use0,id0:id0+width])
    ## Determine time from reference for each sample and correct rate
    if not rate==0:
        if s0.ep0Time:
            t =  (s0.time[id0:id0+width].to_value(u.s) -idPeak*sampleTime.to_value(u.s))[np.newaxis,:] + (s0.ep0[use0]-s0.ep0[0]).to_value(u.s)[:,np.newaxis]
        else:
            t =  (s0.time[id0:id0+width].to_value(u.s) -idPeak*sampleTime.to_value(u.s))[np.newaxis,:] + (s0.ep0[use0]-s0.ep0[0])[:,np.newaxis]
        
        f = s0.freq[use0].to_value(u.Hz)
        spec0 *= np.exp(-1j*2*np.pi*(f[:,np.newaxis]*(rate*t)))
        t=None
    ## Apply correction for ionospheric DM difference
    if not dDM ==0:
        dP = ((dDM/K)/s0.freq[use0]).to_value(u.dimensionless_unscaled)
        spec0 *= np.exp(-1j*2*np.pi*dP)[:,np.newaxis]
    if not tauShift == 0*u.ns:
        spec0*=np.exp(-1j*2*np.pi*(tauShift.to_value(u.s)*s0.freq[use0].to_value(u.Hz)))[:,np.newaxis]
    
    ## Check difference in start times
    startDiff = get_start_off(s0,s1)
    startOff = (np.round(startDiff/sampleTime)).astype(int)
   
    startResidual = startDiff - startOff*sampleTime
    offsets = np.linspace(-maxOff,maxOff,2*maxOff+1,dtype=int)+baseOffIDX
    corrs = np.zeros((spec0.shape[0],offsets.shape[0]),dtype=np.complex64)
    cross = np.zeros(spec0.shape,dtype=spec0.dtype)
    iterator = enumerate(offsets)
    if progress:
        iterator=tqdm(iterator,total=2*maxOff+1,leave=leave)
    try:
        for ido,o in iterator:
            ## Shift second station against first and multiple
            cross[:,:] = spec0*s1.spec[use1,id0+o+startOff:id0+o+startOff+width]
            ## Normalize and apply weights
            cross/=np.abs(cross)
            if weight:
                # cross/=np.abs(cross)
                cross*=model
            ## FFT allong frequency for fine delay
            if meanSub:
                cross-=np.nanmean(cross)
            if mean:
                fCross = np.fft.fftshift(np.fft.fft(np.nan_to_num(np.nanmean(cross,1))))
                corrs[:,ido]=fCross/np.median(np.abs(fCross))
            else:
                fCross = np.fft.fft2(np.nan_to_num(cross))
                fAmp = np.abs(fCross)
                ## Record correlation vs fine delay for given coarse shift
                corrs[:,ido] = np.fft.fftshift(fCross[:,0])/np.median(fAmp)
    except Exception as e:
        print('----')
        print(f'Total Size: {s0.spec.shape[1]},{s1.spec.shape[1]}')
        print(f'Start Offset: {startOff}')
        print(f'Cut Start: {id0}')
        print(f'Max Off: {maxOff}')
        print(f'Base Off: {baseOffIDX}')
        print(f'FirstPx: {id0+offsets.min()+startOff+id0}')
        print(f'LastPx: {id0+offsets.max()+startOff+id0+width}')
        print('----')
        raise e

    tau = fft_axis(f0[use0],u.ns)
    return(offsets,tau,corrs,startResidual,tRef)

def rate_dm_peaks(station0,station1,baseDelay,baseRate,
                width=2048,weight=True,idPeak=None,mean=False,maxOff=20,
                 searchRange=1e-6*u.s*u.Hz,nRate=61,
                 dDMRange=1e-5*u.pc/u.cm**3,dDMFitRange=0,nDMs=1001,
                 progress=False,line=False,subSearch = np.linspace(0,0,1),**kwargs):
    """
    Determine peak SNR over a grid of rates and differential DMs

    Parameters
    ----------
    station0 : StationDump
        The StationDump class with the data for the first station
    station1 : StationDump
        The StationDump class with the data for the second station 
    baseDelay : astropy.Quantity
        The expected signal delay of station1 relative to station0
    baseRate : astropy.Quantity
        The expected fringe rate between station1 and station0
    width : int
        Number of sample around peak to use in correlation. Default = 2048
    weight : bool
        Use fitted pulse profile weights for correlation. Default = True
    Returns
    -------
    """
    if 'meanSub' in kwargs:
        meanSub=kwargs['meanSub']
    else:
        meanSub=False
    dmSlope = -station0.DM
    fOverlap = get_f_overlap(station0,station1)
    sampleTime = station0.time[1]-station0.time[0]
    baseOff = int((baseDelay//sampleTime).to_value(u.dimensionless_unscaled))
    offsets,tau,corrs,startResidual,tRef = correlate_dump(station0,station1,
                                      maxOff=maxOff,width=width,
                                      weight=weight,
                                      progress=progress,
                                      baseOff=baseDelay,
                                      rate=baseRate,idPeak=idPeak,mean=mean,meanSub=meanSub)
    peakOffID = np.argmax(np.abs(corrs).max(0))
    peakOff = offsets[peakOffID]
    ift = np.fft.ifft(np.fft.ifftshift(corrs[:,peakOffID]))
    dDMs = np.linspace(-dDMRange,dDMRange,nDMs)<<u.pc/u.cm**3
    delays = (dDMs[:,np.newaxis]/K)/fOverlap[np.newaxis,:]**2
    corrections = np.exp(-1j*2*np.pi*(delays*fOverlap[np.newaxis,:]).to_value(u.dimensionless_unscaled))
    corrected = corrections*ift
    corrs2 = np.fft.fftshift(np.fft.fft(corrected,axis=1),axes=1)
    dDMref = dDMs[np.argmax(np.max(np.abs(corrs2),1))]
    
    rates=baseRate+np.linspace(-1,1,nRate)*searchRange
    if dDMFitRange>0:
        dDMs = (np.linspace(-1,1,nDMs)*dDMFitRange << (u.pc/u.cm**3))+dDMref
    if line:
        dDMs-=dmSlope*baseRate
    # else:
    #     # dDMs = np.sort((np.linspace(rates.min(),rates.max(),nDMs)-baseRate)*dmSlope + dDMref)

    delays = (dDMs[:,np.newaxis]/K)/fOverlap[np.newaxis,:]**2
    corrections = np.exp(-1j*2*np.pi*((delays[:,:,np.newaxis]+subSearch[np.newaxis,np.newaxis,:]*(tau[1]-tau[0])/2)*fOverlap[np.newaxis,:,np.newaxis]).to_value(u.dimensionless_unscaled))
    peaks = np.zeros((dDMs.shape[0],rates.shape[0]))
    tauOff = np.zeros(peaks.shape)*u.ns
    # taus = np.zeros((dDMs.shape[0],rates.shape[0]))*u.ns
    iterator = enumerate(rates)
    if progress:
        iterator = tqdm(iterator,total=rates.shape[0])
    for i,rate in iterator:
        offsets,tau,corrs,startResidual,tRef = correlate_dump(station0,station1,
                                      maxOff=0,width=width,
                                      weight=weight,
                                      progress=False,
                                      baseOff=peakOff*sampleTime,
                                      rate=rate,idPeak=idPeak,mean=mean,meanSub=meanSub)
        
        ift = np.fft.ifft(np.fft.ifftshift(corrs[:,0]))
        corrected = corrections*ift[np.newaxis,:,np.newaxis]
        if line:
            dDMoff = dmSlope*rate
            corrected*=np.exp(-1j*2*np.pi*((dDMoff/K)/fOverlap[np.newaxis,:,np.newaxis]).to_value(u.dimensionless_unscaled))
        corrs2 = np.abs(np.fft.fftshift(np.fft.fft(corrected,axis=1),axes=1))
        
        peaks[:,i]=corrs2.max((1,2))
        args = np.argwhere(corrs2==peaks[:,i][:,np.newaxis,np.newaxis])
        # print(args,flush=True)
        args = args[np.unique(args[:,0],return_index=True)[1]][:,2]
        # print(args,flush=True)
        tauOff[:,i] = subSearch[args]*(tau[1]-tau[0])/2
        # taus[:,i]=tau[np.argmax(np.abs(corrs2),1)]
    return(peaks**2,dDMs,rates,peakOff,tauOff)


def fit_rate_dm(peaks,dDMs,rates):
    x = np.copy(rates).value
    y = peaks.max(0)
    N0 = y.min()
    A0 = y.max()-N0
    w0 = (x.max()-x.min())/4
    t00 = x[y==y.max()].mean()
    poptR,pcovR=curve_fit(gau1,x,y,p0=np.array([A0,t00,w0,N0]))
    # wts=gau1(x,*poptR)
    # wts/=np.sum(wts)
    rate = poptR[1]*rates.unit
    rateError = np.sqrt(pcovR[1,1])*rates.unit
    x = np.copy(dDMs).value
    # y = np.sum(peaks*wts,1)
    y = peaks.max(1)
    N0 = y.min()
    A0 = y.max()-N0
    w0 = (x.max()-x.min())/4
    t00 = x[y==y.max()].mean()
    poptD,pcovD=curve_fit(gau1,x,y,p0=np.array([A0,t00,w0,N0]))
    if poptD[0]+poptD[-1] < y.max()*.95:
        width = max(2*np.abs(poptD[2]),np.abs(poptD[1]-t00),10*(x[1]-x[0]))
        poptD,pcovD=curve_fit(gau1,x[np.abs(x-t00)<width],y[np.abs(x-t00)<width],p0=poptD)
        # err = (y[np.abs(x-t00)<width]-gau1(x[np.abs(x-t00)<width],*poptD)).std()
        # poptD,pcovD=curve_fit(gau1,x[np.abs(x-t00)<width],y[np.abs(x-t00)<width],p0=poptD,sigma=err,absolute_sigma=True)
    else:
        err = (y-gau1(x,*poptD)).std()
        poptD,pcovD=curve_fit(gau1,x,y,p0=poptD)
        
        
    return(poptR,poptD)


def find_rate_dm(station0,station1,baseDelay,baseRate,
                 searchRange=1e-6*u.s*u.Hz,maxOff=20,dDMRange=1e-5*u.pc/u.cm**3,dDMFitRange=0,
                 plot=False,progress=False,nRate=61,nDMs=1001,width=2048,weight=True,idPeak=None,gauFit = False,showPrediction=True,mean=False,line=False,**kwargs):
    """ 
    Search for dDM and rate that maximize correlation amplitude

    Parameters
    ----------
    station0 : bursttcorr.dumpclass.StationDump
        Dumped data from station 0
    station1 : bursttcorr.dumpclass.StationDump
        Dumped data from station 1
    baseDelay : astropy.Quantity
        reference delay to search around
    maxOff : int
        Number of coarse shifts to search on either side of baseDelay
    baseRate: astropy.Quantity
        Reference rate
    searchFrac : float
        fraction of baseRate to search on either side (searchFrac=1 -> seach from 0 to 2*baseRate)    

    """
    peaks,dDMs,rates,peakOff,tauOffs = rate_dm_peaks(station0,station1,baseDelay,baseRate,
                width,weight,idPeak,mean,maxOff,
                 searchRange,nRate,
                 dDMRange,dDMFitRange,nDMs,
                 progress,line,**kwargs)
    sampleTime = station0.time[1]-station0.time[0]
    offset = peakOff*sampleTime
    print(f'{peakOff} {offset}\n',flush=True)

    idDM = np.argwhere(peaks==peaks.max())[0,0]
    idRate = np.argwhere(peaks==peaks.max())[0,1]
    dDM = dDMs[idDM]
    rate = rates[idRate]
    tauOff = tauOffs[idDM,idRate]
    if gauFit:
        poptR,poptD = fit_rate_dm(peaks,dDMs,rates)
        dDM = poptD[1]*u.pc/u.cm**3
        rate = poptR[1]*u.s*u.Hz
        dDMError = poptD[2]*u.pc/u.cm**3
        rateError = poptR[2]*u.s*u.Hz
        dDMF = poptD[1]*u.pc/u.cm**3
        rateF = poptR[1]*u.s*u.Hz
    else:
        dDMError = 0*u.pc/u.cm**3
        rateError = 0*u.s*u.Hz
        dDMF = 0*u.pc/u.cm**3
        rateF = 0*u.s*u.Hz
    if line:
        dDM-=station0.DM*rate
        dDMF-=station0.DM*rate
    if plot:
        plt.figure()
        plt.imshow(peaks,origin='lower',aspect='auto',extent=ext_find(rates,dDMs))
        plt.xlabel(r'$\rm{rate}~\left(\rm{s}~\rm{Hz}\right)$')
        if line:
            plt.ylabel(r'$\Delta\rm{DM}_0~\left(\rm{pc}~\rm{cm}^{-3}\right)$')
        else:
            plt.ylabel(r'$\Delta\rm{DM}~\left(\rm{pc}~\rm{cm}^{-3}\right)$')
        plt.figure()
        plt.imshow(tauOffs.value,origin='lower',aspect='auto',extent=ext_find(rates,dDMs))
        plt.xlabel(r'$\rm{rate}~\left(\rm{s}~\rm{Hz}\right)$')
        if line:
            plt.ylabel(r'$\Delta\rm{DM}_0~\left(\rm{pc}~\rm{cm}^{-3}\right)$')
        else:
            plt.ylabel(r'$\Delta\rm{DM}~\left(\rm{pc}~\rm{cm}^{-3}\right)$')

    if 'meanSub' in kwargs:
        meanSub=kwargs['meanSub']
    else:
        meanSub=False
    if plot:
        offsets,taus,corrs,startResidual,tRef = correlate_dump(station0,station1,
                                        maxOff=10,width=width,
                                        weight=weight,
                                        progress=False,
                                        baseOff=peakOff*sampleTime,
                                        rate=rate,idPeak=idPeak,mean=mean,dDM=dDM,meanSub=meanSub)
        plot_corr(offsets,taus,corrs)
        peakSNR = np.abs(corrs).max()**2
        tau = taus[np.abs(corrs[:,10])==np.abs(corrs[:,10]).max()].mean()
        tauErr = (taus[1]-taus[0])/2
    else:
        offsets,taus,corrs,startResidual,tRef = correlate_dump(station0,station1,
                                        maxOff=0,width=width,
                                        weight=weight,
                                        progress=False,
                                        baseOff=peakOff*sampleTime,
                                        rate=rate,idPeak=idPeak,mean=mean,dDM=dDM,meanSub=meanSub)
        
        peakSNR = np.abs(corrs).max()**2
        tau = taus[np.abs(corrs[:,0])==np.abs(corrs[:,0]).max()].mean()
        tauErr = (taus[1]-taus[0])/2
    offsets,taus,corrs,startResidual,tRef = correlate_dump(station0,station1,
                                        maxOff=0,width=width,
                                        weight=weight,
                                        progress=False,
                                        baseOff=peakOff*sampleTime,
                                        rate=rateF,idPeak=idPeak,mean=mean,dDM=dDMF,meanSub=meanSub)
        
    corrs = corrs[:,0]
    corrsF = np.fft.ifft(np.fft.ifftshift(corrs))
    fOverlap = get_f_overlap(station0,station1)
    if "subSearch" in kwargs:
        subSearch = kwargs['subSearch']
    else:
        np.linspace(0,0,1)
    corrsF = np.exp(-1j*2*np.pi*((subSearch[:,np.newaxis]*(taus[1]-taus[0])/2)*fOverlap).to_value(u.dimensionless_unscaled))*corrsF
    corrs = np.fft.fftshift(np.fft.fft(corrsF,axis=1),axes=1)
    corrs = np.abs(corrs)
    argmax = np.argmax(corrs)
    tauOffF = subSearch[argmax//taus.shape[0]]*(taus[1]-taus[0])/2
    tauF = taus[np.mod(argmax,taus.shape[0])]
    return(dDM,dDMF,dDMError,rate,rateF,rateError,tau,tauF,tauErr,peakSNR,offset,startResidual,tauOff,tauOffF)



def time_to_year_day(time):
    if time.isscalar:
        year = time.datetime.year
        day = time.datetime.timetuple().tm_yday
    else:
        year = np.array([t.datetime.year for t in time])
        day = np.array([t.datetime.timetuple().tm_yday for t in time])
    return(year,day)

def get_ppp_date(year,day,station):
    pppBaseDir = '/burstt_home/shared/gnss/ppp'
    zipFile = f'{station}-{year}-{str(day).zfill(3)}-full_output.zip'
    zipPath = os.path.join(pppBaseDir,zipFile)
    if not os.path.isfile(zipPath):
        raise FileNotFoundError(f'No ppp for {station} on {year}-{day}')
    with zipfile.ZipFile(zipPath) as z:
        csvFile = [f for f in z.namelist() if f[-4:]=='.csv']
        if len(csvFile)==0:
            raise FileNotFoundError(f'No csv file!')
        csvFile = csvFile[0]
        with z.open(csvFile) as f:
            ppp = np.genfromtxt(f,delimiter = ',')[1:]
            year = ppp[:,-2]
            day = ppp[:,-3]
            hour = ppp[:,-4]
            clock = ppp[:,-1]*u.ns
    pppTime = Time(np.array([f'{int(x)}-01-01T00:00:00' for x in year]))+(day-1)*u.day+hour*u.hour
    return(pppTime[clock!=0],clock[clock!=0])
    
def get_clock_correction(times,station0='FUS',station1='LTN'):
        years,days = time_to_year_day(times+(18*u.s))
        clockCorr = np.zeros(times.shape[0])*np.nan*u.us
        pppTime0,clock0 =  get_ppp_date(years[0],days[0]-1,station0)
        pppTime1,clock1 =  get_ppp_date(years[0],days[0]-1,station1)
        i=1
        while times.min()<pppTime0.min() or times.min()<pppTime1.min():
            years,days = time_to_year_day(times+(18*u.s)-i*u.day)
            pppTime0,clock0 =  get_ppp_date(years[0],days[0],station0)
            pppTime1,clock1 =  get_ppp_date(years[0],days[0],station1)
            i+=1
        interp0 = interp1d(pppTime0.mjd,clock0.to_value(u.us))
        interp1 = interp1d(pppTime1.mjd,clock1.to_value(u.us))
        for i,t in enumerate(times):
            if t+(18*u.s)<max(pppTime0[0],pppTime1[0]) or t+(18*u.s)>min(pppTime0[-1],pppTime1[-1]):
                try:
                    pppTimeA,clockA =  get_ppp_date(years[i],days[i],station0)
                    pppTimeB,clockB =  get_ppp_date(years[i],days[i],station1)
                    pppTime0 = pppTimeA
                    clock0 = clockA
                    pppTime1 = pppTimeB
                    clock1 = clockB
                except:
                    continue
                interp0 = interp1d(pppTime0.mjd,clock0.to_value(u.us))
                interp1 = interp1d(pppTime1.mjd,clock1.to_value(u.us))
            try:
                clockCorr[i] = interp1((t+18*u.s).mjd)*u.us - interp0((t+18*u.s).mjd)*u.us
            except:
                continue
        return(clockCorr)

def _enc(v):
    if isinstance(v,EarthLocation):
        return {'__location__': 1, 'x_m' : v.x.to_value(u.m), 'y_m' : v.y.to_value(u.m), 'z_m' : v.z.to_value(u.m)}
    if isinstance(v, Time):
        return {'__time__': 1, 'jd1': np.asarray(v.jd1).tolist(), 'jd2': np.asarray(v.jd2).tolist(), 'scale': v.scale}
    if isinstance(v, SkyCoord):
        c = v.icrs
        return {'__skycoord__': 1, 'ra_deg': np.asarray(c.ra.deg).tolist(), 'dec_deg': np.asarray(c.dec.deg).tolist()}
    if isinstance(v, u.Quantity):
        return {'__quantity__': 1, 'value': np.asarray(v.value).tolist(), 'unit': v.unit.to_string()}
    if isinstance(v, np.ndarray):  return v.tolist()
    if isinstance(v, np.generic):  return v.item()
    if isinstance(v, dict):        return {k: _enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [_enc(x) for x in v]
    return v

def _dec(v):
    if isinstance(v, dict):
        if v.get('__time__'):      return Time(v['jd1'], v['jd2'], format='jd', scale=v['scale'])
        if v.get('__skycoord__'):  return SkyCoord(ra=np.array(v['ra_deg'])*u.deg, dec=np.array(v['dec_deg'])*u.deg, frame='icrs')
        if v.get('__quantity__'):  return np.array(v['value'])*u.Unit(v['unit'])
        if v.get('__location__'):  return EarthLocation(x=v['x_m']*u.m,y=v['y_m']*u.m,z=v['z_m']*u.m)
        return {k: _dec(x) for k, x in v.items()}
    if isinstance(v, list):        return [_dec(x) for x in v]
    return v

@dataclass
class CorrelationResult:

    ## Event Info
    eventID: str = ''
    stations: tuple = ()
    
    ## Source info
    sourceName: str = ''
    sourceCoords: Optional[SkyCoord] = None
    type: Optional[str] = None

    ## Reference
    tRef: Optional[Time] = None
    modelDelay: Optional[u.Quantity] = None
    modelRate: Optional[u.Quantity] = None
    clockCorr: Optional[dict] = None
    refDM: Optional[u.Quantity] = None
    stationCoords: Optional[dict] = None
    

    ## Results
    coarseDelay: Optional[u.Quantity] = None
    tauInf: Optional[u.Quantity] = None
    tauInfErr: Optional[u.Quantity] = None
    nuRef: Optional[u.Quantity] = None
    tauRef: Optional[u.Quantity] = None
    tauRefErr: Optional[u.Quantity] = None
    dDM: Optional[u.Quantity] = None
    dDMErr: Optional[u.Quantity] = None
    phi0: Optional[float] = None
    phi0Err: Optional[float] = None
    tInfdDMCov: Optional[u.Quantity] = None

    ## Diagnostics
    SNR: Optional[float] = None

    ## Other
    extra: dict = field(default_factory=dict)
    
    @property
    def totalDelay(self):
        return (self.coarseDelay + self.tauInf).to(u.ns)

    @property
    def defaultFile(self):
        return(f'{self.eventID}_{self.sourceName}_{self.stations[0]}-{self.stations[1]}.json')

    def save_json(self,**kwargs):
        if "filename" in kwargs:
            filename = kwargs['filename']
        else:
            filename = self.defaultFile
        with open(filename, 'w', encoding='utf-8') as fh:
            json.dump(_enc(asdict(self)), fh, indent=2)

    @classmethod
    def from_json(cls, filename):
        with open(filename, encoding='utf-8') as fh:
            d = _dec(json.load(fh))
        names = {f.name for f in fields(cls)}
        extra = {k: d.pop(k) for k in list(d) if k not in names}   # tolerate older or newer files
        if 'stations' in d and isinstance(d['stations'], list):
            d['stations'] = tuple(d['stations'])
        obj = cls(**d)
        obj.extra.update(extra)
        return obj