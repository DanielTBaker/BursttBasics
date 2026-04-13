from .dumpclass import *
import astropy.units as u
from ..burstt_func import *
import numpy as np
from scipy.optimize import curve_fit
from .plotting import *
import zipfile


def correlate_dump(s0,s1,
                   baseOff=0,maxOff=100,
                   width=2048,weight=True,idPeak=None,
                   rate=0*u.s*u.Hz,dDM = 0*u.pc/u.cm**3,
                   progress=False,leave=True,sampleTime = 2.56*u.us
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
    f0 = s0.freq
    f1 = s1.freq
    use0 = (f0>=f1.min())*(f0<=f1.max())
    use1 = (f1>=f0.min())*(f1<=f0.max())
    if weight:
        modelT = gau1(s0.time.to_value(u.ms),*s0.poptT)-s0.poptT[-1]
        modelF = s0.freq_function(f0.value,*s0.poptF)-s0.poptF[-1]
        idPeak = np.argmax(modelT)
        id0 = idPeak-width//2
        model = modelF[use0][:,np.newaxis]*modelT[id0:id0+width]
    elif idPeak == None:
        id0=maxOff+1
        if baseOff<0:
            id0-=baseOff
        idPeak = id0+width//2
    else:
        id0=idPeak-width//2

    spec0 = np.conjugate(s0.spec[use0,id0:id0+width])
    if not rate==0:
        t = s0.time[id0:id0+width][np.newaxis,:]+(s0.ep0[use0]-s0.ep0[0])[:,np.newaxis]*u.s -idPeak*sampleTime
        f = s0.freq[use0]*1
        spec0 *= np.exp(-1j*2*np.pi*(f[:,np.newaxis]*(rate*t)).to_value(u.dimensionless_unscaled))
    if not dDM ==0:
        dP = ((dDM/K)/s0.freq[use0]).to_value(u.dimensionless_unscaled)
        spec0 *= np.exp(-1j*2*np.pi*dP)[:,np.newaxis]
    
    startDiff = get_start_off(s0,s1)
    startOff = int(np.round(startDiff/sampleTime))
    # if isinstance(s0.ep0,np.ndarray):
    #     spec0*=np.exp(-1j*2*np.pi*(s0.freq[use0]*(s1.ep0[use1]*u.s+startOff*sampleTime-s0.ep0[use0]*u.s)).to_value(u.dimensionless_unscaled))[:,np.newaxis]

    startResidual = startDiff - startOff*sampleTime
    offsets = np.linspace(-maxOff,maxOff,2*maxOff+1,dtype=int)+baseOff
    corrs = np.zeros((spec0.shape[0],offsets.shape[0]),dtype=np.complex64)
    cross = np.zeros(spec0.shape,dtype=spec0.dtype)
    iterator = enumerate(offsets)
    if progress:
        iterator=tqdm(iterator,total=2*maxOff+1,leave=leave)
    
    try:
        for ido,o in iterator:
            cross[:,:] = spec0*s1.spec[use1,id0+o+startOff:id0+o+startOff+width]
            cross/=np.abs(cross)
            if weight:
                cross/=np.abs(cross)
                cross*=model
            cross-=np.nanmean(cross)
            fCross = np.fft.fft2(np.nan_to_num(cross))
            fAmp = np.abs(fCross)
            corrs[:,ido] = np.fft.fftshift(fCross[:,0])/np.median(fAmp)
    except Exception as e:
        print('----')
        print(f'Total Size: {s1.spec.shape[1]}')
        print(f'Start Offset: {startOff}')
        print(f'Cut Start: {id0}')
        print(f'Max Off: {maxOff}')
        print(f'Base Off: {baseOff}')
        print('----')
        raise e

    tau = thth.fft_axis(f0[use0],u.ns)
    tRef = Time(s0.ep0[0],format='unix')+idPeak*sampleTime
    return(offsets,tau,corrs,startResidual,tRef)


def find_rate_dm(station0,station1,baseDelay,baseRate,
                 searchFrac=.1,maxOff=20,dDMRange=1e-5*u.pc/u.cm**3,dDMFitRange=0,
                 plot=False,progress=False,nRate=61,nDMs=1001,width=2048,weight=True,gauFit = False,showPrediction=True):
    dmSlope = -station0.DM
    fOverlap = get_f_overlap(station0,station1)
    baseOff = int((baseDelay//(2.56*u.us)).to_value(u.dimensionless_unscaled))
    offsets,tau,corrs,startResidual,tRef =correlate_dump(station0,station1,
                                      maxOff=maxOff,width=width,
                                      weight=weight,
                                      progress=progress,
                                      baseOff=baseOff,
                                      rate=baseRate)
    if plot:
        plot_corr(offsets,tau,corrs)
    peakOffID = np.argmax(np.abs(corrs).max(0))
    peakOff = offsets[peakOffID]
    ift = np.fft.ifft(np.fft.ifftshift(corrs[:,peakOffID]))
    dDMs = np.linspace(-dDMRange,dDMRange,nDMs)<<u.pc/u.cm**3
    delays = (dDMs[:,np.newaxis]/K)/fOverlap[np.newaxis,:]**2
    corrections = np.exp(-1j*2*np.pi*(delays*fOverlap[np.newaxis,:]).to_value(u.dimensionless_unscaled))
    corrected = corrections*ift
    corrs2 = np.fft.fftshift(np.fft.fft(corrected,axis=1),axes=1)
    dDMref = dDMs[np.argmax(np.max(np.abs(corrs2),1))]
    
    rates=np.sort(np.linspace((1-searchFrac)*baseRate,(1+searchFrac)*baseRate,nRate))
    if dDMFitRange>0:
        dDMs = (np.linspace(-1,1,nDMs)*dDMFitRange << (u.pc/u.cm**3))+dDMref
    else:
        dDMs = np.sort((np.linspace(rates.min(),rates.max(),nDMs)-baseRate)*dmSlope + dDMref)
    delays = (dDMs[:,np.newaxis]/K)/fOverlap[np.newaxis,:]**2
    corrections = np.exp(-1j*2*np.pi*(delays*fOverlap[np.newaxis,:]).to_value(u.dimensionless_unscaled))
    peaks = np.zeros((dDMs.shape[0],rates.shape[0]))
    taus = np.zeros((dDMs.shape[0],rates.shape[0]))*u.ns
    iterator = enumerate(rates)
    if progress:
        iterator = tqdm(iterator,total=rates.shape[0])
    for i,rate in iterator:
        offsets,tau,corrs,startResidual,tRef = correlate_dump(station0,station1,
                                      maxOff=0,width=width,
                                      weight=weight,
                                      progress=False,
                                      baseOff=peakOff,
                                      rate=rate)
        ift = np.fft.ifft(np.fft.ifftshift(corrs[:,0]))
        corrected = corrections*ift
        corrs2 = np.fft.fftshift(np.fft.fft(corrected,axis=1),axes=1)
        peaks[:,i]=np.abs(corrs2).max(1)**2
        taus[:,i]=tau[np.argmax(np.abs(corrs2),1)]
    rate = rates[np.argmax(peaks.max(0))]
    dDM = dDMs[np.argmax(peaks.max(1))]
    tau = taus[peaks==peaks.max()].mean()
    if gauFit:
        x = np.copy(rates).value
        y = peaks.max(0)
        N0 = y.min()
        A0 = y.max()-N0
        w0 = (x.max()-x.min())/4
        t00 = x[y==y.max()].mean()
        poptR,pcovR=curve_fit(gau1,x,y,p0=np.array([A0,t00,w0,N0]))
        rate = poptR[2]*rates.unit
        rateError = np.sqrt(pcovR[2,2])*rates.unit
        x = np.copy(dDMs).value
        y = peaks.max(1)
        N0 = y.min()
        A0 = y.max()-N0
        w0 = (x.max()-x.min())/4
        t00 = x[y==y.max()].mean()
        poptD,pcovD=curve_fit(gau1,x,y,p0=np.array([A0,t00,w0,N0]))
        dDM = poptD[2]*dDMs.unit
        dDMError = np.sqrt(pcovD[2,2])*dDMs.unit
        tauErr = 1.25*u.ns
    else:
        if peaks.min()<peaks.max()-1:
            rateError = np.abs(rates[peaks.max(0)>=peaks.max()-1]-rate).max()
            dDMError = np.abs(dDMs[peaks.max(1)>=peaks.max()-1]-dDM).max()
            tauErr = np.abs(taus[peaks>=peaks.max()-1]-tau).max()
        else:
            rateError = np.nan*u.s*u.Hz
            dDMError = np.nan*u.pc/u.cm**3
            tauErr=np.nan*u.ns
    peakSNR = peaks.max()
    
    if plot:
        
        grid=plt.GridSpec(nrows=2,ncols=2,
                          width_ratios=[4,1],height_ratios=[1,4],
                          hspace=0,wspace=0)
        ext=ext_find(rates,dDMs)
        plt.figure()
        plt.subplot(grid[1,0])
        plt.imshow(peaks,origin='lower',aspect='auto',extent=ext)
        plt.contour(rates.value,dDMs.value,peaks,[peaks.max()-2,peaks.max()-1],colors=['red','red'])
        if showPrediction:
            plt.plot(rates,(rates-rate)*dmSlope + dDM,'r')
        plt.ylim(ext[2:])
        plt.xlabel(r'$\rm{Rate}~\left(\rm{s}~\rm{Hz}\right)$')
        plt.ylabel(r'$\Delta\rm{DM}~\left(\rm{pc}~\rm{cm}^{-3}\right)$')
        plt.subplot(grid[0,0])
        plt.plot(rates,peaks.max(0))
        plt.xticks([])
        plt.xlim(ext[:2])
        plt.ylabel('Peak SNR')
        if gauFit:
            plt.plot(rates,gau1(rates.value,*poptR))
        plt.subplot(grid[1,1])
        plt.plot(peaks.max(1),dDMs)
        if gauFit:
            plt.plot(gau1(dDMs.value,*poptD),dDMs)
        plt.yticks([])
        plt.xlabel('Peak SNR')
        plt.ylim(ext[2:])
    offset = peakOff*2.56*u.us
    return(dDM,dDMError,rate,rateError,tau,tauErr,peakSNR,offset,startResidual)


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
        years,days = time_to_year_day(times+(18*u.s)-1*u.day)
        clockCorr = np.zeros(times.shape[0])*np.nan*u.us
        pppTime0,clock0 =  get_ppp_date(years[0],days[0],station0)
        pppTime1,clock1 =  get_ppp_date(years[0],days[0],station1)
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