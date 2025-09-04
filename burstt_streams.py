from astropy.time import Time
import astropy.units as u
import numpy as np
import os
import sys


from packet_func import *
from burstt_func import *
from burstt_const import *
from tqdm.notebook import tqdm

from pint.models import get_model
from pint.polycos import Polycos
from  pint.observatory.topo_obs import TopoObs
from astropy.coordinates import EarthLocation,SkyCoord
import pint.logging
from pycalc11 import Calc

import pickle as pkl

class Stream:
    """
    Represent and manipulate a series of file representing
    a single BURSTT observation from a single station.

    Parameters
    ----------
    config: ConfigParser
        Configuration data for observation
    station: str
        Station name for this observation. Usually one of:
        ('FUS','LTN')
    notebook: bool, optional
        Running in notebook and progress bars are desired. Defaults to False
    progressOrder: int, optional
        Base recurision level for progress bars. Defaults to -1 for no bars
    """

    def __init__(self,config,name,notebook=False,progressOrder=-1):
        ## Initialize observation object and load config parameters
        self.notebook = notebook
        ##Station
        self.name = name

        self._read_config(config)


    def _get_block_byte(self):
        self.blockByte = 0

    def _read_config(self,config):
        self.rebin=1
        self.analysisPars =  config['Analysis Properties']
        self.observationPars = config['Observation Properties']
        self.sourcePars = config['Source Properties']
        self.taskPars = config['Tasks']

        self.verbose = self.analysisPars.getboolean('verbose',False)

        self.sourceType = self.sourcePars.get('type')
        self.sourceName = self.sourcePars.get('name')

        self.obsDate = self.observationPars.get('date')

        self.rootDir = self.analysisPars.get('outDir','./')
        self.dateDir = os.path.join(self.rootDir,f'{self.obsDate}_{self.sourceName}')
        self.plotDir = os.path.join(self.dateDir,'plots')
        self.cutDir = os.path.join(self.dateDir,'Data','cuts')
        self.intensityDir = os.path.join(self.dateDir,'Data','intensities')
        self.correlationDir = os.path.join(self.dateDir,'Data','correlations')

        self.station = config.get(self.name,"station")

        self.headSize = 64
        ## Reference Time
        self.epoch0 = Time.now()

        ## Location of data files
        self.dataDir = './'

        ## Files per Ring Per Epoch:
        self.nf = 0

        ## Length of frame (us)
        self.frameSec = 0*u.us
       

        ## Number of channels in full spectrum
        self.nChan0 = 0

        ## Number of blocks per file
        self.nBlock = 0
        
        self.blockFrame = 0

        ## Frequency Information
        self.f0 = 0*u.MHz
        self.bandWidth = 0*u.MHz
        self.freq = np.linspace(self.f0,self.f0+self.bandWidth,self.nChan0,endpoint=False)

        ## DM (Not always required)
        self.DM = 0 * u.pc/u.cm**3

        self.nOrder = 0

        ## Channels per subband
        self.nChan = 0
        self._get_block_byte()
        

    def get_block_from_time(self,time):
        """Determine epoch and block number containing a given time.

        Parameters
        ----------
        time: Astropy time quantity, or Time object
            Time to determine location of
        
        Returns
        -------
        epoch: int
            Epoch ID of block containing time.
        block: int
            Block ID within epoch containing time.
        
        """
        ## Given a time, determine the epoch and block containing that time
        if isinstance(time,Time):
            tRef = time.to_value('unix')*u.s
        else:
            tRef = self.epoch0.to_value('unix')*u.s + time
        if tRef>np.nanmax(self.blockStart):
            return(self.nEpochs-1,self.nf*self.nBlock -1)
        if tRef<np.nanmin(self.blockStart):
            return(0,0)
        epochStarts = self.blockStart[:self.nEpochs,0,0]
        epoch = epochStarts[epochStarts<tRef].shape[0]-1
        block = self.blockStart[epoch][self.blockStart[epoch]<tRef].shape[0]-1
        return(epoch,block)

    def load_chain(self,order,segment,verbose=False):
        files = []
        bID = np.zeros(0)
        fID = 0
        bID2 = bID
        spec  = np.zeros((segment[2], self.blockFrame, self.nChan), dtype=np.complex64)
        pcnt  = np.zeros(segment[2], dtype=np.int64)
        epoch = np.zeros(segment[2])
        return(spec, pcnt, epoch)
            
    def read_order(self,epoch,startBlock,length,order,verbose=False):
        """Read a continuous series of blocks for a single subband.

        Parameters
        ----------
        epoch: int
            Epoch ID of starting block
        startBlock: int
            Block ID of starting block
        length: int
            Number of blocks to read.
        Order: int
            Subband ID.

        Returns
        -------
        spec: array-like
            Voltage spectrum for the given suband spanning the given blocks
        """

        ## Read "length" blocks of data starting at the "startBlock"th block in "epoch" for a given subband
        segments = [[epoch,startBlock,length]]
        while segments[-1][1]+segments[-1][2]>self.nBlock*self.nf:
            newEpoch = segments[-1][0]+1
            newStarts = 0
            newLength = segments[-1][1]+segments[-1][2] - self.nBlock*self.nf
            segments[-1][-1]-=newLength
            segments.append([newEpoch,newStarts,newLength])
        try:
            spec, pcnt, falseEpoch = self.load_chain(order,segments[0],verbose)
        except Exception as e:
            print(f'{segments} ({self.nEpochs*order+segments[0][0]}/{len(self.fbystamp)}):{e}')
            spec = np.zeros((int(self.blockFrame*length/self.rebin),self.nChan),dtype=np.complex64)*np.nan
        for i in range(len(segments)-1):
            try:
                spec2, pcnt, falseEpoch = self.load_chain(order,segments[i+1],verbose)
            except Exception as e:
                print(f'{segments} ({self.nEpochs*order+segments[0][0]}/{len(self.fbystamp)}):{e}')
                spec2 = np.zeros((int(self.blockFrame*length/self.rebin),self.nChan),dtype=np.complex64)*np.nan
            spec = np.concatenate((spec,spec2),axis=0)
        return(spec)

    def basic_spec(self,nChans,nFrames):
        spec = np.zeros((nFrames,nChans),dtype=np.complex64)
        return(spec)
    
    def read_around(self,tCent,tWidth,verbose=False):
        """Read and dedisperse all subands to cover a given time range at the reference frequency

        Parameters
        ----------
        fref: Astropy.Quantity
            Reference frequency for dedispersion
        tCent: Astropy.Time of Astropy.Quanity
            Center of desired time range at reference frequency.
        """
        if isinstance(tCent,Time):
            tRef = tCent
        else:
            tRef = self.epoch0 + tCent
        tEnd = tRef + tWidth
        tStart = tRef - tWidth
        if verbose:
            print(f'Read from {tStart} to {tEnd}')
        nFrames = self.time_to_frames(2*tWidth)
        e0,b0 = self.get_block_from_time(tStart)
        e1,b1 = self.get_block_from_time(tEnd)
        if verbose:
            print(f'Starting: ({e0,b0})')
            print(f'Ending: ({e1,b1})')
        t0 = self.blockStart[e0,b0//self.nf,np.mod(b0,self.nf)]
        spec = self.basic_spec(self.nChan0,nFrames)
        if verbose:
            print(f'Spec Shape: {spec.shape}')
        try:
            for order in range(self.nOrder):
                pxStart = self.time_to_frames(tStart.unix*u.s-t0)
                orderSpec = self.read_order(e0,b0,(e1-e0)*self.nf*self.nBlock + b1-b0+1,order)
                spec[:,order*self.nChan:(order+1)*self.nChan]=orderSpec[pxStart:pxStart+nFrames]
            return(spec,t0+pxStart*self.frameSec*self.rebin)
        except Exception as e:
            print(f'In read_around: {e}')
            return(spec*np.nan,tStart)

    def read_order_dedis(self,epoch,startBlock,length,order,pxStart,nFrame,baseOff,verbose=False):
        """Read and desdisperse a section of data for a given subband

        Parameters
        ----------
        epoch: int
            Epoch ID of starting block
        startBlock: int
            Block ID of starting block
        length: int
            Number of blocks to read.
        Order: int
            Subband ID.
        pxStart: array-like
            Number of frames offset from reference frequency at each frequency
        nFrame: int
            Number of frames in dedispersed array
        baseOff:
            Number of frames offset from start of block containing start time at
            reference frequency to start of starting block
        Returns
        -------
        spec: array-like
            Voltage spectrum for the given suband
        """
        ## As above, but incoherently dedisperses and only saves a portion of the data
        spec = self.basic_spec(self.nChan,nFrame+20)
        e = epoch
        b = startBlock
        px = np.linspace(0,int(self.blockFrame/self.rebin)-1,int(self.blockFrame/self.rebin),dtype=int)+baseOff
        pxEnd = np.round(pxStart).astype(int)+nFrame+20
        pxStartI = np.round(pxStart).astype(int)
        for bid in range(length):
            bspec, pcnt, falseEpoch  = self.load_chain(order,[e,b,1],verbose=verbose)
            for i in range(bspec.shape[2]):
                if (pxEnd[i] > px[0]) and (pxStartI[i] < px[-1]):
                    ss = max(0,px[0]-pxStartI[i])
                    se = min(nFrame+20,px[-1]-pxStartI[i]+1)
                    bss = max(0,pxStartI[i]-px[0])
                    bse = min(int(self.blockFrame/self.rebin),pxEnd[i]-px[0])
                    spec[ss:se,i] = bspec[0,bss:bse,i]
            px+=int(self.blockFrame/self.rebin)
            b+=1
            if b==self.nBlock*self.nf:
                b=0
                e+=1
        return(spec)

    def time_to_frames(self,time):
        if isinstance(time,np.ndarray):
            frames = (time/(self.frameSec*self.rebin)).to_value(u.dimensionless_unscaled).astype(int)
        else:
            frames = int((time/(self.frameSec*self.rebin)).to_value(u.dimensionless_unscaled))
        return(frames)

    def read_around_dedis(self,fref,tCent,tWidth,coherent = True):
        """Read and dedisperse all subands to cover a given time range at the reference frequency

        Parameters
        ----------
        fref: Astropy.Quantity
            Reference frequency for dedispersion
        tCent: Astropy.Time of Astropy.Quanity
            Center of desired time range at reference frequency.
        """
        
        ## Convert reference time to unix
        if isinstance(tCent,Time):
            tRef = tCent
        else:
            tRef = self.epoch0 + tCent
        tEnd = tRef + tWidth
        tStart = tRef - tWidth
        ## Calculate dispersion delay for each frequency
        try:
            dispersionDelay = ((self.DM/K)*(1/self.freq**2 - 1/fref**2)).to(u.s)
        except:
            dispersionDelay = np.zeros(self.freq.shape[0])*u.s
        ## Calculate number of frames in final spectrum
        nFrames = self.time_to_frames(2*tWidth)
    
        ## Find start and end at each frequency
        tStartDispersed = tStart+dispersionDelay
        tEndDispersed = tEnd+dispersionDelay
        ## get epoch and block for start time
        eRef,bRef = self.get_block_from_time(tStart)
        ## Time of first frame
        t0 = self.blockStart[eRef,bRef//self.nf,np.mod(bRef,self.nf)]
        spec = self.basic_spec(self.nChan0,nFrames)
        try:
            for order in range(self.nOrder):
                e0,b0 = self.get_block_from_time(tStartDispersed[(order+1)*self.nChan-1])
                e1,b1 = self.get_block_from_time(tEndDispersed[order*self.nChan])
                baseOff = int(((e0-eRef)*self.nf*self.nBlock + b0-bRef)*self.blockFrame/self.rebin)
                pxStart = self.time_to_frames((tStartDispersed[order*self.nChan:(order+1)*self.nChan].unix*u.s-t0))
                orderSpec = self.read_order_dedis(e0,b0,(e1-e0)*self.nf*self.nBlock + b1-b0+1,order,pxStart,nFrames,baseOff)
                if coherent:
                        fSpec = np.fft.fft(orderSpec[:nFrames+2],axis=0)
                        fftfreq = np.fft.fftfreq(nFrames+2)
                        if len(spec.shape)==2:
                            fSpec *= np.exp(1j*2*np.pi*fftfreq[:,np.newaxis]*(pxStart-np.round(pxStart))[np.newaxis,:])
                        else:
                            fSpec *= np.exp(1j*2*np.pi*fftfreq[:,np.newaxis]*(pxStart-np.round(pxStart))[np.newaxis,:])[:,:,np.newaxis]
                        spec[:,order*self.nChan:(order+1)*self.nChan] = np.fft.ifft(fSpec,axis=0)[:nFrames]
                else:
                    spec[:,order*self.nChan:(order+1)*self.nChan] = orderSpec[:nFrames]
                        
                    # print(f'Dedisperse in {(Time.now()-ts).to_value(u.min):.1f} minutes',flush=True)
            return(spec,tStart)
        except Exception as e:
            print(f'In read_around_dedis({sys.exc_info()[2].tb_lineno}): {e}')
            return(spec*np.nan,tStart)

    def _pulsar_prep(self,config):
        self.ToAs = []
        self.pulseWidth = self.sourcePars.getfloat('pulseWidth')*u.ms

        self.polyCoFreq = self.sourcePars.getfloat('polyCoFreq')*u.MHz
        self.parFile = self.sourcePars.get('parFile')
        self.ToAFile = config[self.name].get('ToAFile',
                                           os.path.join(self.dateDir,
                                                        f'ToAs_{self.obsDate}_{self.sourceName}_{self.name}.npy'))
        self.profileFile = config[self.name].get('profile',os.path.join(self.dateDir,
                                                        f'profile_{self.obsDate}_{self.sourceName}_{self.name}.npy'))
        self.polycoFile = os.path.join(self.dateDir,
                                       f'polyco_{self.obsDate}_{self.sourceName}_{self.name}.dat')
        if os.path.exists(self.polycoFile):
            self.polyco = Polycos.read(self.polycoFile)
        else:
            fushanObs = TopoObs(self.station, location = globals()[f'{self.station}Center'],origin='/home/dbaker/BursttBasics',overwrite=True)
            if self.verbose:
                print('Prep PolyCos')
            model = get_model(self.parFile)
            self.polyco = Polycos.generate_polycos(model,
                                              np.floor(Time(np.nanmin(self.blockStart).value,format='unix').mjd),
                                              np.ceil(Time(np.nanmax(self.blockStart).value,format='unix').mjd + self.frameSec.to_value(u.day)*self.blockFrame)+1,
                                              self.station,
                                              144, 12,
                                              self.polyCoFreq.value,progress=True)
            self.polyco.write_polyco_file(self.polycoFile)
        self.ToA_check() 
        
    def ToA_check(self):
        if len(self.ToAs)==0:
            if os.path.exists(self.ToAFile):
                self.ToAs = np.load(self.ToAFile)
            elif self.sourcePars.get('referenceTime'):
                self.ToAs_from_tref(Time(self.sourcePars.get('referenceTime')))
            elif self.sourcePars.get('referencePhase'):
                self.ToAs_from_pref(self.sourcePars.getfloat('referencePhase'))
            elif os.path.exists(self.profileFile):
                self.profile = np.load(self.profileFile)
                self.profilePhases = np.linspace(-.5,.5,self.profile.shape[1]+1)
                self.profilePhases = (self.profilePhases[1:]+self.profilePhases[:-1])/2
                self.ToAs_from_prof()

    def ToAs_from_prof(self):
        v = self.profile.var(1)
        v2 = sigma_clip(v, sigma=2.0)
        self.profileFlat = np.mean(self.profile[~v2.mask],0)
        self.ToAs_from_pref(self.profilePhases[np.argmax(self.profileFlat)]) 

    def ToAs_from_tref(self,tref):
        phaseRef = self.polyco.eval_phase(tref.mjd)[0].value
        self.ToAs_from_pref(phaseRef)
        
    def ToAs_from_pref(self,phaseRef):
        t = Time(np.nanmin(self.blockStart.value),format='unix').mjd+self.pulseWidth.to_value(u.day)
        phase = self.polyco.eval_phase(t)[0].value
        p = (1./(self.polyco.eval_spin_freq(t)[0]*u.Hz)).to_value(u.day)
        if phase>phaseRef:
            t = t + p*(1-phase)
        while np.abs(phase - phaseRef)>1e-7:
            p = (1./(self.polyco.eval_spin_freq(t)[0]*u.Hz)).to_value(u.day)
            t += p*(phaseRef-phase)
            phase = self.polyco.eval_phase(t)[0].value
        self.ToAs = []
        while t + self.pulseWidth.to_value(u.day) < Time(np.nanmax(self.blockStart).value,format='unix').mjd + self.frameSec.to_value(u.day)*self.blockFrame:
            self.ToAs.append(t)
            p = p = (1./(self.polyco.eval_spin_freq(t)[0]*u.Hz)).to_value(u.day)
            t += p
            phase = self.polyco.eval_phase(t)[0].value
            while np.abs(phase - phaseRef)>1e-7:
                p = p = (1./(self.polyco.eval_spin_freq(t)[0]*u.Hz)).to_value(u.day)
                t += p*(phaseRef-phase)
                phase = self.polyco.eval_phase(t)[0].value
        self.ToAs = Time(np.array(self.ToAs),format='mjd').unix
        np.save(self.ToAFile,self.ToAs)

    def fold_block(self,fref,epoch,startBlock,nBins = 128,upChan = 1):
        try:
            if upChan==1:
                dispersionDelay = ((self.DM/K)*(1/self.freq**2 - 1/fref**2)).to(u.s)
            else:
                freqUP = np.linspace(self.freq[0],self.freq[-1],self.nChan0*upChan)
                dispersionDelay = ((self.DM/K)*(1/freqUP**2 - 1/fref**2)).to(u.s)
        except:
            dispersionDelay = np.zeros(self.freq.shape[0]*upChan)*u.s
        folded = np.zeros((self.nChan0*upChan,nBins))
        wts = np.zeros((self.nChan0*upChan,nBins))
        spec = np.zeros((self.blockFrame//upChan,self.nChan0*upChan))
        for order in range(self.nOrder):
            if upChan == 1:
                spec[:,order*self.nChan:(order+1)*self.nChan] = np.abs(self.read_order(epoch,startBlock,1,order))**2
            else:
                base = self.read_order(epoch,startBlock,1,order)
                baseF = np.fft.fftshift(np.fft.fft(base,axis=1),axes=1)
                baseF = np.reshape(baseF,(baseF.shape[0]//upChan,baseF.shape[1]*upChan))
                base = np.fft.fft(baseF,axis=1)
                spec[:,order*self.nChan*upChan:(order+1)*self.nChan*upChan]=np.abs(base)[:,::-1]**2
        t = np.linspace((upChan-1)/2.,self.blockFrame - (upChan+1)/2,self.blockFrame//upChan)*self.frameSec
        t += self.blockStart[epoch,int(startBlock//self.nf),np.mod(startBlock,self.nf)]
        times = Time(t.to_value(u.s),format='unix')
        phaseRef = self.polyco.eval_phase(times.mjd)
        pulseFreq = self.polyco.eval_spin_freq(times.mjd).mean()*u.Hz
        for i in range(self.nChan0*upChan):
            column = spec[:,i]
            pShift = (dispersionDelay[i]*pulseFreq).to(u.dimensionless_unscaled)
            phases =  np.mod(phaseRef-pShift+.5,1)-.5
            folded[i,:],__ = np.histogram(phases[np.isfinite(column)],weights=column[np.isfinite(column)],bins=np.linspace(-.5,.5,nBins+1))
            wts[i,:],__ = np.histogram(phases[np.isfinite(column)],bins=np.linspace(-.5,.5,nBins+1))
        return(folded,wts)
            

class Baseband(Stream):
    """
    Represent and manipulate a series of files representing
    a single BURSTT Baseband recording from a single station.

    Parameters
    ----------
    config: ConfigParser
        Configuration data for observation
    station: str
        Station name for this observation. Usually one of:
        ('FUS','LTN')
    notebook: bool, optional
        Running in notebook and progress bars are desired. Defaults to False
    progressOrder: int, optional
        Base recurision level for progress bars. Defaults to -1 for no bars
    """

    def __init__(self,config,name,notebook=False,progressOrder=-1):
        super().__init__(config,name,notebook,progressOrder)
        if self.sourceType == 'pulsar':
            self._pulsar_prep(config)

    def _get_block_byte(self):
        self.blockByte = self.blockFrame * self.nChan

    def _read_config(self,config):
        self.analysisPars =  config['Analysis Properties']
        self.observationPars = config['Observation Properties']
        self.sourcePars = config['Source Properties']
        self.taskPars = config['Tasks']


        self.verbose = self.analysisPars.getboolean('verbose',False)

        self.sourceType = self.sourcePars.get('type')
        self.sourceName = self.sourcePars.get('name')

        self.obsDate = self.observationPars.get('date')

        self.rootDir = self.analysisPars.get('outDir','./')
        self.dateDir = os.path.join(self.rootDir,f'{self.obsDate}_{self.sourceName}')
        self.plotDir = os.path.join(self.dateDir,'plots')
        self.cutDir = os.path.join(self.dateDir,'Data','cuts')
        self.intensityDir = os.path.join(self.dateDir,'Data','intensities')
        self.correlationDir = os.path.join(self.dateDir,'Data','correlations')

        self.station = config.get(self.name,"station")
        self.headSize = config[self.name].getint('headSize',64)
        ## Reference Time
        self.epoch0 = Time(config.get('Observation Properties', 'date') + ' 16:00:00')

        # ## Files per Ring Per Epoch:
        # self.nf = config.getint(self.name, 'ringFiles')

        ## Length of frame (us)
        if 'frameSec'.lower() in config.options(self.name):
            self.frameSec = config.getfloat(self.name, 'frameSec')*u.us
        else:
            self.frameSec = config.getfloat('Observation Properties', 'frameSec')*u.us

        ## Number of channels in full spectrum
        if 'nChan0'.lower() in config.options(self.name):
            self.nChan0 = config.getint(self.name, 'nChan0')
        else:
            self.nChan0 = config.getint('Observation Properties', 'nChan0')

        ## Number of blocks per file
        if 'nBlock'.lower() in config.options(self.name):
            self.nBlock = config.getint(self.name, 'nBlock')
        else:
            self.nBlock = config.getint('Observation Properties', 'nBlock')

        ## Number of frames per block
        if 'blockFrame'.lower() in config.options(self.name):
            self.blockFrame = config.getint(self.name, 'blockFrame')
        else:
            self.blockFrame = config.getint('Observation Properties', 'blockFrame')

        ## Frequency Information
        self.f0 = config.getfloat(self.name, 'f0')*u.MHz
        if 'bandWidth'.lower() in config.options(self.name):
            self.bandWidth = config.getfloat(self.name, 'bandWidth')*u.MHz
        else:
            self.bandWidth = config.getfloat('Observation Properties', 'bandWidth')*u.MHz
        self.freq = np.linspace(self.f0,self.f0+self.bandWidth,self.nChan0,endpoint=False)
        self.freq -= (self.freq[1]-self.freq[0])*config.getfloat(self.name, 'fOffset')

        ## DM (Not always required)
        self.DM = config['Source Properties'].getfloat('DM',0) * u.pc/u.cm**3

        self.nOrder = config.getint(self.name, 'nOrder')

        ## Channels per subband
        self.nChan = int(self.nChan0//self.nOrder)

        self._get_block_byte()
        self.rebin=1

        self.dataDir = config.get(self.name,"dataDir")
        self.machines = config.get(self.name,"machines").split(',')
        self.rings : list[Ring] = []
        for machine in self.machines:
            for ring in range(2):
                ringFile = os.path.join(self.dateDir,f'{machine}_ring{ring}.pkl')
                if os.path.exists(ringFile):
                    self.rings.append(Ring.from_pickle(ringFile))
                else:
                    dataDir = f'/burstt_home/shared/{self.station}/{machine}/{self.dataDir}/'
                    fileList = [os.path.join(dataDir,x) for x in os.listdir(dataDir) if (f'ring{ring}' in x)]
                    self.rings.append(Ring(machine, ring, self.nBlock, fileList, self.blockByte, self.frameSec,self.nChan,self.blockFrame,self.headSize))
                    with open(ringFile, 'wb') as f:
                        pkl.dump(self.rings[-1], f, pkl.HIGHEST_PROTOCOL)
        self.earliest = Time(max([x.earliest for x in self.rings]).value, format='unix')
        self.latest = Time(min([x.latest for x in self.rings]).value, format='unix')

    def fold_block(self,fref,block,nBins = 128,upChan = 1):
        try:
            if upChan==1:
                dispersionDelay = ((self.DM/K)*(1/self.freq**2 - 1/fref**2)).to(u.s)
            else:
                freqUP = np.linspace(self.freq[0],self.freq[-1],self.nChan0*upChan)
                dispersionDelay = ((self.DM/K)*(1/freqUP**2 - 1/fref**2)).to(u.s)
        except:
            dispersionDelay = np.zeros(self.freq.shape[0]*upChan)*u.s
        folded = np.zeros((self.nChan0*upChan,nBins))
        wts = np.zeros((self.nChan0*upChan,nBins))
        spec = np.zeros((self.blockFrame//upChan,self.nChan*upChan))
        for order in range(self.nOrder):
            spec[:,:],times = self.rings[order]._read_block(block)
            pulseFreq = self.polyco.eval_spin_freq(times.mjd).mean()*u.Hz
            phaseRef = self.polyco.eval_phase(times.mjd)

            for i in range(spec.shape[1]):
                column = np.abs(spec[:,i])**2
                # phases = self.polyco.eval_phase((times-dispersionDelay[order*self.nChan+i]).mjd)
                pShift = (dispersionDelay[order*self.nChan+i]*pulseFreq).to(u.dimensionless_unscaled)
                phases = np.mod(phaseRef-pShift+.5,1)-.5
                folded[order*self.nChan+i,:],__ = np.histogram(phases[np.isfinite(column)],weights=column[np.isfinite(column)],bins=np.linspace(-.5,.5,nBins+1))
                wts[order*self.nChan+i,:],__ = np.histogram(phases[np.isfinite(column)],bins=np.linspace(-.5,.5,nBins+1))
        return(folded,wts)

    def check_data_window(self,fref,tCent,tWidth,dedisperse =True):
        ## Convert reference time to unix
        if isinstance(tCent,Time):
            tRef = tCent
        else:
            tRef = self.epoch0 + tCent
        tEnd = tRef + tWidth
        tStart = tRef - tWidth
        ## Calculate dispersion delay for each frequency
        try:
            dispersionDelay = ((self.DM/K)*(1/self.freq**2 - 1/fref**2)).to(u.s)
        except:
            dispersionDelay = np.zeros(self.freq.shape[0])*u.s
        ## Calculate number of frames in final spectrum
        nFrames = self.time_to_frames(2*tWidth)
    
        ## Find start and end at each frequency
        tStartDispersed = tStart+dispersionDelay
        tEndDispersed = tEnd+dispersionDelay
        exists = True
        for order in range(self.nOrder):
            exists*=self.rings[order].check_data_window(tStartDispersed[self.nChan*(order+1)-1],tEndDispersed[self.nChan*order])
        return(exists)

    def read_around_dedis(self,fref,tCent,tWidth,coherent = True):
        """Read and dedisperse all subands to cover a given time range at the reference frequency

        Parameters
        ----------
        fref: Astropy.Quantity
            Reference frequency for dedispersion
        tCent: Astropy.Time of Astropy.Quanity
            Center of desired time range at reference frequency.
        """
        
        ## Convert reference time to unix
        if isinstance(tCent,Time):
            tRef = tCent
        else:
            tRef = self.epoch0 + tCent
        tEnd = tRef + tWidth
        tStart = tRef - tWidth
        ## Calculate dispersion delay for each frequency
        try:
            dispersionDelay = ((self.DM/K)*(1/self.freq**2 - 1/fref**2)).to(u.s)
        except:
            dispersionDelay = np.zeros(self.freq.shape[0])*u.s
        ## Calculate number of frames in final spectrum
        nFrames = self.time_to_frames(2*tWidth)
    
        ## Find start and end at each frequency
        tStartDispersed = tStart+dispersionDelay
        tEndDispersed = tEnd+dispersionDelay
        spec = self.basic_spec(self.nChan0,nFrames)
        try:
            for order in range(self.nOrder):
                t0 = Time.now()
                specTemp,timeTemp = self.rings[order].read_blocks_containing(tStartDispersed[self.nChan*(order+1)-1],
                                                                             tEndDispersed[self.nChan*order],True,0
                                                                             )
                fSpec = np.fft.fftshift(np.fft.fft(specTemp,axis=0),axes=0)
                fd = thth.fft_axis(timeTemp,u.mHz)
                phase = 2*np.pi*(dispersionDelay[np.newaxis,self.nChan*order:self.nChan*(order+1)]+(tStart-timeTemp[0]))*fd[:,np.newaxis]
                fSpec *= np.exp(1j*phase.to_value(u.dimensionless_unscaled))
                spec[:,self.nChan*order:self.nChan*(order+1)]=np.fft.ifft(np.fft.ifftshift(fSpec,axes=0),axis=0)[:nFrames,:]
            return(spec,tStart)
        except Exception as e:
            print(f'In read_around_dedis({sys.exc_info()[2].tb_lineno}): {e}')
            print(f'Order:{order}')
            print(f'tStart: {tStartDispersed[self.nChan*(order+1)-1].isot}' )
            print(f'({tStartDispersed[self.nChan*(order+1)-1].unix})' )
            print(f'tEnd: {tEndDispersed[self.nChan*(order+1)-1].isot}' )
            print(f'({tEndDispersed[self.nChan*(order+1)-1].unix})')
            print(f'nTimes: {timeTemp.shape[0]}')
            print(f'Spec Shape: {specTemp.shape[0]}')
            time.sleep(3600)
            return(spec*np.nan,tStart)

    def _pulsar_prep(self,config):
        self.ToAs = []
        self.pulseWidth = self.sourcePars.getfloat('pulseWidth')*u.ms

        self.polyCoFreq = self.sourcePars.getfloat('polyCoFreq')*u.MHz
        self.parFile = self.sourcePars.get('parFile')
        self.ToAFile = config[self.name].get('ToAFile',
                                           os.path.join(self.dateDir,
                                                        f'ToAs_{self.obsDate}_{self.sourceName}_{self.name}.npy'))
        self.profileFile = config[self.name].get('profile',os.path.join(self.dateDir,
                                                        f'profile_{self.obsDate}_{self.sourceName}_{self.name}.npy'))
        self.polycoFile = os.path.join(self.dateDir,
                                       f'polyco_{self.obsDate}_{self.sourceName}_{self.name}.dat')
        if os.path.exists(self.polycoFile):
            self.polyco = Polycos.read(self.polycoFile)
        else:
            fushanObs = TopoObs(self.station, location = globals()[f'{self.station}Center'],origin='/home/dbaker/BursttBasics',overwrite=True)
            if self.verbose:
                print('Prep PolyCos')
            model = get_model(self.parFile)
            self.polyco = Polycos.generate_polycos(model,
                                              np.floor(Time(min([ring.blockStarts.min().value for ring in self.rings]),format='unix').mjd),
                                              np.ceil(Time(max([ring.blockStarts.max().value for ring in self.rings]),format='unix').mjd + self.frameSec.to_value(u.day)*self.blockFrame)+1,
                                              self.station,
                                              144, 12,
                                              self.polyCoFreq.value,progress=True)
            self.polyco.write_polyco_file(self.polycoFile)
        self.ToA_check() 

    def ToAs_from_pref(self,phaseRef):
        t = Time(min([ring.blockStarts.min().value for ring in self.rings]),format='unix').mjd+self.pulseWidth.to_value(u.day)
        phase = self.polyco.eval_phase(t)[0].value
        p = (1./(self.polyco.eval_spin_freq(t)[0]*u.Hz)).to_value(u.day)
        if phase>phaseRef:
            t = t + p*(1-phase)
        while np.abs(phase - phaseRef)>1e-7:
            p = (1./(self.polyco.eval_spin_freq(t)[0]*u.Hz)).to_value(u.day)
            t += p*(phaseRef-phase)
            phase = self.polyco.eval_phase(t)[0].value
        self.ToAs = []
        while t + self.pulseWidth.to_value(u.day) < Time(max([ring.blockStarts.max().value for ring in self.rings]),format='unix').mjd + self.frameSec.to_value(u.day)*self.blockFrame:
            self.ToAs.append(t)
            p = p = (1./(self.polyco.eval_spin_freq(t)[0]*u.Hz)).to_value(u.day)
            t += p
            phase = self.polyco.eval_phase(t)[0].value
            while np.abs(phase - phaseRef)>1e-7:
                p = p = (1./(self.polyco.eval_spin_freq(t)[0]*u.Hz)).to_value(u.day)
                t += p*(phaseRef-phase)
                phase = self.polyco.eval_phase(t)[0].value
        self.ToAs = Time(np.array(self.ToAs),format='mjd').unix
        np.save(self.ToAFile,self.ToAs)

class Intensity(Stream):
    """
    Represent and manipulate a series of files representing
    a single BURSTT Baseband recording from a single station.

    Parameters
    ----------
    config: ConfigParser
        Configuration data for observation
    station: str
        Station name for this observation. Usually one of:
        ('FUS','LTN')
    notebook: bool, optional
        Running in notebook and progress bars are desired. Defaults to False
    progressOrder: int, optional
        Base recurision level for progress bars. Defaults to -1 for no bars
    """

    def __init__(self,config,station,notebook=False,progressOrder=-1):
        super().__init__(config,station,notebook,progressOrder)
        if self.sourceType == 'pulsar':
            self._pulsar_prep(config)

    def _get_fbystamp(self):
        if  os.path.exists(self.fbystampFile):
            with open(self.fbystampFile, 'rb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                self.fbystamp = pkl.load(f)
        else:
            self.fbystamp = file_order(self.files,blockFrame=self.blockFrame,headSize = self.headSize,nBlock=self.nf*self.nBlock)[0]
            with open(self.fbystampFile, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pkl.dump(self.fbystamp, f, pkl.HIGHEST_PROTOCOL)

    def _get_block_byte(self):
        self.blockByte = int(self.blockFrame/self.rebin) * self.nChan * self.nBeams * 2

    def _read_config(self,config):
        self.analysisPars =  config['Analysis Properties']
        self.observationPars = config['Observation Properties']
        self.sourcePars = config['Source Properties']
        self.taskPars = config['Tasks']

        self.verbose = self.analysisPars.getboolean('verbose',False)

        self.sourceType = self.sourcePars.get('type')
        self.sourceName = self.sourcePars.get('name')

        self.obsDate = self.observationPars.get('date')

        self.rootDir = self.analysisPars.get('outDir','./')
        self.dateDir = os.path.join(self.rootDir,f'{self.obsDate}_{self.sourceName}')
        self.plotDir = os.path.join(self.dateDir,'plots')
        self.cutDir = os.path.join(self.dateDir,'Data','cuts')
        self.intensityDir = os.path.join(self.dateDir,'Data','intensities')
        self.correlationDir = os.path.join(self.dateDir,'Data','correlations')

        self.blockStartFile = config[self.name].get('fbystampFile',
                                           os.path.join(self.dateDir,
                                                        f'blockStart_{self.obsDate}_{self.sourceName}_{self.name}.npy'))
        self.fbystampFile = config[self.name].get('fbystampFile',
                                           os.path.join(self.dateDir,
                                                        f'fbystamp_{self.obsDate}_{self.sourceName}_{self.name}.dat'))

        self.station = config.get(self.name,"station")
        self.nBeams = config.getint(self.name, 'nBeams')
        self.headSize = config[self.name].getint('headSize',64)
        ## Reference Time
        self.epoch0 = Time(config.get('Observation Properties', 'date') + ' 16:00:00')

        ## Location of data files
        self.dataDir = config.get(self.name, 'dataDir')

        ## Files per Ring Per Epoch:
        self.nf = config.getint(self.name, 'ringFiles')

        ## Length of frame (us)
        if 'frameSec'.lower() in config.options(self.name):
            self.frameSec = config.getfloat(self.name, 'frameSec')*u.us
        else:
            self.frameSec = config.getfloat('Observation Properties', 'frameSec')*u.us

        ## Number of channels in full spectrum
        if 'nChan0'.lower() in config.options(self.name):
            self.nChan0 = config.getint(self.name, 'nChan0')
        else:
            self.nChan0 = config.getint('Observation Properties', 'nChan0')

        ## Number of blocks per file
        if 'nBlock'.lower() in config.options(self.name):
            self.nBlock = config.getint(self.name, 'nBlock')
        else:
            self.nBlock = config.getint('Observation Properties', 'nBlock')

        ## Number of frames per block
        if 'blockFrame'.lower() in config.options(self.name):
            self.blockFrame = config.getint(self.name, 'blockFrame')
        else:
            self.blockFrame = config.getint('Observation Properties', 'blockFrame')

        ## Frequency Information
        self.f0 = config.getfloat(self.name, 'f0')*u.MHz
        if 'bandWidth'.lower() in config.options(self.name):
            self.bandWidth = config.getfloat(self.name, 'bandWidth')*u.MHz
        else:
            self.bandWidth = config.getfloat('Observation Properties', 'bandWidth')*u.MHz
        self.freq = np.linspace(self.f0,self.f0+self.bandWidth,self.nChan0,endpoint=False)
        self.freq -= (self.freq[1]-self.freq[0])*config.getfloat(self.name, 'fOffset')

        ## DM (Not always required)
        self.DM = config['Source Properties'].getfloat('DM',0) * u.pc/u.cm**3

        self.nOrder = config.getint(self.name, 'nOrder')

        ## Channels per subband
        self.nChan = int(self.nChan0//self.nOrder)

        self.rebin = config.getint(self.name, 'rebin')

        self._get_block_byte()

    def _get_epochs(self,notebook, progressOrder = -1):
        """ Determine start time for all blocks in dataset.

        Parameters
        ----------
        progressOrder: int, optional
            Base recurision level for progress bars. Defaults to -1 for no bars
        """
        if os.path.exists(self.blockStartFile):
            self.blockStart = np.load(self.blockStartFile)*u.s
        else:
            ## Calculate start times for each block
            self.blockStart = np.zeros((len(self.fbystamp),self.nBlock,self.nf))*np.nan*u.s
            iter0 = range(self.blockStart.shape[0])
            if self.notebook and progressOrder>-1:
                iter0 = tqdm(iter0,position=progressOrder,leave=(progressOrder==0))
            for e in iter0:
                iter1 = range(len(self.fbystamp[e]))
                if self.notebook and progressOrder>-1:
                    iter1 = tqdm(iter1,position=progressOrder+1,leave=False)
                for f in iter1:
                    if len(self.fbystamp[e]) < self.nf:
                        print(f'Folder number {e} has only {len(self.fbystamp[e])}/{self.nf} files')
                    fh = open(self.fbystamp[e][f],'rb')
                    fID = int(self.fbystamp[e][f].split('_')[-1][1:])
                    iter2 = range(self.nBlock)
                    if self.notebook and progressOrder>-1:
                        iter2 = tqdm(iter2,position=progressOrder+2,leave=False)
                    for b in iter2:
                        p = (self.blockByte + 64)*b
                        fh.seek(p)
                        hd = fh.read(64)
                        tmp = decHeader2(hd,verbose=False)
                        try:
                            ep = tmp[2]*u.s + 2*u.s + (tmp[0]//2)*self.frameSec
                            self.blockStart[e,b,fID]=ep
                        except:
                            continue
                    fh.close()
            np.save(self.blockStartFile,self.blockStart.value)
        self.nEpochs = int(self.blockStart.shape[0]//self.nOrder)

    def load_chain(self,order,segment,verbose=False):
        files = self.fbystamp[self.nEpochs*order+segment[0]]
        bID = np.linspace(0,segment[2]-1,segment[2],dtype=int)+segment[1]
        fID = np.mod(bID,self.nf)
        bID2 = (bID//self.nf).astype(int)
        spec  = np.zeros((segment[2], int(self.blockFrame/self.rebin), self.nChan,self.nBeams), dtype=np.float32)
        pcnt  = np.zeros(segment[2], dtype=np.int64)
        epoch = np.zeros(segment[2])
        for i in range(segment[2]):
            with open(files[fID[i]],'rb') as fh:
                fh.seek(bID2[i]*(self.headSize+self.blockByte)+self.headSize)
                buf = fh.read(self.blockByte)
                spec[i] = np.reshape(np.frombuffer(buf,dtype=np.float16),
                                    (int(self.blockFrame/self.rebin),
                                    self.nChan,
                                    self.nBeams)).astype(np.float32)
            pcnt[i]  = 0
            epoch[i] = self.blockStart[self.nEpochs*order+segment[0],bID2[i],fID[i]].value
        return(spec, pcnt, epoch)

    def fold_block(self,fref,epoch,startBlock,nBins = 128):
        try:
            dispersionDelay = ((self.DM/K)*(1/self.freq**2 - 1/fref**2)).to(u.s)
        except:
            dispersionDelay = np.zeros(self.freq.shape[0])*u.s
        folded = np.zeros((self.nBeams,self.nChan0,nBins))
        wts = np.zeros((self.nBeams,self.nChan0,nBins))
        spec = np.zeros((int(self.blockFrame/self.rebin),self.nChan0,self.nBeams))
        for order in range(self.nOrder):
            spec[:,order*self.nChan:(order+1)*self.nChan] = self.read_order(epoch,startBlock,1,order)
        t = np.linspace(0,int(self.blockFrame/self.rebin)-1,int(self.blockFrame/self.rebin))*self.frameSec*self.rebin
        t += self.blockStart[epoch,int(startBlock//self.nf),np.mod(startBlock,self.nf)]
        times = Time(t.to_value(u.s),format='unix')
        phaseRef = self.polyco.eval_phase(times.mjd)
        pulseFreq = self.polyco.eval_spin_freq(times.mjd).mean()*u.Hz
        for i in range(self.nChan0):
            band = spec[:,i,:]
            pShift = (dispersionDelay[i]*pulseFreq).to(u.dimensionless_unscaled)
            phases =  np.mod(phaseRef-pShift+.5,1)-.5
            for j in range(self.nBeams):
                beam = band[:,j]
                folded[j,i,:],__ = np.histogram(phases[np.isfinite(beam)],weights=beam[np.isfinite(beam)],bins=np.linspace(-.5,.5,nBins+1))
                wts[j,i,:],__ = np.histogram(phases[np.isfinite(beam)],bins=np.linspace(-.5,.5,nBins+1))
        return(folded,wts)

    def read_order(self,epoch,startBlock,length,order,verbose=False):
        spec = super().read_order(epoch,startBlock,length,order,verbose)
        spec = np.reshape(spec,(-1,self.nChan,self.nBeams))
        return(spec)
    
    def basic_spec(self,nChans,nFrames):
        spec = np.zeros((nFrames,nChans,self.nBeams))
        return(spec)
    
    def time_to_frames(self,time):
        if isinstance(time,np.ndarray):
            frames = (time/(self.frameSec*self.rebin)).to_value(u.dimensionless_unscaled).astype(int)
        else:
            frames = int((time/(self.frameSec*self.rebin)).to_value(u.dimensionless_unscaled))
        return(frames)
    
    def ToAs_from_prof(self):
        v = self.profile[0].var(1)
        v2 = sigma_clip(v, sigma=2.0)
        self.profileFlat = np.mean(self.profile[0,~v2.mask],0)
        self.ToAs_from_pref(self.profilePhases[np.argmax(self.profileFlat)]) 


class Ring:
    """
    Holds information and functionality for a single ring of baseband data
    """


    def __init__(self,machine: str,
                 ringID: int,
                 nBlock : int,
                 fileList: list,
                 blockByte: int,
                 frameSec,
                 nChan,
                 blockFrame,
                 headSize = 64,
                 calcStarts = True
                 ) -> None:
        self.machine = machine
        self.ringID = ringID
        self.fileList = fileList
        self.nBlock = nBlock
        self.blockByte = blockByte
        self.frameSec = frameSec
        self.blockFrame = blockFrame
        self.nChan = nChan
        self.headSize = headSize
        if calcStarts:
            self._block_starts()

    def _block_starts(self):
        nFiles = len(self.fileList)
        self.blockStarts = np.zeros(nFiles*self.nBlock)*np.inf*u.s
        self.blockFileID = np.linspace(0,nFiles-1,nFiles).repeat(self.nBlock).astype(int)
        self.blockID = np.tile(np.linspace(0,self.nBlock-1,self.nBlock),nFiles).astype(int)
        for fileID,file in enumerate(self.fileList):
            with open(file,'rb') as fh:
                for b in range(self.nBlock):
                    p = (self.blockByte + 64)*b
                    try:
                        fh.seek(p)
                        hd = fh.read(64)
                        tmp = decHeader2(hd,verbose=False)
                        ep = tmp[2]*u.s + 2*u.s + (tmp[0]/2)*self.frameSec
                        self.blockStarts[fileID*self.nBlock+b]=ep
                    except:
                        pass
        order = np.argsort(self.blockStarts)
        self.blockStarts = self.blockStarts[order]
        self.blockFileID = self.blockFileID[order]
        self.blockID = self.blockID[order]
        self.totalBlocks = self.blockStarts[np.isfinite(self.blockStarts)].shape[0]
        self.blockStarts = self.blockStarts[:self.totalBlocks]
        self.blockFileID = self.blockFileID[:self.totalBlocks]
        self.blockID = self.blockID[:self.totalBlocks]
        self.earliest = np.nanmin(self.blockStarts)
        self.latest = np.nanmax(self.blockStarts)+np.nanmedian(np.diff(self.blockStarts))
        
        
    def _read_block(self,blockID):
        readFile = self.fileList[self.blockFileID[blockID]]
        readBlock = self.blockID[blockID]
        readStart = self.blockStarts[blockID]
        if np.isfinite(readStart):
            spec,pcnt,epoch = loadBeam(readFile,
                        nStart=readBlock,
                        nBlock=1,
                        nFrame=self.blockFrame,
                        nChan=self.nChan,
                        hdlen=self.headSize)
            spec=spec[0]
            times = Time(readStart.to_value(u.s),format='unix') + np.linspace(0,self.blockFrame-1,self.blockFrame)*self.frameSec
            success = True
        else:
            spec = np.zeros((self.blockFrame,self.nChan),dtype=complex)*np.nan
            times = np.zeros(self.blockFrame)*np.nan*u.s
            success = False
        return(spec,times)
    
    def read_blocks_containing(self,startTime,endTime,gapFill=False,fill_value=np.nan):
        endUnix = (endTime+2*self.frameSec).unix*u.s
        startUnix = (startTime-2*self.frameSec).unix*u.s
        startBlock = np.argwhere(self.blockStarts<startUnix).max()
        endBlock = np.argwhere((self.blockStarts+self.frameSec*self.blockFrame)>endUnix).min()
        toRead = endBlock-startBlock + 1
        t0 = Time(self.blockStarts[startBlock],format='unix')
        tf = Time(self.blockStarts[startBlock+toRead-1]+self.frameSec*self.blockFrame,format='unix')
        while t0+self.frameSec*self.blockFrame < startTime-2*self.frameSec:
            startBlock+=1
            toRead -=1
            if toRead <=0 or startBlock>=self.totalBlocks:
                nf = int(np.round(((endTime-startTime)/self.frameSec).to_value(u.dimensionless_unscaled)))+5
                times = startTime+np.linspace(0,nf-1,nf)*self.frameSec
                spec = np.zeros((nf,self.nChan),dtype=complex)
                return(spec,times)
            t0 = Time(self.blockStarts[startBlock],format='unix')
        while self.blockStarts[startBlock+toRead-1] > endUnix:
            toRead -=1
            if toRead <=0 or startBlock>=self.totalBlocks:
                nf = int(np.round(((endTime-startTime)/self.frameSec).to_value(u.dimensionless_unscaled)))+5
                times = startTime+np.linspace(0,nf-1,nf)*self.frameSec
                spec = np.zeros((nf,self.nChan),dtype=complex)
                return(spec,times)
            tf = Time(self.blockStarts[startBlock+toRead-1]+self.frameSec*self.blockFrame,format='unix')
            
        nf = int(np.round(((tf-t0)/self.frameSec).to_value(u.dimensionless_unscaled)))+1

        spec = np.zeros((nf,self.nChan),dtype=complex)*fill_value
        times = t0 + np.linspace(0,nf-1,nf)*self.frameSec
        for i in range(toRead):
            specNew,timesNew = self._read_block(startBlock+i)
            tb = Time(self.blockStarts[startBlock+i],format='unix')
            id0 = int(np.round(((tb-t0)/self.frameSec).to_value(u.dimensionless_unscaled)))
            spec[id0:id0+specNew.shape[0]]=specNew
        mask = (times>startTime-2*self.frameSec)*(times<endTime+2*self.frameSec)
        return(spec[mask],times[mask])
    
    def check_data_window(self,startTime,endTime):
        endUnix = (endTime+2*self.frameSec).unix*u.s
        startUnix = (startTime-2*self.frameSec).unix*u.s
        preblocks = np.argwhere(self.blockStarts<startUnix)
        if preblocks.shape[0] == 0:
            return(False)
        startBlock = preblocks.max()
        postblocks = np.argwhere((self.blockStarts+self.frameSec*self.blockFrame)>endUnix)
        if postblocks.shape[0] == 0:
            return(False)
        endBlock = postblocks.min()
        toRead = endBlock-startBlock + 1
        t0 = Time(self.blockStarts[startBlock],format='unix')
        while t0+self.frameSec*self.blockFrame < startTime-2*self.frameSec:
            startBlock+=1
            toRead -=1
            if toRead <=0 or startBlock>=self.totalBlocks:
                return(False)
            t0 = Time(self.blockStarts[startBlock],format='unix')
        while self.blockStarts[startBlock+toRead-1] > endUnix:
            toRead -=1
            if toRead <=0 or startBlock>=self.totalBlocks:
                return(False)
        return(True)
        
            
    @classmethod
    def from_pickle(cls,file,**kwargs):
        with open(file,'rb') as f:
                old = pkl.load(f)
        needed = ['machine','ringID','nBlock','fileList','blockByte','frameSec','nChan','blockFrame']

        for attr in needed:
            if hasattr(old,attr):
                kwargs.update({attr : getattr(old,attr)})
            if not attr in kwargs:
                raise AttributeError(f'Missing {attr}')
        kwargs.update({'calcStarts': False})
        new = cls(**kwargs)
        new.blockStarts = old.blockStarts 
        new.blockFileID = old.blockFileID 
        new.blockID = old.blockID 
        new.earliest = old.earliest 
        new.latest = old.latest 
        new.totalBlocks = old.totalBlocks 
        return(new)

