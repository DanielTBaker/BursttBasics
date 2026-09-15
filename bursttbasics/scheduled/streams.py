"""
Stream classes used to read and manipulate single beam scheduled recording data from a single station
"""
from astropy.time import Time
import astropy.units as u
import numpy as np
import os
import sys


from ..packet_func import *
from ..burstt_func import *
from ..burstt_const import *
from tqdm.notebook import tqdm

from pint.models import get_model
from pint.polycos import Polycos
from  pint.observatory.topo_obs import TopoObs


import pickle as pkl
from glob import glob
from braceexpand import braceexpand



def braced_glob(path):
    l = []
    for x in braceexpand(path):
        l.extend(glob(x))
            
    return l

def file_order(files,blockFrame=51200,headSize = 64,nBlock=1600):
    dirs = []
    fs = []
    rings = []
    beams = []
    stamps = []
    machines = []
    
    for file in files:
        dr =''
        sp = file.split('/')
        for i in range(len(sp)-1):
            sp[i]+='/'
            if sp[i][:6]=='burstt':
                try:
                    machines.append(int(sp[i][6:-1]))
                    continue
                except:
                    pass
        dr = dr.join(sp[:-1])
        dirs.append(dr)
        parts = sp[-1].split('_')
        f=None
        ring=None
        beam=None
        fh = open(file,'rb')
        hd = fh.read(headSize)
        tmp = decHeader2(hd,verbose=False)
        fh.close()
        stamp = (tmp[0]//2)/(blockFrame*nBlock)
        for i,p in enumerate(parts):
            if p[:4]=='ring':
                ring=p[4:]
                continue
            elif p[:4]=='beam':
                beam = p[4:]
                continue
            elif p[0]=='f':
                f=p[1:]
                continue
        rings.append(ring)
        beams.append(beam)
        fs.append(f)
        stamps.append(stamp)
    machines = np.array(machines)
    dirs = np.array(dirs)
    rings = np.array(rings)
    fs = np.array(fs)
    beams = np.array(beams)
    stamps = np.array(stamps)
    stamps-=stamps.min()
    stamps=np.round(stamps)
    
    mU = np.sort(np.unique(machines))
    dU = np.sort(np.unique(dirs))
    rU = np.sort(np.unique(rings))
    fU = np.sort(np.unique(fs))
    # bU = np.sort(np.unique(beams))
    sU = np.sort(np.unique(stamps))
    fbystamp = []
    stampbystamp = []
    for m in mU:
        for r in rU:
            for s in sU:
                tmp = []
                tmp2 = []
                for f in fU:
                    for i in range(len(files)):
                        if machines[i]==m and rings[i]==r and stamps[i]==s and fs[i]==f:
                            tmp.append(files[i])
                            tmp2.append(s)
                            continue
                fbystamp.append(tmp)
                stampbystamp.append(tmp2)
    return(fbystamp,stampbystamp)

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
        ## List of files
        self.files = braced_glob(self.dataDir)
        self.files.sort()
        self._get_fbystamp()

        ## Generate start times for each block
        self._get_epochs(config,notebook,progressOrder)

    def _get_fbystamp(self):
        """
        Sort data files by epoch
        """
        self.fbystamp = []

    def _get_block_byte(self):
        """
        Determine Number of Bytes per block of data
        """
        self.blockByte = 0

    def _read_config(self,config):
        """
        Read conf
        """
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
        
    def time_from_packet(self,packetNumber):
        return(Time(self.blockEpoch0,format='unix')+packetNumber*self.frameSec)
    
    def packet_from_time(self,time):
        return(np.round(((time-Time(self.blockEpoch0,format='unix'))/self.frameSec).to_value(u.dimensionless_unscaled)))
    
    def read_range_order(self,order,ts,nSample,outputSpec = None,verbose=False,flat=False,useNAN=False):
        if flat and isinstance(ts,int):
            ts = np.ones(self.nChan)*ts
        if outputSpec is None:
            outputSpec = self.basic_spec(self.nChan,nSample)
            if useNAN:
                outputSpec*=np.nan
        chanOffset=0
        if outputSpec.shape[1]>self.nChan:
            chanOffset=self.nChan*order
        te = ts[chanOffset:chanOffset + self.nChan] + nSample
        mask = (te.max()<=self.blockStart[order*self.nEpochs:(order+1)*self.nEpochs])
        mask += (ts[chanOffset:chanOffset + self.nChan].min()>=self.blockStart[order*self.nEpochs:(order+1)*self.nEpochs]+self.blockFrame)
        use = np.argwhere(~mask)
        epochID = use[:,0]
        blockID = use[:,1]
        fileID = use[:,2]
        fileStarts = self.blockStart[order*self.nEpochs+epochID,blockID,fileID]
        startOrder = np.argsort(fileStarts)
        epochID = epochID[startOrder]
        blockID = blockID[startOrder]
        fileID = fileID[startOrder]
        fileStarts = fileStarts[startOrder]
        startPx = fileStarts[:,np.newaxis]-ts[np.newaxis,chanOffset:chanOffset + self.nChan]
        x = np.maximum(startPx,0)
        y = np.maximum(-startPx,0)
        l = np.maximum(np.minimum(nSample-x,self.blockFrame-y),0)
        iterator = range(fileID.shape[0])
        # print(f'{self.station} {order} : {x.max(1)} {y.max(1)} {l.max(1)} {blockID}\n',flush=True)
        if verbose:
            iterator = tqdm(iterator)
        for blockNum in iterator:
            # print(self.fbystamp[order*self.nEpochs+epochID[blockNum]][fileID[blockNum]].split("/")[-1])
            if flat:
                outputSpec[x[blockNum,0]:x[blockNum,0]+l[blockNum,0],chanOffset:chanOffset+self.nChan] = loadBeam(self.fbystamp[order*self.nEpochs+epochID[blockNum]][fileID[blockNum]],
                                nStart=blockID[blockNum],nBlock=1,nFrame=self.blockFrame,
                                nChan=self.nChan,hdlen=self.headSize,verbose=verbose)[0][0][y[blockNum,0]:y[blockNum,0]+l[blockNum,0]]
            else:
                beam = loadBeam(self.fbystamp[order*self.nEpochs+epochID[blockNum]][fileID[blockNum]],
                                nStart=blockID[blockNum],nBlock=1,nFrame=self.blockFrame,
                                nChan=self.nChan,hdlen=self.headSize,verbose=verbose)[0][0]
                # print(np.abs(beam).max())
                for chan in range(self.nChan):
                    outputSpec[x[blockNum,chan]:x[blockNum,chan]+l[blockNum,chan],chanOffset+chan] = beam[y[blockNum,chan]:y[blockNum,chan]+l[blockNum,chan],chan]
        return(outputSpec)
        
    def _get_epochs(self,config,notebook=False, progressOrder = -1):
        """ Determine start time for all blocks in dataset.

        Parameters
        ----------
        progressOrder: int, optional
            Base recurision level for progress bars. Defaults to -1 for no bars
        """
        ## Calculate start times for each block
        self.blockStart = np.zeros((len(self.fbystamp),self.nBlock,self.nf),dtype=int)
        self.blockEpoch0 = 0
        self.nEpochs = 0


    def load_chain(self,order,segment,verbose=False):
        try:
            files = []
            bID = np.zeros(0)
            fID = 0
            bID2 = bID
            spec  = np.zeros((segment[2], self.blockFrame, self.nChan), dtype=np.complex64)
            pcnt  = np.zeros(segment[2], dtype=np.int64)
            epoch = np.zeros(segment[2])
        except Exception as e:
            print(f'In load_chain({sys.exc_info()[2].tb_lineno}): {e}',flush=True)
            print(segment)
            raise(e)
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
        try:
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
        except Exception as e:
            print(f'In read_order_dedis({sys.exc_info()[2].tb_lineno}): {e}',flush=True)
            raise(e)
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
        pxStartDispersed = self.packet_from_time(tStartDispersed).astype(int)
        spec = self.basic_spec(self.nChan0,nFrames)
        ep0 = self.time_from_packet(pxStartDispersed)
        for order in range(self.nOrder):
            self.read_range_order(order,pxStartDispersed,nFrames,outputSpec=spec)
        return(spec,ep0)

    def read_block(self,p0,nFrames):
            """Read and dedisperse all subands to cover a given time range at the reference frequency
    
            Parameters
            ----------
            fref: Astropy.Quantity
                Reference frequency for dedispersion
            tCent: Astropy.Time of Astropy.Quanity
                Center of desired time range at reference frequency.
            """
            
        
            ## Find start and end at each frequency
            spec = self.basic_spec(self.nChan0,nFrames)
            pxStart = np.ones(self.nChan0,dtype=int)*p0
            ep0 = self.time_from_packet(pxStart)
            for order in range(self.nOrder):
                self.read_range_order(order,pxStart,nFrames,outputSpec=spec)
            return(spec,ep0)

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
                                              np.floor(self.time_from_packet(np.nanmin(self.blockStart[self.blockStart>0])).mjd),
                                              np.ceil(self.time_from_packet(np.nanmax(self.blockStart)).mjd + self.frameSec.to_value(u.day)*self.blockFrame)+1,
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
                self.read_profile_file()
                self.ToAs_from_prof()

    def read_profile_file(self):
        self.profile = np.load(self.profileFile)
        self.profilePhases = np.linspace(-.5,.5,self.profile.shape[1]+1)
        self.profilePhases = (self.profilePhases[1:]+self.profilePhases[:-1])/2
        # self.ToAs_from_prof()

    def get_flat_profile(self):
        v = self.profile.var(1)
        v2 = sigma_clip(v, sigma=2.0)
        self.profileFlat = np.mean(self.profile[~v2.mask],0)

    def ToAs_from_prof(self):
        self.get_flat_profile()
        self.ToAs_from_pref(self.profilePhases[np.argmax(self.profileFlat)]) 

    def ToAs_from_tref(self,tref):
        phaseRef = self.polyco.eval_phase(tref.mjd)[0].value
        self.ToAs_from_pref(phaseRef)
        
    def ToAs_from_pref(self,phaseRef):
        t = self.time_from_packet(np.nanmin(self.blockStart[self.blockStart>0])).mjd+self.pulseWidth.to_value(u.day)
        phase = self.polyco.eval_phase(t)[0].value
        p = (1./(self.polyco.eval_spin_freq(t)[0]*u.Hz)).to_value(u.day)
        if phase>phaseRef:
            t = t + p*(1-phase)
        while np.abs(phase - phaseRef)>1e-7:
            p = (1./(self.polyco.eval_spin_freq(t)[0]*u.Hz)).to_value(u.day)
            t += p*(phaseRef-phase)
            phase = self.polyco.eval_phase(t)[0].value
        self.ToAs = []
        while t + self.pulseWidth.to_value(u.day) < self.time_from_packet(np.nanmax(self.blockStart)).mjd + self.frameSec.to_value(u.day)*self.blockFrame:
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
            start = self.blockStart[order*self.nEpochs+epoch,startBlock//self.nf,np.mod(startBlock,self.nf)]
            if start>0:
                I = self.read_range_order(order,
                                              np.ones(self.nChan0,dtype=int)*self.blockStart[order*self.nEpochs+epoch,startBlock//self.nf,np.mod(startBlock,self.nf)],
                                            self.blockFrame,flat=True)
                I=np.abs(I)**2
                times = self.time_from_packet(start)+np.linspace(0,self.blockFrame,self.blockFrame,endpoint=False)*self.frameSec
                phaseRef = self.polyco.eval_phase(times.mjd)
                pulseFreq = self.polyco.eval_spin_freq(times.mjd.mean())*u.Hz
                pShift = (dispersionDelay[self.nChan*order:self.nChan*(order+1)]*pulseFreq).to(u.dimensionless_unscaled)
                for i in range(self.nChan*upChan):
                    column = I[:,i]
                    phases =  np.mod(phaseRef-pShift[i]+.5,1)-.5
                    folded[order*self.nChan+i,:],__ = np.histogram(phases[np.isfinite(column)],weights=column[np.isfinite(column)],bins=np.linspace(-.5,.5,nBins+1))
                    wts[order*self.nChan+i,:],__ = np.histogram(phases[np.isfinite(column)],bins=np.linspace(-.5,.5,nBins+1))
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

    def _get_fbystamp(self):
        if  os.path.exists(self.fbystampFile):
            with open(self.fbystampFile, 'rb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                self.fbystamp = pkl.load(f)
        else:
            self.fbystamp = file_order(self.files,blockFrame=self.blockFrame,headSize = 64,nBlock=self.nf*self.nBlock)[0]
            with open(self.fbystampFile, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pkl.dump(self.fbystamp, f, pkl.HIGHEST_PROTOCOL)
        

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

        self.blockStartFile = config[self.name].get('fbystampFile',
                                           os.path.join(self.dateDir,
                                                        f'blockStart_{self.obsDate}_{self.sourceName}_{self.name}.npz'))
        self.fbystampFile = config[self.name].get('fbystampFile',
                                           os.path.join(self.dateDir,
                                                        f'fbystamp_{self.obsDate}_{self.sourceName}_{self.name}.dat'))
        self.station = config.get(self.name,"station")
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
        self.freq -= (self.freq[1]-self.freq[0])*config[self.name].getfloat('fOffset',0)

        ## DM (Not always required)
        self.DM = config['Source Properties'].getfloat('DM',0) * u.pc/u.cm**3

        self.nOrder = config.getint(self.name, 'nOrder')

        ## Channels per subband
        self.nChan = int(self.nChan0//self.nOrder)

        self._get_block_byte()
        self.rebin=1
        
    def _get_epochs(self,config,notebook = False, progressOrder = -1):
        """ Determine start time for all blocks in dataset.

        Parameters
        ----------
        progressOrder: int, optional
            Base recurision level for progress bars. Defaults to -1 for no bars
        """
        ## Calculate start times for each block
        if os.path.exists(self.blockStartFile):
            self.blockStart = np.load(self.blockStartFile)['blockStart'].astype(int)
            self.blockEpoch0 =  np.load(self.blockStartFile)['blockEpoch0'].astype(int)
        else:
            self.blockStart = np.zeros((len(self.fbystamp),self.nBlock,self.nf),dtype=int)
            self.blockEpoch0 = 0
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
                            self.blockEpoch0 = tmp[2]+2
                            self.blockStart[e,b,fID]=(tmp[0]//2)
                        except:
                            continue
                    fh.close()
            np.savez(self.blockStartFile,blockStart = self.blockStart, blockEpoch0 = self.blockEpoch0)
        self.nEpochs = int(self.blockStart.shape[0]//self.nOrder)
        self.earliest = self.time_from_packet(self.blockStart[self.blockStart>0].min())
        self.latest = self.time_from_packet(self.blockStart[self.blockStart>0].max()+self.blockFrame-1)

    def load_chain(self,order,segment,verbose=False):
        try:
            files = self.fbystamp[self.nEpochs*order+segment[0]]
            bID = np.linspace(0,segment[2]-1,segment[2],dtype=int)+segment[1]
            fID = np.mod(bID,self.nf)
            bID2 = (bID//self.nf).astype(int)
            spec  = np.zeros((segment[2], self.blockFrame, self.nChan), dtype=np.complex64)
            pcnt  = np.zeros(segment[2], dtype=np.int64)
            epoch = np.zeros(segment[2])

            for i in range(segment[2]):
                tmp = loadBeam(files[fID[i]],
                            nStart=bID2[i],
                            nBlock=1,
                            nFrame=self.blockFrame,
                            nChan=self.nChan,
                            hdlen=self.headSize,
                            verbose=verbose)
                spec[i]  = tmp[0][0]
                pcnt[i]  = tmp[1][0]
                epoch[i] = tmp[2][0]
        except Exception as e:
            print(f'In load_chain({sys.exc_info()[2].tb_lineno}): {e}\n',flush=True)
            print(f'S:{segment}, fl:{len(files)}, fidl:{fID.shape[0]}, i:{i}\n',flush=True)
        return(spec, pcnt, epoch)
    
    def read_order(self,epoch,startBlock,length,order,verbose=False):
        spec = super().read_order(epoch,startBlock,length,order,verbose)
        spec = np.reshape(spec,(-1,self.nChan))
        return(spec)


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
                                                        f'blockStart_{self.obsDate}_{self.sourceName}_{self.name}.npz'))
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
        self.freq -= (self.freq[1]-self.freq[0])*config[self.name].getfloat('fOffset',0)

        ## DM (Not always required)
        self.DM = config['Source Properties'].getfloat('DM',0) * u.pc/u.cm**3

        self.nOrder = config.getint(self.name, 'nOrder')

        ## Channels per subband
        self.nChan = int(self.nChan0//self.nOrder)

        self.rebin = config.getint(self.name, 'rebin')

        self._get_block_byte()

        
        
    def _get_epochs(self,config,notebook=False, progressOrder = -1):
        """ Determine start time for all blocks in dataset.

        Parameters
        ----------
        progressOrder: int, optional
            Base recurision level for progress bars. Defaults to -1 for no bars
        """
        if os.path.exists(self.blockStartFile):
            self.blockStart = np.load(self.blockStartFile)['blockStart']
            self.blockEpoch0 =  np.load(self.blockStartFile)['blockEpoch0']
        else:
            ## Calculate start times for each block
            self.blockStart = np.zeros((len(self.fbystamp),self.nBlock,self.nf))
            self.blockEpoch0 = 0.
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
                            self.blockEpoch0 = tmp[2] + 2*u.s
                            self.blockStart[e,b,fID]=(tmp[0]//2)
                        except:
                            continue
                    fh.close()
            np.savez(self.blockStartFile,self.blockStart.value)
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