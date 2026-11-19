"""
Stream classes used to read and manipulate single beam scheduled recording data from a single station
"""
from astropy.time import Time
import astropy.units as u
import numpy as np
import os
import sys


from ..packet_func import toSigned
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
    orders = []
    beams = []
    stamps = []
    
    for file in files:
        dr =''
        sp = file.split('/')
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
        orders.append(tmp[-1])
        for i,p in enumerate(parts):
            if p[:4]=='beam':
                beam = p[4:]
                continue
            elif p[0]=='f':
                f=p[1:]
                continue
        beams.append(beam)
        fs.append(f)
        stamps.append(stamp)
    orders = np.array(orders)
    dirs = np.array(dirs)
    fs = np.array(fs)
    beams = np.array(beams)
    stamps = np.array(stamps)
    stamps-=stamps.min()
    stamps=np.round(stamps)
    
    oU = np.sort(np.unique(orders))
    dU = np.sort(np.unique(dirs))
    fU = np.sort(np.unique(fs))
    # bU = np.sort(np.unique(beams))
    sU = np.sort(np.unique(stamps))
    fbystamp = []
    stampbystamp = []
    for o in oU:
        for s in sU:
            tmp = []
            tmp2 = []
            for f in fU:
                for i in range(len(files)):
                    if orders[i]==o and stamps[i]==s and fs[i]==f:
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

        ## Number of frames per blocks
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

    def time_to_frames(self,time):
        if isinstance(time,np.ndarray):
            frames = (time/(self.frameSec*self.rebin)).to_value(u.dimensionless_unscaled).astype(int)
        else:
            frames = int((time/(self.frameSec*self.rebin)).to_value(u.dimensionless_unscaled))
        return(frames)
        
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


    def basic_spec(self,nChans,nFrames):
        spec = np.zeros((nFrames,nChans),dtype=np.complex64)
        return(spec)
    

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

    def read_blocks(self,startBlock,nBLocks = 1, dropFill = True):
        return(self.basic_spec(self.nChan0,self.blockFrame))

    def read_blocks_power(self,startBlock,nBLocks = 1, dropFill = True):
        return(np.abs(self.read_block(startBlock,nBLocks))**2)

    
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
        testInts = np.arange(256)
        self.LUT_COMPLEX = (toSigned(testInts & 0x0f,4)+1j*toSigned(np.right_shift(testInts,4),4)).astype(np.complex64)
        self.LUT_POWER = (toSigned(testInts & 0x0f,4)**2+toSigned(np.right_shift(testInts,4),4)**2).astype(np.float32)

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
                            self.blockStart[e,b,fID]=((tmp[0]-tmp[4])//self.nOrder*self.nOrder)//2
                        except:
                            continue
                    fh.close()
            np.savez(self.blockStartFile,blockStart = self.blockStart, blockEpoch0 = self.blockEpoch0)
        self.nEpochs = int(self.blockStart.shape[0]//self.nOrder)
        self.earliest = self.time_from_packet(self.blockStart[self.blockStart>0].min())
        self.latest = self.time_from_packet(self.blockStart[self.blockStart>0].max()+self.blockFrame-1)

    def repair_block_start(self):
        ## Block startes per order in block order
        orderStarts = np.reshape(self.blockStart,(self.nOrder,-1))
        ## How many blocks off from the start of the run each block thinks the start of the run is
        blocksOff = np.rint((orderStarts - orderStarts[0,0])/self.blockFrame).astype(int) - np.linspace(0,orderStarts.shape[1],orderStarts.shape[1],endpoint=False).astype(int)
        changeOff = np.diff(blocksOff,axis=1)
        ## Find places where a single block differs from its neighbours and the difference is symmetric
        singleProblems = (changeOff[:,:-1]+changeOff[:,1:] == 0) & (np.abs(changeOff[:,:-1])>0)
        singleProblems &= (orderStarts[:, :-2] > 0) & (orderStarts[:, 1:-1] > 0) & (orderStarts[:, 2:] > 0)
        staleHeaders = [[] for i in range(self.nOrder)]
        for i in range(self.nOrder):
            ## Get list of wrong headers for order
            wrongIDs = np.where(singleProblems[i])[0]+1
            staleHeaders[i]+=list(wrongIDs)
            ## Loop over all ids of wrong headers
            for bID in wrongIDs:
                orderStarts[i,bID] = orderStarts[i,bID-1]+self.blockFrame
                blocksOff[i,bID] = blocksOff[i,bID-1]
        self.blockStart = np.reshape(orderStarts,self.blockStart.shape)
                


    def get_blocks_spec(self,startBlock,nBlocks = 1, dropFill = True):
        spec=None
        if dropFill:
            orderStarts = np.reshape(self.blockStart,(self.nOrder,-1))
            for order in range(self.nOrder):
                afterIDs = orderStarts[order,startBlock:min(startBlock+nBlocks,orderStarts.shape[1])]>=orderStarts[order,startBlock]
                beforeEndIDs = orderStarts[order,startBlock:min(startBlock+nBlocks,orderStarts.shape[1])]<=(orderStarts[order,startBlock]+(nBlocks+.5)*self.blockFrame)
                orderIDs = np.argwhere(afterIDs & beforeEndIDs)[0]
                fileNumbers = np.mod(orderIDs,self.nf).astype(int)
                blockNumbers = np.mod((orderIDs-fileNumbers)//self.nf,self.nBlock)
                epochNumbers = (orderIDs-blockNumbers*self.nf-fileNumbers)//(self.nf*self.nBlock)
            pass
            ## generate contiguous spectra of nBlocks*blockFrame length with 0s for missing data
        else:
            pass
            ## generate spectra of nBlocks*blockFrame length with gaps ignored
        return(spec)

    def read_blocks_power(self,startBlock,nBlocks = 1, dropFill = True):
        I=None
        if dropFill:
            pass
            ## generate contiguous spectra of nBlocks*blockFrame length with 0s for missing data
        else:
            pass
            ## generate spectra of nBlocks*blockFrame length with gaps ignored
        return(I)

    def read_file(self,fname,startBlock,nBlocks,table):
        bb = self.blockByte + self.headSize
        buf = np.fromfile(fname,dtype=np.uint8,count=nBlocks*bb,offset=startBlock * bb)
        if buf.size != nBlocks * self.blockByte:
            raise ValueError(f'{fname}: asked for {nBlocks} blocks from {startBlock}, file has {buf.size // self.blockByte}!')
        spec = table[buf.reshape(nBlocks,bb)[:,self.headSize:]].reshape((nBlocks,self.blockFrame,self.nChan))
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