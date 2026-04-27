from multiprocessing import Process, Pipe
import numpy as np
from ..burstt_func import *
from ..burstt_const import *
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
from scipy.optimize import curve_fit
from ..trigger.dumpclass import StationDump
from ..trigger.correlator import correlate_dump,find_rate_dm


class SinglePulseProcess(Process):
    def __init__(self,poolID,verbose=False):
        self.poolID = poolID
        self.verbose = verbose
        super().__init__()
        self.front_pipe, self.back_pipe = Pipe()
    
    # BACKEND

    def run(self):
        self.active = True
        while self.active:
            self.active = self.listenFront__()

    def listenFront__(self):
        message = self.back_pipe.recv()
        if message[0]== "stop":
            return False
        elif message[0] == "obs":
            self.set_obs__(message[1])
        elif message[0] == "load":
            self.load__(message[1])
        elif message[0] == "correlate":
            self.correlate__(message[1])
        elif message[0] == "save cut":
            self.save_cuts__(message[1])
        elif message[0] == "load SNR":
            self.load_SNR__(message[1])
        elif message[0] == 'get':
            self.get__(message[1])
        else:
            print(f"listenFront__ : unknown message in process {self.poolID} : {message}", flush=True)
        return True

    ## Call in Main
    def stop(self):
        self.front_pipe.send(["stop"])
        self.join()

    def give_obs(self,observations):
        self.front_pipe.send(["obs",observations])

    def load_data(self,x):
        self.front_pipe.send(['load',x])
    def load_data_SNR(self,x):
        self.front_pipe.send(['load SNR',x])

    def correlate(self,x):
        self.front_pipe.send(['correlate',x])

    def get_parameter(self,param):
        self.front_pipe.send(['get',param])

    def save_cuts(self,fname):
        self.front_pipe.send(['save cut',fname])

    ## Call Local
    def set_obs__(self,observations):
        self.observations = observations
        self.back_pipe.send('observations set')
    
    def load__(self,x):
        self.specs = []
        self.Is = []
        self.var = []
        self.ep0s = []
        for obs in self.observations:
            spec,ep0s = obs.read_around_dedis(*x)
            self.var.append(spec.var(0))
            self.Is.append(np.ma.array(np.reshape(np.abs(spec[:400*(spec.shape[0]//400)])**2,(-1,400,spec.shape[1])).mean(1),mask=False))
            self.ep0s.append(ep0s) 
            self.specs.append(np.ma.array(spec,mask=False))
            # print(spec.shape)
        self.back_pipe.send(['data loaded',self.Is,self.var])

    def load_SNR__(self,x):
        profFits = []
        for j in range(len(self.observations)):
            C0 = np.median(self.observations[j].profileFlat)
            A0 = np.max(self.observations[j].profileFlat)-C0
            x00 =  self.observations[j].profilePhases[np.argmax(self.observations[j].profileFlat)]
            s0 = .1
            p0 = np.array([A0,x00,s0,C0])
            profFits.append(curve_fit(gau1,
                                    self.observations[j].profilePhases,
                                    self.observations[j].profileFlat,
                                    p0 = p0)[0])
        SNs = np.zeros(len(self.observations))
        for i,obs in enumerate(self.observations):
            spec,tStart = obs.read_around_dedis(*x)
            times = tStart+np.linspace(0,spec.shape[0]-1,spec.shape[0])*obs.frameSec
            phases = obs.polyco.eval_phase(times.mjd)
            I = np.abs(spec)**2
            rb = 400
            Ir = np.reshape(I[:rb*(I.shape[0]//rb)],(-1,rb,I.shape[1])).mean(1)
            IrMed = np.median(Ir,0)
            I/=IrMed
            Ir/=IrMed
            var = Ir.var(0)
            var2 = sigma_clip(var,2)
            I[:,var2.mask]=np.nan

            I1D = np.nanmean(I,1)
            model = gau1(phases,*profFits[i])
            # plt.figure()
            # plt.plot(self.observations[j].profilePhases,self.observations[j].profileFlat)
            # plt.plot(self.observations[j].profilePhases,gau1(self.observations[j].profilePhases,*profFits[i]))
            # plt.plot(phases,model)
            # plt.savefig(f'{tStart.mjd}_{self.observations[i].name}.png')
            # plt.close()
            model -= profFits[i][-1]
            model/=np.nansum(model)
            SNs[i] = np.nansum((I1D-1)*model)
            

        self.back_pipe.send(['data loaded',SNs])

    def save_cuts__(self,fname):
        for i in range(len(self.specs)):
            stationName = self.observations[i].station
            np.savez(f'{fname}_{stationName}.npz',data = self.specs[i], startTime = self.tStarts[i])
        self.back_pipe.send(['cuts saved'])

    def correlate__(self,x):
        masks,pT,pF,width,nSearch,predictedDelays,dmSearch,fref,dmPhase,nPad = x
        if dmSearch:
            dms = np.concatenate((-np.logspace(-4,-6,10),np.zeros(1),np.logspace(-6,-4,10)))*u.pc/u.cm**3
        else:
            dms = np.zeros(1)*u.pc/u.cm**3
        for i in range(len(masks)):
            self.specs[i][:,masks[i]] = np.nan+1j*np.nan
            # self.specs[i].mask = masks[i].reshape((-1,masks[i].shape[0]))
        summaries = []
        tPeaks = []
        offsets = []
        bb = 0
        for specID0 in range(len(self.specs)-1):
            freq0 = self.observations[specID0].freq
            times = np.linspace(0,self.specs[specID0].shape[0]-1,self.specs[specID0].shape[0])*self.observations[0].frameSec
            tModel = gau1(times.to_value(u.us),*pT[:-1],0)
            peakPx = np.argmax(tModel)
            fModel = power(freq0.to_value(u.MHz),*pF)
            if dms.shape[0]>0:
                fds = np.fft.fftfreq(2*width+4,self.observations[0].frameSec).to_value(u.Hz)
                fSpec0 = np.fft.fft(self.specs[specID0][peakPx-width-2:peakPx+width+2],axis=1)
            for dmID,dm in enumerate(dms):
                if dm!=0:
                    delays = ((dm/K)*(1/freq0**2 - 1/fref**2)).to_value(u.s)
                    spec0=np.fft.ifft(fSpec0*np.exp(-1j*2*np.pi*delays[np.newaxis,:]*fds[:,np.newaxis]))[2:-2]
                    if dmPhase:
                        spec0*=np.exp(1j*2*np.pi*delays*freq0.to_value(u.Hz))[np.newaxis,:]
                else:
                    spec0 = np.copy(self.specs[specID0][peakPx-width:peakPx+width])
                for deltaID1 in range(len(self.specs)-specID0-1):
                    
                    baseID = bb+deltaID1
                    ts = Time.now()
                    specID1 = specID0+deltaID1+1
                    freq1 = self.observations[specID1].freq
                    fOverlap0 = (freq0>=freq1.min())*(freq0<=freq1.max())
                    fOverlap1 = (freq1>=freq0.min())*(freq1<=freq0.max())
                    ## Model coordinates
                    freq = freq0[fOverlap0]
                    baseNF = freq.shape[0]
                    prePad = 0
                    postPad = 0
                    if nPad>0:
                        if nPad<baseNF:
                            prePad = 0
                            postPad = (nPad-baseNF)
                        else:
                            postPad = max(0,min(nPad,1024)-baseNF)
                            if nPad > 1024:
                                prePad = nPad - 1024
                    nf = baseNF + prePad + postPad
                    cross2D = np.zeros((spec0.shape[0],nf),dtype=np.complex64)*np.nan
                    ## Cutout and normalize spec0
                    spec0_con = np.conjugate(spec0[:,fOverlap0])
                    spec0_con /= np.ma.abs(spec0_con)
                    # print(spec0_con.shape)
                    summaries.append(np.zeros((dms.shape[0],nSearch,nf),dtype=np.complex64))
                    model = fModel[fOverlap0][np.newaxis,:]*tModel[:,np.newaxis]
                    spec0_con *= model[peakPx-width:peakPx+width]
                    if dmID == 0:
                        offsets.append((np.linspace(0,nSearch-1,nSearch) - nSearch//2 + predictedDelays[baseID])*self.observations[0].frameSec)
                    tPeaks.append([self.tStarts[0].unix+times[peakPx].to_value(u.s)])
                    rawSize = min(spec0_con.shape[1],nf)
                    for i in range(nSearch):
                        j = i - nSearch//2 +predictedDelays[baseID]
                        ## Cutout and normalize shifted spec1
                        spec1_sel = self.specs[specID1][peakPx-width+j:peakPx+width+j][:,fOverlap1]
                        if not spec1_sel.shape[0]==spec0_con.shape[0]:
                            continue
                        ## Correlate
                        cross2D[:, prePad : prePad + rawSize] = spec1_sel[:,:rawSize] * spec0_con[:,:rawSize]
                        cross2D[:, prePad : prePad + rawSize] /= np.ma.abs(spec1_sel[:,:rawSize])
                        # cross2D *= model[peakPx-width:peakPx+width]
                        # cross2D.fill_value = np.nan+np.nan*1j
                        cross2D -= np.nanmean(cross2D)
                        ## FFT
                        FTcross = np.fft.fftshift(np.fft.fft2(np.nan_to_num(cross2D)))
                        FTamp = np.abs(FTcross)
                        ## Normalize and take fd=0 (average over time) response
                        summaries[baseID][dmID,i] = FTcross[width]/np.median(FTamp)
                    if self.verbose:
                        print(f'pid{self.poolID}: Corralation of {self.observations[specID0].station}-{self.observations[specID1].station} near {predictedDelays[baseID]} at deltaDM {dm} took {(Time.now()-ts).to_value(u.min):.3f} minutes',flush=True)
            bb+=len(self.specs)-1-specID0
        self.back_pipe.send(['data correlated',summaries,tPeaks,offsets])

    def get__(self,param):
        self.back_pipe.send(getattr(self,param))
            
class FoldingProcess(Process):
    def __init__(self,poolID,verbose=False):
        self.poolID = poolID
        self.verbose = verbose
        super().__init__()
        self.front_pipe, self.back_pipe = Pipe()
    
    # BACKEND

    def run(self):
        self.active = True
        while self.active:
            self.active = self.listenFront__()

    def listenFront__(self):
        message = self.back_pipe.recv()
        if message[0]== "stop":
            return False
        elif message[0] == "obs":
            self.set_obs__(message[1])
        elif message[0] == "fold":
            self.fold__(message[1])
        elif message[0] == 'get':
            self.get__(message[1])
        else:
            print(f"listenFront__ : unknown message in process {self.id} : {message}", flush=True)
        return True
    
        ## Call in Main
    
    def stop(self):
        self.front_pipe.send(["stop"])
        self.join()

    def give_obs(self,observations):
        self.front_pipe.send(["obs",observations])

    def fold(self,x):
        self.front_pipe.send(['fold',x])

    def get_parameter(self,param):
        self.front_pipe.send(['get',param])

    ## Call Local
    def set_obs__(self,observations):
        self.observations = observations
        self.back_pipe.send('observations set')
    
    def fold__(self,x):
        if len(x) == 4:
            startBlock,numBlocks,nBins,fref = x
            upChan = 1
        else:
            startBlock,numBlocks,nBins,fref,upChan = x
        nf = self.observations[0].nf
        first = True
        didFold = False
        folded = np.zeros(0)
        for blockID in range(numBlocks):
            if self.verbose:
                t = Time.now()
            block = startBlock+blockID
            eid = int(block//(nf*self.observations[0].nBlock))
            bid = block - eid*(nf*self.observations[0].nBlock)
            if np.isfinite(self.observations[0].blockStart[eid,bid//nf,np.mod(bid,nf)]):
                A,B = self.observations[0].fold_block(fref,eid,bid,nBins,upChan=upChan)
                if first:
                    folded = A
                    wts = B
                    first = False
                    didFold = True
                else:
                    folded += A
                    wts += B
            if self.verbose:
                dt = (Time.now()-t).to_value(u.min)
                print(f'pid{self.poolID}: {blockID+1}/{numBlocks} Complete in {dt:.2f}')
        if didFold:
            self.back_pipe.send(['data loaded',folded,wts])
        else:
            self.back_pipe.send(['no data'])
            

    def get__(self,param):
        self.back_pipe.send(getattr(self,param))

class SingleChunkProcess(Process):
    def __init__(self,poolID,verbose=False):
        self.poolID = poolID
        self.verbose = verbose
        super().__init__()
        self.front_pipe, self.back_pipe = Pipe()
    
    # BACKEND

    def run(self):
        self.active = True
        while self.active:
            self.active = self.listenFront__()

    def listenFront__(self):
        message = self.back_pipe.recv()
        if message[0]== "stop":
            return False
        elif message[0] == "obs":
            self.set_obs__(message[1])
        elif message[0] == "load":
            self.load__(message[1])
        elif message[0] == "correlate":
            self.correlate__(message[1])
        elif message[0] == 'get':
            self.get__(message[1])
        else:
            print(f"listenFront__ : unknown message in process {self.id} : {message}", flush=True)
        return True

    ## Call in Main
    def stop(self):
        self.front_pipe.send(["stop"])
        self.join()

    def give_obs(self,observations):
        self.front_pipe.send(["obs",observations])

    def load_data(self,x):
        self.front_pipe.send(['load',x])

    def correlate(self,x):
        self.front_pipe.send(['correlate',x])

    def get_parameter(self,param):
        self.front_pipe.send(['get',param])

    ## Call Local
    def set_obs__(self,observations):
        self.observations = observations
        self.back_pipe.send('observations set')
    
    def load__(self,x):
        self.specs = []
        self.Is = []
        self.var = []
        self.tStarts = []
        for obs in self.observations:
            spec,tStart = obs.read_around(*x)
            self.var.append(spec.var(0))
            self.Is.append(np.ma.array(np.reshape(np.abs(spec[:400*(spec.shape[0]//400)])**2,(-1,400,spec.shape[1])).mean(1),mask=False))
            self.tStarts.append(tStart)
            self.specs.append(np.ma.array(spec,mask=False))
        self.back_pipe.send(['data loaded',self.Is,self.var])

    def correlate__(self,x):
        masks,nSearch,doFFT,predictedDelays,norms = x
        for i in range(len(masks)):
            self.specs[i].mask = masks[i].reshape((-1,masks[i].shape[0]))
            self.specs[i]/=norms[i]
        summaries = []
        tPeaks = []
        offsets = []
        bb = 0
        for specID0 in range(len(self.specs)-1):
            for deltaID1 in range(len(self.specs)-specID0-1):
                
                ts = Time.now()
                specID1 = specID0+deltaID1+1
                baseOff = predictedDelays[bb+deltaID1]
                if self.verbose:
                    print(f'pid{self.poolID}: Cuts Offset by {self.tStarts[specID1]-self.tStarts[specID0]}',flush=True)
                freq0 = self.observations[specID0].freq
                freq1 = self.observations[specID1].freq
                fOverlap0 = (freq0>=freq1.min())*(freq0<=freq1.max())
                fOverlap1 = (freq1>=freq0.min())*(freq1<=freq0.max())
                ## Model coordinates
                freq = freq0[fOverlap0]
                times = np.linspace(0,self.specs[specID0].shape[0]-1,self.specs[specID0].shape[0])*self.observations[0].frameSec
                l = min(self.specs[specID0].shape[0],self.specs[specID1].shape[0])
                start = max(-baseOff+nSearch//2,0)
                stop = l + min(-baseOff-nSearch//2,0)
                
                # print(f'pid:{self.poolID}-{l},{start},{stop},{stop-start+1},{start-nSearch//2+baseOff},{stop+nSearch//2+baseOff}')
                time.sleep(10)
                spec0_con = np.conjugate(self.specs[specID0][start:stop][:,fOverlap0])
                # spec0_con /= np.ma.abs(spec0_con)
            
                summary = np.zeros((nSearch,freq.shape[0]),dtype=complex)
                offsets.append((np.linspace(0,nSearch-1,nSearch) - nSearch//2 + baseOff)*self.observations[0].frameSec)
                
                for i in range(nSearch):
                    if self.verbose:
                        tp = Time.now()
                    j = i - nSearch//2
                    ## Cutout and normalize shifted spec1
                    spec1_sel = self.specs[specID1][start+j+baseOff:stop+j+baseOff][:,fOverlap1]
                    ## Correlate
                    cross2D = spec1_sel * spec0_con
                    # cross2D /= np.ma.abs(spec1_sel)
                    # cross2D *= model[peakPx-width:peakPx+width]
                    cross2D -= cross2D.mean()
                    cross2D.fill_value = 0+0j
                    ## FFT
                    if doFFT:
                        FTcross = np.fft.fft2(cross2D.filled())
                        FTamp = np.abs(FTcross)
                        ## Normalize and take fd=0 (average over time) response
                        summary[i] = np.fft.fftshift(FTcross[0])/np.median(FTamp)
                    else:
                        FTcross = np.fft.fft(cross2D.filled(),axis=1).mean(0)
                        summary[i] = np.fft.fftshift(FTcross)/np.median(np.abs(FTcross))
                    if self.verbose:
                        print(f'pid{self.poolID}: Corralation of shift {i+1} took {(Time.now()-tp).to_value(u.min):.3f} minutes',flush=True)
                if self.verbose:
                    print(f'pid{self.poolID}: Corralation of {self.observations[specID0].station}-{self.observations[specID1].station} took {(Time.now()-ts).to_value(u.min):.3f} minutes',flush=True)
                summaries.append(summary)
            bb+=len(self.specs)-1-specID0
        self.back_pipe.send(['data correlated',summaries,offsets])

    def get__(self,param):
        self.back_pipe.send(getattr(self,param))
