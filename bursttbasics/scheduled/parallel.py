from multiprocessing import Process, Pipe
import numpy as np
from ..burstt_func import *
from ..burstt_const import *
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
from scipy.optimize import curve_fit
from ..trigger import dumpclass as dc
from ..trigger import correlator as bcorr
from ..trigger import calc as calc
from ..trigger.plotting import plot_corr
from matplotlib.backends.backend_pdf import PdfPages



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
        elif message[0] == 'loadCalc':
            self.load_calc__()
        else:
            print(f"listenFront__ : unknown message in process {self.poolID} : {message}", flush=True)
        return True

    ## Call in Main
    def stop(self):
        self.front_pipe.send(["stop"])
        self.join()

    def load_calc(self):
        self.front_pipe.send(['loadCalc'])

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
    
    def load_calc__(self):
        from ..trigger.calc import dump_delay
        self.back_pipe.send('calc loaded')

    def load__(self,x):
        """
        Load in a single pulse worth of data at all stations
        Parameters
        ----------
        Returns
        -------

        """
        fRef,ToA,pulseWidth,coherent = x
        self.specs = []
        self.Is = []
        self.var = []
        self.ep0s = []
        lastStart = Time(0,format='unix')
        for obs in self.observations:
            print(f'{self.poolID}: load {obs.name}\n',flush=True)
            spec,ep0s = obs.read_around_dedis(fRef,ToA,pulseWidth)
            spec+=.5+.5j
            # self.specs.append(np.ma.array(spec,mask=False))
            self.specs.append(dc.StationDump(obs.name))
            self.specs[-1].freq = obs.freq
            self.specs[-1].spec = spec.T
            self.specs[-1].ep0=ep0s
            self.specs[-1].DM=obs.DM
            self.specs[-1].time = np.linspace(0,spec.shape[0],spec.shape[0],endpoint=False)*obs.frameSec
            self.specs[-1].modeled = False
            self.specs[-1].normed = False
            self.specs[-1].clipped = False
            self.specs[-1].ep0Time = True
            self.specs[-1].freq_function = frequency_model
            self.specs[-1].profModel = gau1
            lastStart = max(lastStart,(ep0s+(obs.DM/K)*(1/fRef**2 - 1/obs.freq**2)).max())

        for i,spec in enumerate(self.specs):
            if coherent:
                print(f'{self.poolID}: align {self.observations[i].name}\n',flush=True)
                spec.align_start(lastStart+(spec.DM/K)*((1/spec.freq**2)-(1/fRef**2)),True)
            self.var.append(spec.spec.var(1))
            self.Is.append(np.ma.array(np.reshape(np.abs(spec.spec[:,:400*(spec.spec.shape[1]//400)])**2,(spec.spec.shape[0],-1,400)).mean(2),mask=False).T)
            self.ep0s.append(spec.ep0) 
        
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
        pT,pF,source,width,meanSub,nSearch,nSub,nRate,dDMRange,rateRange,doFull,plot,save,eventID = x
        # print(f'{self.poolID} : {nSearch}')
        specID0 = 0
        self.specs[0].poptT = pT
        self.specs[0].poptF = pF
        self.specs[0].modeled = True
        settings ={
                'width':width,
                'gauFit':False,
                'plot':plot,
                'progress':False,
                'nRate':nRate,
                'mean':False,
                'dDMFitRange':dDMRange,
                'searchRange':rateRange,
                'subSearch':np.linspace(-1,1,nSub),
                'meanSub':meanSub,
                'maxOff':nSearch
                }
        results = {}
        for spec in self.specs:
            spec.norm_spec()
            spec.clip_spec()
        for dID in range(len(self.specs)-specID0-1):
            specID1 = specID0+dID+1
            
            baseDelay,baseRate = calc.dump_delay([self.specs[specID0],self.specs[specID1]],source)
            if doFull:
                output = bcorr.find_rate_dm(self.specs[specID0],self.specs[specID1],baseDelay,
                                        baseRate,weight=True,line=True,**settings)
                dDM,dDMF,dDMError,rate,rateF,rateError,tau,tauF,tauErr,peakSNR,offset,startResidual,tauOff,tauOffF = output
                results.update({f'{self.specs[specID0].station}-{self.specs[specID1].station}':{
                                            'time' : [bcorr.get_peak_time(self.specs[specID0])[0]],
                                            'offsets' : [offset],
                                            'offsetResidual' : [startResidual],
                                            'tau' : [tau],
                                            'tauFit' : [tauF],
                                            'tauErr' : [tauErr],
                                            'tauSub' : [tauOff],
                                            'tauSubFit' : [tauOffF],
                                            'rate' : [rate],
                                            'rateFit' : [rateF],
                                            'rateErr' : [rateError],
                                            'dDM' : [dDM],
                                            'dDMFit' : [dDMF],
                                            'dDMErr' : [dDMError],
                                            'SNR' : [peakSNR],
                                            'mask' : [True],
                                            'refDM' : [self.specs[specID0].DM],
                                            'refFreq' : [dc.get_f_overlap(self.specs[specID0],self.specs[specID1])[0]]
                                        }})
            else:
                if save:
                    # print(f'{self.poolID} : {nSearch}')
                    print(f'{self.poolID}: Make corr\n',flush=True)
                    offsets,taus,corrs,startResidual,tRef = bcorr.correlate_dump(self.specs[specID0],self.specs[specID1],maxOff=nSearch,
                                                                        progress=False,weight=True,
                                                                        baseOff=baseDelay,width=width,rate=baseRate)
                    np.savez(os.path.join(self.observations[specID0].correlationDir,f'{self.observations[specID0].station}-{self.observations[specID1].station}_{eventID}_{self.poolID}.npz'),
                            offsets = offsets,
                            taus=taus,
                            corrs=np.abs(corrs)**2
                            )
                    if plot:
                        plot_corr(offsets,taus,corrs)
            
            if plot:
                with PdfPages(os.path.join(self.observations[0].plotDir,
                                                    f'{self.specs[specID0].station}-{self.specs[specID1].station}_{eventID}_{self.poolID}.pdf'
                                                    )) as pdf:
                    for num in plt.get_fignums():
                        pdf.savefig(plt.figure(num))
                    plt.close('all')
        self.back_pipe.send(['data correlated',results])


        # masks,pT,pF,width,nSearch,predictedDelays,dmSearch,fref,dmPhase,nPad = x
        # if dmSearch:
        #     dms = np.concatenate((-np.logspace(-4,-6,10),np.zeros(1),np.logspace(-6,-4,10)))*u.pc/u.cm**3
        # else:
        #     dms = np.zeros(1)*u.pc/u.cm**3
        # for i in range(len(masks)):
        #     self.specs[i][:,masks[i]] = np.nan+1j*np.nan
        #     # self.specs[i].mask = masks[i].reshape((-1,masks[i].shape[0]))
        # summaries = []
        # tPeaks = []
        # offsets = []
        # bb = 0
        # for specID0 in range(len(self.specs)-1):
        #     freq0 = self.observations[specID0].freq
        #     times = np.linspace(0,self.specs[specID0].shape[0]-1,self.specs[specID0].shape[0])*self.observations[0].frameSec
        #     tModel = gau1(times.to_value(u.us),*pT[:-1],0)
        #     peakPx = np.argmax(tModel)
        #     fModel = frequency_model(freq0.to_value(u.MHz),*pF)
        #     if dms.shape[0]>0:
        #         fds = np.fft.fftfreq(2*width+4,self.observations[0].frameSec).to_value(u.Hz)
        #         fSpec0 = np.fft.fft(self.specs[specID0][peakPx-width-2:peakPx+width+2],axis=1)
        #     for dmID,dm in enumerate(dms):
        #         if dm!=0:
        #             delays = ((dm/K)*(1/freq0**2 - 1/fref**2)).to_value(u.s)
        #             spec0=np.fft.ifft(fSpec0*np.exp(-1j*2*np.pi*delays[np.newaxis,:]*fds[:,np.newaxis]))[2:-2]
        #             if dmPhase:
        #                 spec0*=np.exp(1j*2*np.pi*delays*freq0.to_value(u.Hz))[np.newaxis,:]
        #         else:
        #             spec0 = np.copy(self.specs[specID0][peakPx-width:peakPx+width])
        #         for deltaID1 in range(len(self.specs)-specID0-1):
                    
        #             baseID = bb+deltaID1
        #             ts = Time.now()
        #             specID1 = specID0+deltaID1+1
        #             freq1 = self.observations[specID1].freq
        #             fOverlap0 = (freq0>=freq1.min())*(freq0<=freq1.max())
        #             fOverlap1 = (freq1>=freq0.min())*(freq1<=freq0.max())
        #             ## Model coordinates
        #             freq = freq0[fOverlap0]
        #             baseNF = freq.shape[0]
        #             prePad = 0
        #             postPad = 0
        #             if nPad>0:
        #                 if nPad<baseNF:
        #                     prePad = 0
        #                     postPad = (nPad-baseNF)
        #                 else:
        #                     postPad = max(0,min(nPad,1024)-baseNF)
        #                     if nPad > 1024:
        #                         prePad = nPad - 1024
        #             nf = baseNF + prePad + postPad
        #             cross2D = np.zeros((spec0.shape[0],nf),dtype=np.complex64)*np.nan
        #             ## Cutout and normalize spec0
        #             spec0_con = np.conjugate(spec0[:,fOverlap0])
        #             spec0_con /= np.ma.abs(spec0_con)
        #             # print(spec0_con.shape)
        #             summaries.append(np.zeros((dms.shape[0],nSearch,nf),dtype=np.complex64))
        #             model = fModel[fOverlap0][np.newaxis,:]*tModel[:,np.newaxis]
        #             spec0_con *= model[peakPx-width:peakPx+width]
        #             if dmID == 0:
        #                 offsets.append((np.linspace(0,nSearch-1,nSearch) - nSearch//2 + predictedDelays[baseID])*self.observations[0].frameSec)
        #             tPeaks.append([self.ep0s[specID0]+times[peakPx].to_value(u.s)])
        #             rawSize = min(spec0_con.shape[1],nf)
        #             for i in range(nSearch):
        #                 j = i - nSearch//2 +predictedDelays[baseID]
        #                 ## Cutout and normalize shifted spec1
        #                 spec1_sel = self.specs[specID1][peakPx-width+j:peakPx+width+j][:,fOverlap1]
        #                 if not spec1_sel.shape[0]==spec0_con.shape[0]:
        #                     continue
        #                 ## Correlate
        #                 cross2D[:, prePad : prePad + rawSize] = spec1_sel[:,:rawSize] * spec0_con[:,:rawSize]
        #                 cross2D[:, prePad : prePad + rawSize] /= np.ma.abs(spec1_sel[:,:rawSize])
        #                 # cross2D *= model[peakPx-width:peakPx+width]
        #                 # cross2D.fill_value = np.nan+np.nan*1j
        #                 cross2D -= np.nanmean(cross2D)
        #                 ## FFT
        #                 FTcross = np.fft.fftshift(np.fft.fft2(np.nan_to_num(cross2D)))
        #                 FTamp = np.abs(FTcross)
        #                 ## Normalize and take fd=0 (average over time) response
        #                 summaries[baseID][dmID,i] = FTcross[width]/np.median(FTamp)
        #             if self.verbose:
        #                 print(f'pid{self.poolID}: Corralation of {self.observations[specID0].station}-{self.observations[specID1].station} near {predictedDelays[baseID]} at deltaDM {dm} took {(Time.now()-ts).to_value(u.min):.3f} minutes',flush=True)
        #     bb+=len(self.specs)-1-specID0
        # self.back_pipe.send(['data correlated',summaries,tPeaks,offsets])

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
        elif message[0] == 'loadCalc':
                    self.load_calc__()
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

    def load_calc(self):
        self.front_pipe.send(['loadCalc'])

    def load_calc__(self):
        from ..trigger.calc import dump_delay
        self.back_pipe.send('calc loaded')
    
    def load__(self,x):
            """
            Load in a single pulse worth of data at all stations
            Parameters
            ----------
            Returns
            -------
    
            """
            chunkID,chunkOffset,chunkWidth,source,nSearch,coherent = x
            self.specs = []
            self.Is = []
            self.var = []
            self.ep0s = []
            lastStart = Time(0,format='unix')
            obs = self.observations[0]
            print(f'{self.poolID}: load {obs.name}\n',flush=True)
            spec,ep0s = obs.read_block(obs.blockStart[0,0,0]+chunkOffset+chunkID*chunkWidth,chunkWidth)
            spec+=.5+.5j
            self.specs.append(dc.StationDump(obs.name))
            self.specs[-1].freq = obs.freq
            self.specs[-1].spec = spec.T
            self.specs[-1].ep0=ep0s
            self.specs[-1].DM=obs.DM
            self.specs[-1].time = np.linspace(0,spec.shape[0],spec.shape[0],endpoint=False)*obs.frameSec
            self.specs[-1].modeled = False
            self.specs[-1].normed = False
            self.specs[-1].clipped = False
            self.specs[-1].ep0Time = True
            print(f'{self.poolID}: {obs.name} dtype is {self.specs[-1].spec.dtype}\n',flush=True)

            for obs in self.observations[1:]:
                print(f'{self.poolID}: load {obs.name}\n',flush=True)
                self.specs.append(dc.StationDump(obs.name))
                baseDelay,baseRate = calc.dump_delay([self.specs[0],self.specs[-1]],source)
                targetStart = self.specs[0].ep0[0]+baseDelay-(nSearch+5)*obs.frameSec
                targetStartPx = int(obs.packet_from_time(targetStart-obs.frameSec))
                
                spec,ep0s = obs.read_block(targetStartPx,chunkWidth+2*nSearch+10)
                spec+=.5+.5j
                # self.specs.append(np.ma.array(spec,mask=False))
                
                self.specs[-1].freq = obs.freq
                self.specs[-1].spec = spec.T
                self.specs[-1].ep0=ep0s
                self.specs[-1].DM=obs.DM
                self.specs[-1].time = np.linspace(0,spec.shape[0],spec.shape[0],endpoint=False)*obs.frameSec
                self.specs[-1].modeled = False
                self.specs[-1].normed = False
                self.specs[-1].clipped = False
                self.specs[-1].ep0Time = True
                if coherent:
                    print(f'{self.poolID}: align {obs.name}\n',flush=True)
                    self.specs[-1].align_start(self.observations[0].time_from_packet(int(self.observations[0].packet_from_time(self.specs[-1].ep0[0]))+1),True)
                # print(f'{self.poolID}: {self.specs[0].spec.shape[1]} {self.specs[1].spec.shape[1]} {chunkWidth}\n',flush=True)
                # print(f'{self.poolID}: {(self.specs[1].ep0[0]-self.specs[0].ep0[0]).to(u.us)} {baseDelay.to(u.us):.2f}\n',flush=True)
                print(f'{self.poolID}: {obs.name} dtype is {self.specs[-1].spec.dtype}\n',flush=True)
            self.back_pipe.send(['data loaded'])

    def correlate__(self,x):
        source,nSearch,doFull,plot,save,eventID,nRate,dDMRange,rateRange,nSub,meanSub = x
        specID0 = 0
        # self.specs[0].poptT = pT
        # self.specs[0].poptF = pF
        # self.specs[0].modeled = True
        settings ={
                'width':self.specs[0].spec.shape[1],
                'gauFit':False,
                'plot':plot,
                'progress':False,
                'nRate':nRate,
                'mean':True,
                'dDMFitRange':dDMRange,
                'searchRange':rateRange,
                'subSearch':np.linspace(-1,1,nSub),
                'meanSub':meanSub,
                'maxOff':nSearch,
                'idPeak' : self.specs[0].spec.shape[1]//2
                }
        results = {}
        for spec in self.specs:
            spec.norm_spec()
            spec.clip_spec()
        for dID in range(len(self.specs)-specID0-1):
            specID1 = specID0+dID+1
            
            baseDelay,baseRate = calc.dump_delay([self.specs[specID0],self.specs[specID1]],source)
            if doFull:
                output = bcorr.find_rate_dm(self.specs[specID0],self.specs[specID1],baseDelay,
                                        baseRate,weight=False,line=False,**settings)
                dDM,dDMF,dDMError,rate,rateF,rateError,tau,tauF,tauErr,peakSNR,offset,startResidual,tauOff,tauOffF = output
                results.update({f'{self.specs[specID0].station}-{self.specs[specID1].station}':{
                                            'time' : [bcorr.get_peak_time(self.specs[specID0],weight=False)[0]],
                                            'offsets' : [offset],
                                            'offsetResidual' : [startResidual],
                                            'tau' : [tau],
                                            'tauFit' : [tauF],
                                            'tauErr' : [tauErr],
                                            'tauSub' : [tauOff],
                                            'tauSubFit' : [tauOffF],
                                            'rate' : [rate],
                                            'rateFit' : [rateF],
                                            'rateErr' : [rateError],
                                            'dDM' : [dDM],
                                            'dDMFit' : [dDMF],
                                            'dDMErr' : [dDMError],
                                            'SNR' : [peakSNR],
                                            'mask' : [True],
                                            'refDM' : [self.specs[specID0].DM],
                                            'refFreq' : [dc.get_f_overlap(self.specs[specID0],self.specs[specID1])[0]]
                                        }})
                if save:
                    for i in range(len(self.specs)):
                        stationName = self.observations[i].name
                        np.savez(os.path.join(self.observations[i].cutDir,f'{stationName}_{eventID}_{self.poolID}.npz'),
                                 spec = self.specs[i].spec,
                                 time = self.specs[i].time,
                                 freq = self.specs[i].freq,
                                 ep0 = self.specs[i].ep0,
                                 delay = baseDelay,
                                 rate = baseRate,
                                 )
                    offsets,taus,corrs,startResidual,tRef = bcorr.correlate_dump(self.specs[specID0],self.specs[specID1],maxOff=nSearch,
                                                                    progress=False,
                                                                    baseOff=baseDelay,weight=False,mean=True,
                                                                    width=self.specs[0].spec.shape[1],
                                                                    idPeak=self.specs[0].spec.shape[1]//2,rate=baseRate)
                    np.savez(os.path.join(self.observations[specID0].correlationDir,f'{self.observations[specID0].station}-{self.observations[specID1].station}_{eventID}_{self.poolID}.npz'),
                            offsets = offsets,
                            taus=taus,
                            corrs=corrs
                                            )
            else:
                if save:
                    for i in range(len(self.specs)):
                        stationName = self.observations[i].name
                        np.savez(os.path.join(self.observations[i].cutDir,f'{stationName}_{eventID}_{self.poolID}.npz'),
                                    spec = self.specs[i].spec,
                                    time = self.specs[i].time,
                                    freq = self.specs[i].freq,
                                    ep0 = self.specs[i].ep0,
                                    delay = baseDelay,
                                    rate = baseRate,
                                    )
                    print(f'{self.poolID}: Make corr\n',flush=True)
                    offsets,taus,corrs,startResidual,tRef = bcorr.correlate_dump(self.specs[specID0],self.specs[specID1],maxOff=nSearch,
                                                                    progress=False,
                                                                    baseOff=baseDelay,weight=False,mean=True,
                                                                    width=self.specs[0].spec.shape[1],
                                                                    idPeak=self.specs[0].spec.shape[1]//2,rate=baseRate)
                    np.savez(os.path.join(self.observations[specID0].correlationDir,f'{self.observations[specID0].station}-{self.observations[specID1].station}_{eventID}_{self.poolID}.npz'),
                            offsets = offsets,
                            taus=taus,
                            corrs=corrs
                            )
                    if plot:
                        plot_corr(offsets,taus,corrs)
            
            if plot:
                with PdfPages(os.path.join(self.observations[0].plotDir,
                                                    f'{self.specs[specID0].station}-{self.specs[specID1].station}_{eventID}_{self.poolID}.pdf'
                                                    )) as pdf:
                    for num in plt.get_fignums():
                        pdf.savefig(plt.figure(num))
                    plt.close('all')
        print(f'{self.poolID} sending back\n',flush=True)
        self.back_pipe.send(['data correlated',results])

    def get__(self,param):
        self.back_pipe.send(getattr(self,param))
