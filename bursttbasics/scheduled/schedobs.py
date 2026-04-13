from ..packet_func import *
from ..burstt_const import *
from ..burstt_func import *
from .parallel import *
from .streams import *

import configparser
import sys
from astropy.time import Time
from multiprocessing.pool import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
import scintools.ththmod as thth
from scipy.optimize import curve_fit
import astropy.constants as const

from pint.models import get_model
from pint.polycos import Polycos
from  pint.observatory.topo_obs import TopoObs
from astropy.coordinates import EarthLocation,SkyCoord
import pint.logging
import os
from pycalc11 import Calc
from scipy.interpolate import interp1d

pint.logging.setup(level='ERROR')

class Observation:
    def __init__(self,configFile):
        self.config = configparser.ConfigParser()
        self.config.read(configFile)

        self.analysisPars =  self.config['Analysis Properties']
        self.observationPars = self.config['Observation Properties']
        self.sourcePars = self.config['Source Properties']
        self.taskPars = self.config['Tasks']

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

        os.makedirs(self.plotDir,exist_ok=True)
        os.makedirs(self.cutDir,exist_ok=True)
        os.makedirs(self.intensityDir,exist_ok=True)
        os.makedirs(self.correlationDir,exist_ok=True)

        self.pulseWidth = self.sourcePars.getfloat('pulseWidth')*u.ms

        self.streamNames = np.array(self.observationPars.get('streams').split(','))
        self.streams = []
        if self.verbose:
            print('Prep Streams')
        for s in self.streamNames:
            isBaseband = self.config[s].getboolean('baseband',True)
            if isBaseband:
                self.streams.append(Baseband(self.config,s))
            else:
                self.streams.append(Intensity(self.config,s))
        self.earliest = Time(np.array([np.nanmin(stream.blockStart).value for stream in self.streams]).max(),
                        format='unix').mjd
        self.latest = Time(np.array([np.nanmax(stream.blockStart).value+stream.blockFrame*stream.frameSec.to_value(u.s)  for stream in self.streams]).min(),
                      format='unix').mjd
        if 'endTime' in self.analysisPars:
            parLatest = Time(self.analysisPars['endTime']).mjd
            self.latest = min(self.latest,parLatest)
        if 'startTime' in self.analysisPars:
            parEarliest = Time(self.analysisPars['startTime']).mjd
            self.earliest = max(self.earliest,parEarliest)
        if self.sourceType == 'pulsar':
            self.polyCoFreq = self.sourcePars.getfloat('polyCoFreq')*u.MHz

    def do_tasks(self):
        if self.sourceType == 'pulsar':
            if self.taskPars.getboolean('fold',False):
                self.fold_pulsar(nBins = int(self.config['Folding'].getfloat('nbins',256)),
                                 nPool = int(self.config['Folding'].getfloat('nPool',80)))
            if self.taskPars.getboolean('improveToAs'):
                if self.taskPars.getboolean('fold',False):
                    foldStreams = self.config['Folding'].get('streams').split(',')
                    for stream in foldStreams:
                        try:
                            streamID = np.argwhere(self.streamNames==stream).min()
                        except Exception as e:
                            print(e)
                            continue
                        self.streams[streamID].profilePhases = np.linspace(-.5,.5,self.streams[streamID].profile.shape[1]+1)
                        self.streams[streamID].profilePhases = (self.streams[streamID].profilePhases[1:]+self.streams[streamID].profilePhases[:-1])/2
                        self.streams[streamID].ToAs_from_prof()
            if self.taskPars.getboolean('SNR',False):
                self.pulse_SN(self.config['SN'].getint('nPool'))
            if self.taskPars.getboolean('correlate',False) or self.taskPars.getboolean('makeIntensities',False):
                self.pulsar_correlate(corrWidth = int(self.config['Correlating'].getfloat('corrFrames',200)),
                                     nSearch = int(self.config['Correlating'].getfloat('nSearch',200)),
                                     doCorr = self.taskPars.getboolean('correlate',False),
                                     nPool = int(self.config['Correlating'].getfloat('nPool',80)),
                                     nCorr = int(self.config['Correlating'].getfloat('nCorr',40)),
                                     saveCuts = self.config['Correlating'].getboolean('savecuts',False),
                                     predictDelays = self.config['Correlating'].getboolean('predictDelay',False),
                                     useClock = self.config['Correlating'].getboolean('useClock',False)
                                    )
            if self.taskPars.getboolean('formDspec',False):
                self.make_dspec(intBlocks=self.config['Dspec'].getint('intBlocks'),
                                nBins=self.config['Dspec'].getint('nBins'),
                                nPool=self.config['Dspec'].getint('nPool'),
                                upChan=self.config['Dspec'].getint('upChan')
                                )
        if self.sourceType == 'quasar':
            if self.taskPars.getboolean('correlate',False):
                self.chunk_correlate(chunkWidth=self.config['Correlating'].getfloat('chunkWidth',1)*u.s,
                                     maxShift=int(self.config['Correlating'].getfloat('nSearch',200)),
                                     nPool = int(self.config['Correlating'].getfloat('nPool',80))
                                     )
        if self.taskPars.getboolean('corrPlots',False):
            self.examine_correlation()

    def fold_pulsar(self,nBins = 256,startOff = 0*u.s, length = np.inf*u.s, nPool=80):
        foldStreams = self.config['Folding'].get('streams').split(',')
        for stream in foldStreams:
            try:
                streamID = np.argwhere(self.streamNames==stream).min()
            except Exception as e:
                print(e)
                continue
            totalBlocks = self.streams[streamID].nf*self.streams[streamID].nBlock*self.streams[streamID].nEpochs
            lastStarts = self.streams[streamID].blockStart[-1,:,0]
            missing = lastStarts.shape[0]-lastStarts[np.isfinite(lastStarts)].shape[0]
            totalBlocks-=missing*self.streams[streamID].nf
            startBlock = int((startOff/(self.streams[streamID].blockFrame*self.streams[streamID].frameSec)).to(u.dimensionless_unscaled))
            useBlocks = totalBlocks-startBlock
            if np.isfinite(length):
                lengthBlock = int((length/(self.streams[streamID].blockFrame*self.streams[streamID].frameSec)).to(u.dimensionless_unscaled))+1
                useBlocks = min(useBlocks,lengthBlock)

            blocksPer = np.ones(nPool,dtype=int)*int(useBlocks//nPool)
            if useBlocks>np.sum(blocksPer):
                blocksPer[:useBlocks-np.sum(blocksPer)]+=1
            startBlocks = np.cumsum(blocksPer)-blocksPer[0]+startBlock
            if self.verbose:
                print(f'Starting Fold with {nPool} processes')
                print(f'Start at {startBlock} and use {useBlocks}/{totalBlocks}')
            procs = []
            for i in range(nPool):
                procs.append(FoldingProcess(i,self.verbose))
            for i,p in enumerate(procs):
                p.start()
                p.give_obs([self.streams[streamID]])
                p.front_pipe.recv()
                p.fold([startBlocks[i],blocksPer[i],nBins,self.polyCoFreq])
            wts = np.zeros(0)
            for i,p in enumerate(procs):
                res = p.front_pipe.recv()
                if res[0]=='data loaded':
                    if wts.shape[0]==0:
                        wts=res[2]
                        folded=res[1]
                    else:
                        wts+=res[2]
                        folded+=res[1]
                p.stop()
            
            profile = folded/wts
            profile/=np.expand_dims(np.nanmean(profile,-1),-1)
            profileFile = f'profile_{self.obsDate}_{self.sourceName}_{self.streamNames[streamID]}.npy'
            if self.verbose:
                print(f'Saving Profile to {profileFile}')

            v = profile.var(-1)
            v2 = sigma_clip(v, sigma=2.0)
            profileFlat = np.mean(profile[~v2.mask,:],0)
            np.save(os.path.join(self.dateDir,profileFile),profile)
            self.streams[streamID].ToA_check()
            
    def make_dspec(self,intBlocks=1,nBins = 256,startOff = 0*u.s, length = np.inf*u.s, nPool=80,upChan = 1):
        foldStreams = self.config['Dspec'].get('streams').split(',')
        for stream in foldStreams:
            try:
                streamID = np.argwhere(self.streamNames==stream).min()
            except Exception as e:
                print(e)
                continue
            totalBlocks = self.streams[streamID].nf*self.streams[streamID].nBlock*self.streams[streamID].nEpochs
            lastStarts = self.streams[streamID].blockStart[-1,:,0]
            missing = lastStarts.shape[0]-lastStarts[np.isfinite(lastStarts)].shape[0]
            totalBlocks-=missing*self.streams[streamID].nf
            startBlock = int((startOff/(self.streams[streamID].blockFrame*self.streams[streamID].frameSec)).to(u.dimensionless_unscaled))
            useBlocks = totalBlocks-startBlock
            if np.isfinite(length):
                lengthBlock = int((length/(self.streams[streamID].blockFrame*self.streams[streamID].frameSec)).to(u.dimensionless_unscaled))+1
                useBlocks = min(useBlocks,lengthBlock)
            useBlocks = intBlocks*(useBlocks//intBlocks)
            numInts = useBlocks//intBlocks
            nPool = min(nPool,numInts)
            startBlocks = np.linspace(0,numInts-1,numInts,dtype=int)*intBlocks
            if self.verbose:
                print(f'Starting Dynspec with {nPool} processes')
                print(f'Start at {startBlock} and use {useBlocks}/{totalBlocks}')
            procs = []
            for i in range(nPool):
                procs.append(FoldingProcess(i,False))
            for i,p in enumerate(procs):
                p.start()
                p.give_obs([self.streams[streamID]])
                p.front_pipe.recv()
                p.fold([startBlocks[i],intBlocks,nBins,self.polyCoFreq,upChan])
            specs = np.zeros((numInts,self.streams[streamID].nChan0*upChan,nBins))
            wts = np.zeros((numInts,self.streams[streamID].nChan0*upChan,nBins))
            for integration in range(numInts):
                p = procs[np.mod(integration,nPool)]
                res = p.front_pipe.recv()
                if res[0]=='data loaded':
                    wts[integration]=res[2]
                    specs[integration]=res[1]
                if self.verbose:
                    print(f'Completed Integration {integration+1}/{numInts}')
                if integration+nPool<numInts:
                    p.fold([startBlocks[integration+nPool],intBlocks,nBins,self.polyCoFreq,upChan])
                else:
                    p.stop()     
            specs/=wts
            specsFile = f'dspecfold_{self.obsDate}_{self.sourceName}_{self.streamNames[streamID]}_{upChan}.npy'
            np.save(os.path.join(self.dateDir,specsFile),specs)

    def improve_ToAs(self):
        pass

    def predict_delay(self,times,stationNames):
        if self.sourceType =='pulsar':
            model = get_model(self.streams[0].parFile)
            srce = model.as_ICRS(Time(self.streams[0].ToAs.mean(),format='unix')).coords_as_ICRS()
        else:
            srce = SkyCoord.from_name(self.sourceName)
        srce = SkyCoord(ra=[srce.ra],dec=[srce.dec])
        start_time = times[0]-1*u.min
        duration_min = np.ceil((times.max()-start_time).to_value(u.min))+1
        ci = Calc(
        station_names = stationNames,
        station_coords=[globals()[code+'Center'] for code in stationNames],
        source_coords=srce.icrs,
        start_time=start_time,
        duration_min=duration_min,
        d_interval=1,base_mode='baseline'
        )
        ci.run_driver()
        delayVLBI = ci.interpolate_delays(times)[:,:,:,0]
        allDelays = np.concatenate([(delayVLBI[:,i,i+1:].T) for i in range(delayVLBI.shape[1])],axis=0)
        return(allDelays)

    def pulsar_correlate(self,
                         corrWidth = 2000,
                         nSearch = 200,
                         doCorr = True,
                         nPool = 80,
                         nCorr = 40,
                         saveCuts = False,
                         predictDelays = False,
                         useClock = False,
                        ):
        correlationStreamNames = np.array(self.config['Correlating'].get('streams').split(','))
        correlationStreams = []
        for name in correlationStreamNames:
            for i in range(self.streamNames.shape[0]):
                if self.streamNames[i]==name:
                    correlationStreams.append(self.streams[i])
        correlationStations = [stream.station for stream in correlationStreams]
        ToAs = np.copy(correlationStreams[0].ToAs)
        for stream in correlationStreams:
            maxDispdelay = ((stream.DM/K)*(1/(stream.freq.min()**2)-1/(self.polyCoFreq**2))).to(u.s)
            ToAs = ToAs[(ToAs*u.s>stream.earliest+self.pulseWidth)*(ToAs*u.s<stream.latest-self.pulseWidth-maxDispdelay)]
        if predictDelays:
            predictedDelays = (self.predict_delay(Time(ToAs,format='unix'),correlationStations)/self.streams[0].frameSec).to_value(u.dimensionless_unscaled)
        else:
            predictedDelays = np.zeros((self.nBaselines,ToAs.shape[0]))
        if useClock:
            try:
                clock = (self.get_clock_correction(Time(ToAs,format='unix'))/self.streams[0].frameSec).to_value(u.dimensionless_unscaled)
                predictedDelays += clock
            except Exception as e:
                if self.verbose:
                    print('No PPP available')
                    print(e)
                    print(clock[0],predictedDelays[0])
        predictedDelays = np.round(predictedDelays).astype(int)
        groups = int(ToAs.shape[0]//nPool) +1
        procs = []
        if self.verbose:
            print('Building Proccesses')
        for i in range(nPool):
            procs.append(SinglePulseProcess(i,self.verbose))
        for p in procs:
            p.start()
            p.give_obs(correlationStreams)
            p.front_pipe.recv()
        skips = self.config['Correlating'].getint('skip',0)
        for groupID in range(groups):
            if groupID<skips:
                continue
            if self.verbose:
                print(f'Start group {groupID}')
            gs = Time.now()
            ## Determine group Indexes
            groupStart = groupID*nPool
            groupSize = min(nPool,ToAs.shape[0]-groupStart)
            corrSize = min(nCorr,ToAs.shape[0]-groupStart)
        
            ## Load Data
            for i in range(groupSize):
                procs[i].load_data([self.polyCoFreq,Time(ToAs[groupStart+i],format='unix'),self.pulseWidth,self.config['Correlating'].get('dedisperse')=='coherent'])
            if self.verbose:
                print(f'Reads Started at {(Time.now()-gs).to_value(u.min):.1f} minutes coherent is {self.config["Correlating"].get("dedisperse")=="coherent"}',flush=True)
            ## Read data from each process as it becomes available
            for i in range(groupSize):
                results = procs[i].front_pipe.recv()
                if i == 0: 
                    I = results[1]
                    var = results[2]
                else:
                    for obsID in range(len(correlationStreams)):
                        I[obsID] += results[1][obsID]
                        var[obsID] += results[2][obsID]
            if self.verbose:
                print(f'Reads Complete at {(Time.now()-gs).to_value(u.min):.1f} minutes',flush=True)
            if saveCuts:
                for i in range(groupSize):
                    procs[i].save_cuts(os.path.join(self.cutDir,
                                                    f'BURSTT_{self.sourceName}_{self.obsDate}_{groupStart+i}'))
                    procs[i].front_pipe.recv()
            ## Finalize Averaging
            I = [x/groupSize for x in I]
            Imeds = [x.mean(0) for x in I]
            for streamID in range(len(correlationStreams)):
                var[streamID]/=Imeds[streamID]
            var2 = [sigma_clip(x, sigma=2.0) for x in var]
            masks = [x.mask for x in var2]
            for streamID in range(len(correlationStreams)):
                I[streamID].mask = masks[streamID].reshape((-1,masks[streamID].shape[0]))
                I[streamID] = np.reshape(I[streamID]/Imeds[streamID],(I[streamID].shape[0],-1,32)).mean(2)
            ## Make Plots
            for streamID,stream in enumerate(correlationStreams):
                grid = plt.GridSpec(nrows=4,ncols=4)
                plotTime = np.linspace(0,I[streamID].shape[0]*400-1,
                                       I[streamID].shape[0]*400)*stream.frameSec.to(u.ms)
                plotTime = np.reshape(plotTime,(-1,400)).mean(1)
                plotFreq = np.reshape(stream.freq,(-1,32)).mean(1)
                ext = thth.ext_find(plotTime,plotFreq)
                plt.figure(figsize=(6,6))
                plt.subplot(grid[:3,:3])
                plt.imshow(I[streamID].T,
                           origin='lower',aspect='auto'
                           ,vmin=2-(I[streamID][:,:20].mean(1).max()),vmax=I[streamID][:,:20].mean(1).max(),
                           extent=ext)
                plt.ylabel('Freq (MHz)')
                plt.xticks([])
                plt.subplot(grid[-1,:3])
                plt.plot(plotTime,I[streamID][:,:20].mean(1))
                plt.xlabel('Time (ms)')
                plt.ylabel('I')
                plt.xlim(ext[:2])
                plt.subplot(grid[:3,-1])
                plt.plot(var[streamID],stream.freq)
                plt.plot(var2[streamID],stream.freq)
                plt.xlabel('Variance')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plotDir,f'BURSTT_{self.sourceName}_{self.obsDate}_{groupID}_{stream.name}_I_{self.config["Correlating"].get("dedisperse")}.png'))
                np.savez(os.path.join(self.intensityDir,f'BURSTT_{self.sourceName}_{self.obsDate}_{groupID}_{stream.name}_I_{self.config["Correlating"].get("dedisperse")}.npz'),I=I[streamID],f=stream.freq,t=plotTime)
                ## Fit model to first I
                if streamID == 0:
                    y=I[streamID][:,:20].mean(1)
                    C0 = 1.
                    A0 = y.max()-C0
                    x0 = plotTime[y==y.max()].mean().to_value(u.us)
                    s0 = plotTime.max().to_value(u.us)/10
                    try:
                        poptT,pcovT = curve_fit(gau1,plotTime.to_value(u.us),y,p0=[A0,x0,s0,C0])
                    except:
                        poptT = np.array([A0,plotTime.mean().to_value(u.us),s0,C0])
                    tProf = gau1(plotTime.to_value(u.us),*poptT[:-1],0)
                    y2=((I[streamID]-poptT[-1])*tProf[:,np.newaxis]).sum(0)/np.sum(tProf)
                    p0 = [y2[0]/np.power(stream.freq[0].to_value(u.MHz),-2),-2]
                    try:
                        poptF,pcovF = curve_fit(power,np.reshape(stream.freq,(I[streamID].shape[1],-1)).mean(1)[:25].to_value(u.MHz),y2[:25],p0=p0,maxfev=10000)
                    except:
                        poptF = p0
                    model = power(np.reshape(stream.freq,(I[streamID].shape[1],-1)).mean(1)[:25].to_value(u.MHz),*poptF)[:,np.newaxis]*tProf
                    plt.figure(figsize=(6,6))
                    plt.subplot(grid[:3,:3])
                    plt.imshow(model,
                               origin='lower',aspect='auto',
                               extent=ext)
                    plt.ylabel('Freq (MHz)')
                    plt.xticks([])
                    plt.subplot(grid[-1,:3])
                    plt.plot(plotTime,y-poptT[-1])
                    plt.plot(plotTime,tProf)
                    plt.xlabel('Time (ms)')
                    plt.ylabel('I')
                    plt.xlim(ext[:2])
                    plt.subplot(grid[:3,-1])
                    plt.plot(y2,np.reshape(stream.freq,(I[streamID].shape[1],-1)).mean(1).to_value(u.MHz))
                    plt.plot(power(np.reshape(stream.freq,(I[streamID].shape[1],-1)).mean(1).to_value(u.MHz),*poptF),np.reshape(stream.freq,(I[streamID].shape[1],-1)).mean(1).to_value(u.MHz))
                    plt.xlabel('I')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.plotDir,f'BURSTT_{self.sourceName}_{self.obsDate}_{groupID}_{stream.name}_model_{self.config["Correlating"].get("dedisperse")}.png'))
            if doCorr:
                for procID,proc in enumerate(procs[:corrSize]):
                    proc.correlate([masks,
                                    poptT,poptF,
                                    corrWidth,
                                    nSearch,predictedDelays[:,procID+groupStart],
                                    self.config["Correlating"].getboolean("dmSearch",False),
                                    self.polyCoFreq,
                                    self.config["Correlating"].getboolean("dmPhase",False),
                                    self.config["Correlating"].getint("size",2048)
                                   ])
                print(f'Correlations Started at {(Time.now()-gs).to_value(u.min):.1f} minutes',flush=True)
                for i,p in enumerate(procs[:groupSize]):
                    results = p.front_pipe.recv()
                    if i == 0:
                        corrs = results[1]
                        peakT = results[2]
                        offsets = results[3]
                    else:
                        for pairID in range(len(results[1])):
                            peakT[pairID] += results[2][pairID]
                            if i==1:
                                corrs[pairID] = np.concatenate([corrs[pairID][np.newaxis,:,:],results[1][pairID][np.newaxis,:,:]],axis=0)
                                offsets[pairID] = np.concatenate([offsets[pairID][np.newaxis,:],results[3][pairID][np.newaxis,:]],axis=0)
                            else:
                                corrs[pairID] = np.concatenate([corrs[pairID],results[1][pairID][np.newaxis,:,:]],axis=0)
                                offsets[pairID] = np.concatenate([offsets[pairID],results[3][pairID][np.newaxis,:]],axis=0)
                    if i+nCorr<groupSize:
                        procs[i+nCorr].correlate([masks,
                                                  poptT,poptF,
                                                  corrWidth,
                                                  nSearch,
                                                  predictedDelays[:,i+nCorr+groupStart],
                                                  self.config["Correlating"].getboolean("dmSearch",False),
                                                  self.polyCoFreq,
                                                  self.config["Correlating"].getboolean("dmPhase",False)])
                print(f'Correlations Complete at {(Time.now()-gs).to_value(u.min):.1f} minutes',flush=True)
                pairID = 0 
                for specID0 in range(len(correlationStreams)-1):
                    for deltaID1 in range(len(correlationStreams)-specID0-1):
                        specID1 = specID0+deltaID1+1
                        fname = f'BURSTT_{self.sourceName}_{self.obsDate}_{groupID}_{correlationStreams[specID0].name}_{correlationStreams[specID1].name}_{self.config["Correlating"].get("dedisperse")}'
                        if self.config["Correlating"].getboolean("dmSearch",False):
                            fname+='_dmS'
                        if self.config["Correlating"].getboolean("dmPhase",False):
                            fname+='_dmP'
                        fname+=f"_{self.config['Correlating'].getint('size',2048)}"
                        fname+='_corr.npz'
                        np.savez(os.path.join(self.correlationDir,fname),
                                 corr=corrs[pairID],
                                 corrTime=np.array(peakT[pairID]),
                                 offsets=offsets[pairID])
                        pairID+=1
                print(f'Group {groupID} complete after {(Time.now()-gs).to_value(u.min):.1f} minutes',flush=True)
        for proc in procs:
            proc.stop()

    def chunk_correlate(self,chunkWidth = 1*u.s,
                        maxShift = 100,
                        nPool = 80,
                        doCorr = True,
                        doFFT=True,
                        predictDelays = True):
        nCents = int(((self.latest-self.earliest)*u.day-chunkWidth/5.).to_value(u.s)//chunkWidth.to_value(u.s))
        tCents = (np.linspace(0,nCents-1,nCents))*chunkWidth + self.earliest*u.day+chunkWidth/2
        tCents = Time(tCents.to_value(u.day),format='mjd')
        for i in range(len(self.streams)):
            print(self.streamNames[i]+':')
            print(f'{Time(np.nanmin(self.streams[i].blockStart).value,format="unix").isot}-{Time(np.nanmax(self.streams[i].blockStart).value,format="unix").isot}')
            print('--------')
        print('Chunks:')
        print(f'{tCents[0].isot}-{tCents[-1].isot}')

        correlationStreamNames = np.array(self.config['Correlating'].get('streams').split(','))
        correlationStreams = []
        for name in correlationStreamNames:
            for i in range(self.streamNames.shape[0]):
                if self.streamNames[i]==name:
                    correlationStreams.append(self.streams[i])
        correlationStations = [stream.station for stream in correlationStreams]
        if predictDelays:
            predictedDelays = np.round((self.predict_delay(tCents,correlationStations)/self.streams[0].frameSec).to_value(u.dimensionless_unscaled)).astype(int)
        else:
            predictedDelays = np.zeros((self.nBaselines,tCents.shape[0]),dtype=int)

        groups = int(tCents.shape[0]//nPool) +1
        procs = []
        if self.verbose:
            print('Building Proccesses')
        for i in range(nPool):
            procs.append(SingleChunkProcess(i,self.verbose))
        for p in procs:
            p.start()
            p.give_obs(correlationStreams)
            p.front_pipe.recv()

        
            
        for groupID in range(groups):
            if self.verbose:
                print(f'Start group {groupID}')
            gs = Time.now()
            ## Determine group Indexes
            groupStart = groupID*nPool
            groupSize = min(nPool,tCents.shape[0]-groupStart)
        
            ## Load Data
            for i in range(groupSize):
                procs[i].load_data([tCents[i+groupStart],chunkWidth/2])
            if self.verbose:
                print(f'Reads Started at {(Time.now()-gs).to_value(u.min):.1f} minutes',flush=True)
            ## Read data from each process as it becomes available
            for i in range(groupSize):
                results = procs[i].front_pipe.recv()
                if i == 0: 
                    I = results[1]
                    var = results[2]
                else:
                    for obsID in range(len(correlationStreams)):
                        I[obsID] += results[1][obsID]
                        var[obsID] += results[2][obsID]
            if self.verbose:
                print(f'Reads Complete at {(Time.now()-gs).to_value(u.min):.1f} minutes',flush=True)
            ## Finalize Averaging
            I = [x/groupSize for x in I]
            Imeds = [x.mean(0) for x in I]
            for streamID in range(len(correlationStreams)):
                var[streamID]/=Imeds[streamID]**2
            var2 = [sigma_clip(x, sigma=2.0) for x in var]
            masks = [x.mask for x in var2]
            for streamID in range(len(correlationStreams)):
                I[streamID].mask = masks[streamID].reshape((-1,masks[streamID].shape[0]))
                I[streamID] = np.reshape(I[streamID]/Imeds[streamID],(I[streamID].shape[0],-1,32)).mean(2)
            ## Make Plots
            for streamID,stream in enumerate(correlationStreams):
                grid = plt.GridSpec(nrows=4,ncols=4)
                plotTime = np.linspace(0,I[streamID].shape[0]*400-1,
                                       I[streamID].shape[0]*400)*stream.frameSec.to(u.ms)
                plotTime = np.reshape(plotTime,(-1,400)).mean(1)
                plotFreq = np.reshape(stream.freq,(-1,32)).mean(1)
                ext = thth.ext_find(plotTime,plotFreq)
                plt.figure(figsize=(6,6))
                plt.subplot(grid[:3,:3])
                plt.imshow(I[streamID].T,
                           origin='lower',aspect='auto'
                           ,vmin=2-(I[streamID][:,:20].mean(1).max()),vmax=I[streamID][:,:20].mean(1).max(),
                           extent=ext)
                plt.ylabel('Freq (MHz)')
                plt.xticks([])
                plt.subplot(grid[-1,:3])
                plt.plot(plotTime,I[streamID][:,:20].mean(1))
                plt.xlabel('Time (ms)')
                plt.ylabel('I')
                plt.xlim(ext[:2])
                plt.subplot(grid[:3,-1])
                plt.plot(var[streamID],stream.freq)
                plt.plot(var2[streamID],stream.freq)
                plt.xlabel('Variance')
                plt.tight_layout()
                plt.savefig(os.path.join(self.plotDir,f'BURSTT_{self.sourceName}_{self.obsDate}_{groupID}_{stream.name}_I.png'))
                # np.savez(os.path.join(self.intensityDir,f'BURSTT_{self.sourceName}_{self.obsDate}_{groupID}_{station.station}_I_{self.ddis}.npz'),I=I[stationID],f=station.freq,t=plotTime)
            if doCorr:
                for procID,proc in enumerate(procs[:min(40,groupSize)]):
                    proc.correlate([masks,2*maxShift,doFFT,predictedDelays[:,procID+groupStart,Imeds]])
                print(f'Correlations Started at {(Time.now()-gs).to_value(u.min):.1f} minutes',flush=True)
                for i,p in enumerate(procs[:groupSize]):
                    results = p.front_pipe.recv()
                    if i == 0:
                        corrs = results[1]
                        offsets = results[2]
                    else:
                        for pairID in range(len(results[1])):
                            if i==1:
                                corrs[pairID] = np.concatenate([corrs[pairID][np.newaxis,:,:],results[1][pairID][np.newaxis,:,:]],axis=0)
                                offsets[pairID] = np.concatenate([offsets[pairID][np.newaxis,:],results[2][pairID][np.newaxis,:]],axis=0)
                            else:
                                corrs[pairID] = np.concatenate([corrs[pairID],results[1][pairID][np.newaxis,:,:]],axis=0)
                                offsets[pairID] = np.concatenate([offsets[pairID],results[2][pairID][np.newaxis,:]],axis=0)
                    if i+40<groupSize:
                        procs[i+40].correlate([masks,2*maxShift,doFFT,predictedDelays[:,i+40+groupStart]])
                print(f'Correlations Complete at {(Time.now()-gs).to_value(u.min):.1f} minutes',flush=True)
                pairID = 0 
                for specID0 in range(len(correlationStreams)-1):
                    for deltaID1 in range(len(correlationStreams)-specID0-1):
                        specID1 = specID0+deltaID1+1
                        np.savez(os.path.join(self.correlationDir,f'BURSTT_{self.sourceName}_{self.obsDate}_{groupID}_{correlationStreams[specID0].name}_{correlationStreams[specID1].name}_corr.npz'),
                                 corr=corrs[pairID],
                                 corrTime=(tCents-correlationStreams[0].epoch0).to_value(u.s),
                                 offsets=offsets[pairID])
                        pairID+=1
                print(f'Group {groupID} complete after {(Time.now()-gs).to_value(u.min):.1f} minutes',flush=True)
        for proc in procs:
            proc.stop()


    def get_clock_correction(self,times,station0='FUS',station1='LTN'):
        pppBaseDir = '/burstt_home/shared/gnss/ppp'

        hasFirst=False

        for deltaDay in [-1,0,1]:
            year = (times[0]+deltaDay*u.day).datetime.year
            dt = (times[0]+deltaDay*u.day).datetime
            dayOfYear=int((times[0]+deltaDay*u.day-Time(f'{year}-01-01T00:00:00')).to_value(u.day))+1
            print(f'Year: {year}')
            print(f'Day: {dayOfYear}')
            ppp0Dir = f'{station0}-{year}-{dayOfYear}-full_output'
            ppp1Dir = f'{station1}-{year}-{dayOfYear}-full_output'
            try:
                ppp1File = [f for f in os.listdir(os.path.join(pppBaseDir,ppp1Dir)) if f[-3:]=='csv'][0]
                ppp0File = [f for f in os.listdir(os.path.join(pppBaseDir,ppp0Dir)) if f[-3:]=='csv'][0]
            except Exception as e:
                print(e)
                continue
            
            ppp0 = np.genfromtxt(os.path.join(pppBaseDir,ppp0Dir,ppp0File),delimiter = ',')[1:]
            ppp1 = np.genfromtxt(os.path.join(pppBaseDir,ppp1Dir,ppp1File),delimiter = ',')[1:]

            if hasFirst:
                year0 = ppp0[:,-2]
                day0 = ppp0[:,-3]
                hour0 = ppp0[:,-4]
                clock0New = ppp0[:,-1]*u.ns
                year1 = ppp1[:,-2]
                day1 = ppp1[:,-3]
                hour1 = ppp1[:,-4]
                clock1New = ppp1[:,-1]*u.ns
                ppp0TimeNew = Time(np.array([f'{int(x)}-01-01T00:00:00' for x in year0]))+(day0-1)*u.day+hour0*u.hour
                ppp1TimeNew = Time(np.array([f'{int(x)}-01-01T00:00:00' for x in year1]))+(day1-1)*u.day+hour1*u.hour
                clock1 = np.concatenate((clock1,clock1New[ppp1TimeNew.mjd>ppp1Time.mjd.max()]))
                clock0 = np.concatenate((clock0,clock0New[ppp0TimeNew.mjd>ppp0Time.mjd.max()]))
                ppp1Time = Time(np.concatenate((ppp1Time.mjd,ppp1TimeNew[ppp1TimeNew.mjd>ppp1Time.mjd.max()].mjd)),format='mjd')
                ppp0Time = Time(np.concatenate((ppp0Time.mjd,ppp0TimeNew[ppp0TimeNew.mjd>ppp0Time.mjd.max()].mjd)),format='mjd')

            else:
                year0 = ppp0[:,-2]
                day0 = ppp0[:,-3]
                hour0 = ppp0[:,-4]
                clock0 = ppp0[:,-1]*u.ns
                year1 = ppp1[:,-2]
                day1 = ppp1[:,-3]
                hour1 = ppp1[:,-4]
                clock1 = ppp1[:,-1]*u.ns
                ppp0Time = Time(np.array([f'{int(x)}-01-01T00:00:00' for x in year0]))+(day0-1)*u.day+hour0*u.hour
                ppp1Time = Time(np.array([f'{int(x)}-01-01T00:00:00' for x in year1]))+(day1-1)*u.day+hour1*u.hour
                hasFirst=True


        # FUS_clock-=FUS_clock.mean()
        # LTN_clock-=LTN_clock.mean()

        interp0 = interp1d(ppp0Time.mjd,clock0.to_value(u.us))
        interp1 = interp1d(ppp1Time.mjd,clock1.to_value(u.us))

        clockCorr = interp1(times.mjd+(18*u.s).to_value(u.day))*u.us - interp0(times.mjd+(18*u.s).to_value(u.day))*u.us
        return(clockCorr)

    def plot_shifts(self,suffix,rolledDelays,fullToAs,fullArr,shiftedDelays,shiftedArr,peakOffset):
        grid = plt.GridSpec(nrows=1,ncols=2,width_ratios=[3,1])
        plt.figure(figsize=(8,16))
        plt.subplot(grid[0,0])
        ext = thth.ext_find(rolledDelays,fullToAs)
        plt.imshow(np.abs(fullArr),
                origin='lower',aspect='auto',
                vmin=1,vmax=3,
                extent=ext,cmap='Greys')
        plt.ylabel(r'$\rm{Arrival~Time}~\left(\rm{s}\right)$')
        plt.xlabel(r'$\rm{Delay}~\left(\mu\rm{s}\right)$')

        ext2 = thth.ext_find(shiftedDelays.to(u.us),fullToAs)
        plt.subplot(grid[0,1])
        plt.imshow(np.abs(shiftedArr),
                origin='lower',aspect='auto',
                vmin=1,vmax=3,
                extent=ext2,cmap='Greys')
        plt.yticks([])
        plt.xlabel(r'$\rm{Residual~Delay}~\left(\mu\rm{s}\right)$')
        plt.xlim((shiftedDelays[peakOffset].to_value(u.us)-.1,shiftedDelays[peakOffset].to_value(u.us)+.1))
        plt.tight_layout()
        plt.savefig(os.path.join(self.plotDir,f'BURSTT_{self.sourceName}_{self.obsDate}_{suffix}.png'))
        plt.close()

    def examine_correlation(self):
        skipFiles = self.config['Correlating'].get('fileSkip','').split(',')
        skipFiles = [x for x in skipFiles if len(x)>0]
        epoch0 = Time(self.observationPars.get('date') + ' 16:00:00')
        epoch0s = epoch0.to_value('unix')*u.s
        corrName0 = f"BURSTT_{self.sourceName}_{self.obsDate}_0_{self.streamNames[0]}_{self.streamNames[1]}"
        if self.sourceType == 'pulsar':
            corrName0+=f'_{self.config["Correlating"].get("dedisperse")}'
        corrName0+="_corr.npz" 
        arch = np.load(f'{self.correlationDir}/{corrName0}')
        freqs = [self.streams[0].freq,self.streams[1].freq]
        fref=np.max([freqs[i].min().value for i in range(len(freqs))])*u.MHz
        coarse = arch['offsets']*u.us
        fine = thth.fft_axis(np.linspace(0,arch['corr'].shape[-1]-1,arch['corr'].shape[-1])*(freqs[0][1]-freqs[0][0]),u.ns)
        fullArr = None
        fullToAs = np.array([])*u.s
        used = np.array([],dtype=bool)
        coarse0s = np.array([])*u.us
        # delaysFull = np.array([])*u.us
        cs = -2
        nc = 5
        i=0
        unbuilt = True
        while True:
            if str(i) in skipFiles:
                i+=1
                if not unbuilt:
                    ncorr = int(self.config['Correlating'].getfloat('nPool',80))
                    ToAs = np.linspace(1,ncorr,ncorr)*np.diff(fullToAs).mean() + fullToAs[-1]
                    fullToAs = np.concatenate((fullToAs,ToAs))
                    fullArr=np.concatenate((fullArr,np.zeros((ncorr,fullArr.shape[1]))*np.nan),axis=0)
                    used = np.concatenate((used,np.zeros(ncorr,dtype=bool)))
                continue
            try:
                corrName = f"BURSTT_{self.sourceName}_{self.obsDate}_{i}_{self.streamNames[0]}_{self.streamNames[1]}"
                if self.sourceType == 'pulsar':
                    corrName+=f'_{self.config["Correlating"].get("dedisperse")}'
                corrName+="_corr.npz" 
                arch = np.load(f'{self.correlationDir}/{corrName}')
            except Exception as e:
                print(e)
                break
            try:
                coarse = arch['offsets']*u.us
                coarse0s = np.concatenate((coarse0s,coarse[:,0]))
                if self.sourceType == 'pulsar':
                    c = arch['corr'][:,0,:,:]
                else:
                    c = arch['corr']
                ToAs = arch['corrTime']*u.s - epoch0.unix*u.s
                fullToAs = np.concatenate((fullToAs,ToAs))
                if unbuilt:
                    delaysFull = np.concatenate([coarse[:,i][:,np.newaxis]+fine[np.newaxis,:] for i in range(coarse.shape[1])],axis=1)
                    unbuilt = False
                else:
                    delaysFull =np.concatenate((delaysFull,np.concatenate([coarse[:,i][:,np.newaxis]+fine[np.newaxis,:] for i in range(coarse.shape[1])],axis=1)),axis=0)
                conArr = np.reshape(np.ravel(c),(c.shape[0],-1))
                if not isinstance(fullArr,np.ndarray):
                    fullArr = conArr
                else:
                    fullArr = np.concatenate((fullArr,conArr),axis=0)
                used = np.concatenate((used,np.ones(c.shape[0],dtype=bool)))
            except Exception as e:
                ToAs = np.linspace(1,c.shape[0],c.shape[0])*np.diff(fullToAs).mean() + fullToAs[-1]
                fullToAs = np.concatenate((fullToAs,ToAs))
                fullArr=np.concatenate((fullArr,np.zeros((c.shape[0],2560))*np.nan),axis=0)
                used = np.concatenate((used,np.zeros(c.shape[0],dtype=bool)))
            i+=1
        stepoff = np.round((coarse0s-coarse0s.min())/(2.56*u.us)).astype(int)
        fullArr = np.pad(fullArr,((0,0),(0,c.shape[2]*stepoff.max())),mode='constant',constant_values=np.nan)
        for i in range(fullArr.shape[0]):
            fullArr[i] = np.roll(fullArr[i],stepoff[i]*c.shape[2])
        rolledDelays = np.linspace(delaysFull.min(),delaysFull.max(),fullArr.shape[1])
        fullArr0s = np.nan_to_num(fullArr)

        times = epoch0+fullToAs
        delayVLBI = self.predict_delay(times,[self.streams[0].station,self.streams[1].station])[0].to(u.us)
        print(f'Average VLBI Delay: {delayVLBI.mean().to(u.us):0.2f}')
        print(f'Center VLBI Delay: {delayVLBI[int(delayVLBI.shape[0]//2)].to(u.us):0.2f}')
        VLBIPhases = (2*np.pi*u.rad*delayVLBI*fref).to_value(u.rad)

        slopes = np.linspace(-.005,.005,100)*u.us/u.s
        maxes = np.zeros(slopes.shape[0])
        for slopeID in tqdm(range(slopes.shape[0])):
            fullDelay = delayVLBI + slopes[slopeID]*(fullToAs-fullToAs.mean())
            shiftedArr,shiftedDelays = shiftArr(fullArr0s,rolledDelays,fullDelay)
            maxes[slopeID]=np.nansum(np.abs(shiftedArr),0).max()
        delayLinear = (slopes[np.argmax(maxes)]*(fullToAs-fullToAs.mean())).to(u.us)

        fullDelay = delayVLBI + delayLinear
        shiftDelta = (fullDelay.mean()-rolledDelays.mean())
        
        shiftedArr,shiftedDelays = shiftArr(fullArr0s,rolledDelays,fullDelay-shiftDelta)
        shiftedDelays-=shiftDelta

        temp = np.nanmean(np.abs(shiftedArr**2),0)

        peakOffset = np.argwhere(temp==np.nanmax(temp)).max()

        self.plot_shifts('rawshifts',rolledDelays,fullToAs,fullArr,shiftedDelays,shiftedArr,peakOffset)
        
        slicedCorr = shiftedArr[:,peakOffset-5:peakOffset+6].mean(1)

        clockCorr = self.get_clock_correction(times)    

        clockPhases = (2*np.pi*u.rad*clockCorr*fref).to_value(u.rad)
        modClockPhases = np.mod(clockPhases+np.pi,2*np.pi)-np.pi

        np.savez(os.path.join(self.dateDir,'combinedCorr'),
                 fullArr = fullArr,
                 rolledDelays = rolledDelays.to_value(u.us),
                 delayVLBI = delayVLBI,
                 clockCorr = clockCorr.to_value(u.us),
                 slicedCorr = slicedCorr
                 )

        correctedCorrs = slicedCorr*np.exp(-1j*VLBIPhases)

        angs = np.angle(correctedCorrs)
        angs2 = np.angle(correctedCorrs*np.exp(-1j*clockPhases))
        angsUnwrap = unwrap(angs,100)
        angs2Unwrap = unwrap(angs2,100)
        clockPlot = clockPhases-2*np.pi*(clockPhases[0]//(2*np.pi))
        if clockPlot[0]>np.pi:
            clockPlot-=np.pi
        if clockPlot[0]<-np.pi:
            clockPlot+=np.pi

        grid = plt.GridSpec(nrows=4,ncols=1)
        plt.figure(figsize=(6,6))
        plt.subplot(grid[0,0])
        plt.plot(fullToAs-fullToAs[0],angs,'.')
        plt.plot(fullToAs-fullToAs[0],modClockPhases)
        plt.ylabel('Phase (rad)')
        plt.xticks([])
        plt.title('Pre-Clock')
        plt.subplot(grid[1,0])
        plt.plot(fullToAs-fullToAs[0],(angsUnwrap/(2*np.pi*fref)).to(u.ns),'.')
        plt.plot(fullToAs-fullToAs[0],(clockPlot/(2*np.pi*fref)).to(u.ns))
        plt.ylabel('Delay (ns)')
        plt.xticks([])
        plt.subplot(grid[2,0])
        plt.plot(fullToAs-fullToAs[0],angs2,'.')
        plt.xticks([])
        plt.ylabel('Phase (rad)')
        plt.title('Post-Clock')
        plt.subplot(grid[3,0])
        plt.plot(fullToAs-fullToAs[0],(angs2Unwrap/(2*np.pi*fref)).to(u.ns),'.')
        plt.ylabel('Delay (ns)')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plotDir,f'BURSTT_{self.sourceName}_{self.obsDate}_rawshifts_residuals.png'))
        plt.close()

        fullPhase = VLBIPhases+clockPhases+angs2Unwrap

        slopes = np.linspace(-.005,.005,100)*u.us/u.s
        maxes = np.zeros(slopes.shape[0])
        for slopeID in tqdm(range(slopes.shape[0])):
            fullDelay = delayVLBI + clockCorr + slopes[slopeID]*(fullToAs-fullToAs.mean())
            shiftedArr,shiftedDelays = shiftArr(fullArr,rolledDelays,fullDelay)
            maxes[slopeID]=np.nansum(shiftedArr*np.exp(-1j*fullPhase[:,np.newaxis]),0).real.max()
        delayLinear = (slopes[np.argmax(maxes)]*(fullToAs-fullToAs.mean())).to(u.us)
        
        fullDelay = delayVLBI + delayLinear + clockCorr
        shiftDelta = (fullDelay.mean()-rolledDelays.mean())
        
        shiftedArr,shiftedDelays = shiftArr(fullArr0s,rolledDelays,fullDelay-shiftDelta)
        shiftedDelays-=shiftDelta

        temp = np.nanmean(np.abs(shiftedArr**2),0)

        peakOffset = np.argwhere(temp==np.nanmax(temp)).max()

        self.plot_shifts('refinedshifts',rolledDelays,fullToAs,fullArr,shiftedDelays,shiftedArr,peakOffset)
        
        slicedCorr = shiftedArr[:,peakOffset-5:peakOffset+6].mean(1)

        correctedCorrs = slicedCorr*np.exp(-1j*VLBIPhases)

        angs = np.angle(correctedCorrs)
        angs2 = np.angle(correctedCorrs*np.exp(-1j*clockPhases))
        angsUnwrap = unwrap(angs,100)
        angs2Unwrap = unwrap(angs2,100)
        clockPlot = clockPhases-2*np.pi*(clockPhases[0]//(2*np.pi))
        if clockPlot[0]>np.pi:
            clockPlot-=np.pi
        if clockPlot[0]<-np.pi:
            clockPlot+=np.pi

        grid = plt.GridSpec(nrows=4,ncols=1)
        plt.figure(figsize=(6,6))
        plt.subplot(grid[0,0])
        plt.plot(fullToAs-fullToAs[0],angs,'.')
        plt.plot(fullToAs-fullToAs[0],modClockPhases)
        plt.ylabel('Phase (rad)')
        plt.xticks([])
        plt.title('Pre-Clock')
        plt.subplot(grid[1,0])
        plt.plot(fullToAs-fullToAs[0],(angsUnwrap/(2*np.pi*fref)).to(u.ns),'.')
        plt.plot(fullToAs-fullToAs[0],(clockPlot/(2*np.pi*fref)).to(u.ns))
        plt.ylabel('Delay (ns)')
        plt.xticks([])
        plt.subplot(grid[2,0])
        plt.plot(fullToAs-fullToAs[0],angs2,'.')
        plt.xticks([])
        plt.ylabel('Phase (rad)')
        plt.title('Post-Clock')
        plt.subplot(grid[3,0])
        plt.plot(fullToAs-fullToAs[0],(angs2Unwrap/(2*np.pi*fref)).to(u.ns),'.')
        plt.ylabel('Delay (ns)')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plotDir,f'BURSTT_{self.sourceName}_{self.obsDate}_rawshifts_residuals.png'))
        plt.close()

    def pulse_SN(self,nPool = 80):
        if self.verbose:
            print('Start SNR Measurements')
        SNStreamNames = np.array(self.config['SN'].get('streams').split(','))
        SNStreams = []
        for name in SNStreamNames:
            for i in range(self.streamNames.shape[0]):
                if self.streamNames[i]==name:
                    SNStreams.append(self.streams[i])
        for stream in SNStreams:
            stream.read_profile_file()
            stream.get_flat_profile()
        correlationStations = [stream.station for stream in SNStreams]
        ToAs = np.copy(SNStreams[0].ToAs)
        for stream in SNStreams:
            maxDispdelay = ((stream.DM/K)*(1/(stream.freq.min()**2)-1/(self.polyCoFreq**2))).to(u.s)
            ToAs = ToAs[(ToAs*u.s>stream.earliest+self.pulseWidth)*(ToAs*u.s<stream.latest-self.pulseWidth-maxDispdelay)]
        groups = int(ToAs.shape[0]//nPool) +1
        procs = []
        if self.verbose:
            print('Building Proccesses')
        for i in range(nPool):
            procs.append(SinglePulseProcess(i,self.verbose))
        for p in procs:
            p.start()
            p.give_obs(SNStreams)
            p.front_pipe.recv()
        SNs = np.zeros((ToAs.shape[0],len(SNStreams)))
        for groupID in range(groups):
            if self.verbose:
                print(f'Start group {groupID}')
            gs = Time.now()
            ## Determine group Indexes
            groupStart = groupID*nPool
            groupSize = min(nPool,ToAs.shape[0]-groupStart)
        
            ## Load Data
            for i in range(groupSize):
                procs[i].load_data_SNR([self.polyCoFreq,Time(ToAs[groupStart+i],format='unix'),self.pulseWidth,self.config['Correlating'].get('dedisperse')=='coherent'])
            if self.verbose:
                print(f'Reads Started at {(Time.now()-gs).to_value(u.min):.1f} minutes coherent is {self.config["Correlating"].get("dedisperse")=="coherent"}',flush=True)
            ## Read data from each process as it becomes available
            for i in range(groupSize):
                results = procs[i].front_pipe.recv()
                print(results)
                SNs[groupStart+i]=results[1]
                
        for i in range(len(SNStreamNames)):
            specsFile = f'SNR_{self.obsDate}_{self.sourceName}_{SNStreamNames[i]}.npz'
            np.savez(os.path.join(self.dateDir,specsFile),SNR=SNs[:,i],ToA = ToAs)
        for proc in procs:
            proc.stop()