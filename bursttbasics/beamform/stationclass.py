import os
import re
import astropy.units as u
import numpy as np
from glob import glob
import astropy.constants as const
from astropy.coordinates import SkyCoord

import sys
sys.path.insert(0, "/data/kylin/bin/")
import beamform as bf
from astropy.time import Time
from configparser import ConfigParser

import warnings
warnings.filterwarnings('ignore')
from .bfutils import *
from ..packet_func import *


class Station:
    def __init__(self,nAnt =16,nRow = 16,
                 nChan = 1024, nOrder = 8,
                 BW = 400*u.MHz, f0 = 400*u.MHz,
                 fRef = 400*u.MHz,
                 xBeam0 = -7.5,sepX = 1.0*u.m,
                 yBeam0 = -7.5,sepY = 0.5*u.m,
                 byteMeta = 64,bytePack = 8256,
                 orderOff = 0,
                 fpp = 4,fl2int = 256,beamform = bf.beamform256,calib = 'calibration',skipRows = [],
                 targetIP = '140.109.178.116',
                 stationID = 'STN',bwLimit=30,saveDir='./'
                ):
        
        self.set_cal_options(calib)
        self.orderOff = orderOff
        
        self.beamform = beamform
        self.fl2int = fl2int
        
        self.fpp = fpp
        self.byteMeta = byteMeta
        self.bytePack = bytePack
        
        self.sepX = sepX # antenna separation in X, meters
        self.sepY = sepY # in Y, meters
        self.xBeam0 = xBeam0
        self.yBeam0 = yBeam0
    
    
        self.nAnt = nAnt
        self.nRow = nRow
        self.nChan = nChan
        self.nOrder = nOrder
        self.BW = BW << u.MHz
        self.f0 = f0 << u.MHz
        self.fRef = fRef << u.MHz
        self.skipRows = []
        self.targetIP = targetIP
        self.stationID = stationID
        self.bwLimit = bwLimit
        self.calFormat='.txt'
        self.saveDir = saveDir
        
    def set_cal_options(self,calDir,verbose=False):
        if os.path.isdir(calDir):
            self.calibrationDir = calDir
            if self.calibrationDir[-1]=='/':
                self.calibrationDir = self.calibrationDir[:-1]
            self.calibrations = glob(f'{self.calibrationDir}/*.check')+glob(f'{self.calibrationDir}**/*/*.check')
            self.calibrations = [cal for cal in self.calibrations if cal[-6:]=='.check']
            self.calibrations = np.array([cal for cal in self.calibrations if cal.split('/')[-1][:3]=='cal'])
            self.calibrationTimes = [cal.split('/')[-1].strip('cal_').strip('.check').replace('Z','').replace('_','') for cal in self.calibrations]
            self.calibrationTimes = [str.join('',re.split('-aref\d',cal)) for cal in self.calibrationTimes]
            self.calibrationTimes = Time([time_from_event(cal[:8]+'_'+cal[8:]+'Z') for cal in self.calibrationTimes],scale='utc')
        else:
            if verbose:
                print('Calibration Dir Does Not Exist')
            
    def get_2nd_cal(self,calFile,verbose=False):
            '''
            load the delay calibration between FPGAs
            ifile lists the delay of each FPGA in ns
            freq is an array (length nChan) in MHz
        
            return
            '''
            self.calFile = calFile

            if calFile[-4:]=='.txt':
                with open(calFile, 'r') as fh:
                    tmp = fh.readline() # skip the first line
                    line = fh.readline()
                    tmp = line.split("'")[1].split()
                    tau = np.array([float(x) for x in tmp])
                    if verbose:
                        print('delays (ns):', tau)
                self.cal2 = np.exp(2.j*np.pi*tau.reshape(-1,1)*self.freq.reshape(1,-1).to_value(u.MHz)*1e-3).reshape(1,self.nRow,self.nChan)
            else:
                tmp = np.load(calFile)
                #print(tmp.files)
                V = tmp['tau_i'].T  # eigenvector with instrumental delay and weighting
                                    # shape (nRow, nChan), after .T
                auto = tmp['auto2'] # relative amplitude betweeb rows
                                    # shape (nRow, nChan)
                phiCorr = np.exp(-1.j * np.angle(V)).reshape(1,self.nRow,self.nChan)
                normCorr = (np.abs(V)/0.25 / auto).reshape(1,self.nRow,self.nChan)
                self.cal2 = normCorr*phiCorr

    def find_best_cal(self,time,verbose=False):
        preMask = (self.calibrationTimes<=time)
        bestID = np.argmin(np.abs(self.calibrationTimes[preMask]-time))
        npzFile = os.path.join(self.calibrations[preMask][bestID],"solution_2ndCal.npz")
        if self.calFormat=='.npz' and os.path.isfile(npzFile):
            self.get_2nd_cal(npzFile,verbose=verbose)
        else:
            self.get_2nd_cal(os.path.join(self.calibrations[preMask][bestID],"ant_delay_correct.txt"),verbose=verbose)
        if verbose:
            print(f'Best Calibration: {self.calibrations[preMask][bestID].split("/")[-1]}')
        
    def get_BFMY(self,sinThetaY,beamIDX,norm=True,files=[],nSample = 1024,verbose=False):
        BFMY = np.exp(2.j*np.pi*self.pos2.reshape(1,self.nRow,1)*sinThetaY/self.lamb.reshape(1,1,self.nChan))
        BFMY[:,self.skipRows,:]=0
        BFMY = BFMY * self.cal2

        normFile = os.path.join(os.path.dirname(self.calFile),"ant_norm_correct.txt")
        if self.calFile[-4:]=='.txt' and os.path.isfile(normFile):
            with open(normFile, 'r') as fh:
                tmp = fh.readline() # skip the first line
                line = fh.readline()
                tmp = line.split("'")[1].split()
                norms = np.array([float(x) for x in tmp])
            BFMY*=norms[np.newaxis,:,np.newaxis]
        elif norm and self.calFile[-4:]=='.txt' and len(files)>0:
            amp = np.zeros((self.nRow, self.nChan))
            nFile = len(files)
            for fi in range(nFile):
                currentFile = files[fi]
                ##################################
                try:
                    sz = os.path.getsize(currentFile)   # follows symlink; raises if target missing
                except (OSError, FileNotFoundError) as e:
                    if verbose:
                        print(f"Skipping broken/missing symlink: {currentFile}  ({e})")
                    continue
                # skip zero-length target files
                if sz == 0:
                    if verbose:
                        print(f"Skipping (size=0): {currentFile}")
                    continue
                ############################
                with open(currentFile, 'rb') as fh:
                    mt = fh.read(64)
                    hd = fh.read(64)
                    tmp = decHeader2(hd)
                    od = tmp[4]
                ch1 = self.nSubCh*(od - self.orderOff)
                ch2 = ch1 + self.nSubCh
                if verbose:
                    print('... order:',od)
            
                fpga = np.memmap(currentFile, dtype='u1', mode='r')
                p0 = 0
                data_off = self.byteMeta + self.bytePack * p0
            
                v1 = np.zeros((self.nSubCh, self.nAnt, 2), dtype='<i2')
                v1[:, beamIDX, 0] = self.fl2int
            
                for ri in range(self.nRow):
                    data = fpga[data_off:]
                    v2 = np.zeros((self.nSubCh, self.nRow, 2), dtype='<i2')
                    v2[:, ri, 0] = self.fl2int
            
                    res = np.ones((self.nSubCh, nSample), dtype='<c8')
                    self.beamform(res, data, v1, v2, nSample, self.scale)
                    amp[ri,ch1:ch2] = np.abs(res).mean(axis=1)
            ## renormalize amp so that median of rows is 1
            med_amp = np.median(amp, axis=1)    # med of each row
            med_med_amp = np.median(med_amp)    # med of all rows
            amp /= med_med_amp
            
            ## normalize BFMY according to the amp
            ## and zero out channels or rows that have low sensitivity
            BFMY = BFMY / amp.reshape(1,self.nRow,self.nChan)
            w_low_amp = amp<0.2
            BFMY[0][w_low_amp] *= 0.
        BFMY[:,self.skipRows,:]=0
        return(BFMY)

    def beamIDs_from_FUS(self,beamIDF):
        beamIDFX = beamIDF%16
        beamIDFY = beamIDF//16
        Ybeam0F = -7.5 #for nantou ##-7.5 for fushan
        sepYF = 0.5*u.m
        nRowF = 16
        nYBeamF = nRowF
        sinThetaYF = (self.lamb0/sepYF/nRowF*(np.arange(nYBeamF)+Ybeam0F)).to_value(u.dimensionless_unscaled)
        thetaYF = np.arcsin(sinThetaYF) # rad
        anglesYF = thetaYF/np.pi*180. # deg
        angleIDY = anglesYF[beamIDFY]
        sinThetaIDY = sinThetaYF[beamIDFY]
        return(beamIDFX,sinThetaIDY,beamIDFY)

    def beamform_event(self,files,beamIDF,DM,ep0,secWin = 0.03*u.s,timeOffset = 0*u.s,find_cal=True,skipOrder=[],verbose=False): 
        hasData = False
        epPulse = (Time(ep0, scale='utc').to_value('unix')*u.s+(4148.8*(u.s*u.MHz**2) * DM * (1/self.freq**2 - 1/self.fRef**2))).to(u.s)
        epPulse+=timeOffset
        if find_cal:
            self.find_best_cal(Time(ep0, scale='utc'))
        
        beamIDX,sinThetaIDY,beamIDY =  self.beamIDs_from_FUS(beamIDF)
        BFMY = self.get_BFMY(sinThetaIDY,beamIDX,files=files,verbose=verbose)

        pixWin = int(secWin * 2 / self.timeFrame)
        pixWin = (pixWin // 4) * 4
        
        ddSpec = np.zeros((self.nChan, pixWin), dtype=np.complex64)
        startTimes = np.zeros(self.nChan)*u.s

        combined_dd_inten = []
        combinedFreq = []      # To store frequencies per file
        winLenList = []  

        
        for i in range(len(files)):
            fbin = files[i]
            try:
                sz = os.path.getsize(fbin)   # follows symlink; raises if target missing
            except (OSError, FileNotFoundError) as e:
                if verbose:
                    print(f"Skipping broken/missing symlink: {fbin}  ({e})")
                continue
            
            # skip zero-length target files
            if sz == 0:
                if verbose:
                    print(f"Skipping (size=0): {fbin}")
                continue
            packMax = (sz - self.byteMeta) // self.bytePack
        
            with open(fbin, 'rb') as fh:
                mt = fh.read(64)
                hd = fh.read(64)
                tmp = decHeader2(hd)
                od = tmp[4]
                if verbose:
                    print(f'\n--- Processing File {i}: {fbin} (order {od}) ---')
            if od in skipOrder:
                continue
            epBegin = tmp[2]*u.s + 2*u.s + (tmp[0] - tmp[4]) // self.nOrder * self.fpp * self.timeFrame
            epEnd = epBegin + packMax // self.nRow * self.fpp * self.timeFrame
            chLow = self.nSubCh * (od-self.orderOff)
            chHigh = self.nSubCh * (od-self.orderOff + 1) - 1
            epLow = epPulse[chLow]
            epHigh = epPulse[chHigh]
    
            # p1, p2 are nearest packet to the requested frame
            ## fix: increase margin to make sure [p1,p2] convers the window
            p1 = int((epHigh - secWin - epBegin) / (self.fpp * self.timeFrame) - 1) * self.nRow
            p2 = int((epLow + secWin - epBegin) / (self.fpp * self.timeFrame) + 1) * self.nRow
            if p1<0 or p2>=packMax:
                continue
            # p1 = max(p1,0)
            # p2 = min(p2,packMax-1)
            if p2<p1:
                continue
            #p1 = int((ep_h - secWin - ep_begin) / (frame_per_pack * timeFrame)) * nRow
            #p2 = int((ep_l + secWin - ep_begin) / (frame_per_pack * timeFrame)) * nRow
            nPack = p2 - p1
            nFrame = int(nPack / self.nRow * self.fpp)
            nPack = nFrame * int(self.nRow / self.fpp)
            p2 = p1 + nPack

            if verbose:
                print(f'File {i}: packMax={packMax}, ep_begin={epBegin:.4f}, ep_end={epEnd:.4f}')
                print(f'File {i}: p1={p1}, p2={p2}, nPack={nPack}, nFrame={nFrame}')
    
            ## nFrame should be multiple of 16
            nPack = int(nFrame / 16) * 16 * 4
            nFrame = nPack / 4
            nSample = int(nFrame)

    
            fpga = np.memmap(fbin, dtype='u1', mode='r')
            p0 = p1
            data_off = self.byteMeta + self.bytePack * p0
            data = fpga[data_off:]
    
            v1 = np.zeros((self.nSubCh, self.nAnt, 2), dtype='<i2')
            v1[:, beamIDX, 0] = self.fl2int
    
            BFMYvec = BFMY[0, :, chLow:chLow + self.nSubCh].T * self.fl2int
            v2 = BFMYvec.ravel().view('<f8').reshape((self.nSubCh, self.nRow, 2)).astype('<i2')
    
            res = np.ones((self.nSubCh, nSample), dtype='<c8')
            self.beamform(res, data, v1, v2, nSample, self.scale)
    
            ep1 = epBegin + (p1//self.nRow)*self.fpp*self.timeFrame
            pixStart = ((epPulse[chLow:(chLow+self.nSubCh)] - secWin - ep1) / self.timeFrame).to_value(u.dimensionless_unscaled).astype(int)
            startTimes[chLow:chLow+self.nSubCh] = ep1
            startTimes[chLow:chLow+self.nSubCh] += pixStart*self.timeFrame
            if verbose:
                print('first pix:', pixStart[-1])
    
            for chi in range(self.nSubCh):
                ch = chi + chLow
                if pixStart[chi]>=0:
                    pp = pixStart[chi]
                    tmp = res[chi, pp:pp+pixWin]
                    max_len = tmp.size
                    ddSpec[ch,:max_len] = tmp[:]
                else:
                    pp = pixStart[chi]
                    if pp+pixWin>0:
                        tmp = res[chi, :pixWin+pp]
                        max_len = tmp.size
                        ddSpec[ch,-pp:-pp+max_len] = tmp[:]
            hasData=True
        return(ddSpec,startTimes,beamIDX,beamIDY,hasData)
        
    def fold_on_target(self,files,ep0,nSample=20000,period = 1*u.s,DM = 0,nBins = 256,find_cal=True,**kwargs):
        ep0T = Time(ep0, scale='utc')
        if find_cal:
            self.find_best_cal(ep0T)
        profile = np.zeros((self.nChan,nBins))
        counts = np.zeros((self.nChan,nBins),dtype=int)
        if 'beamIDF' in kwargs:
            beamIDX,sinThetaIDY,beamIDY =  self.beamIDs_from_FUS(kwargs['beamIDF'])
        elif 'source' in kwargs:
            beamIDF = determine_beam(kwargs['srce'],ep0T)
            beamIDX,sinThetaIDY,beamIDY =  self.beamIDs_from_FUS(beamIDF)
        elif 'sourceName' in kwargs:
            beamIDF = determine_beam(SkyCoord.from_name(kwargs['sourceName']),ep0T)
            beamIDX,sinThetaIDY,beamIDY =  self.beamIDs_from_FUS(beamIDF)
        elif 'ra' in kwargs and 'dec' in kwargs:
            beamIDF = determine_beam(SkyCoord(ra = kwargs['ra']<<u.deg,dec=kwargs['dec']<<u.deg),ep0T)
            beamIDX,sinThetaIDY,beamIDY =  self.beamIDs_from_FUS(beamIDF)
        else:
            print('No Target Specified')
            return(profile)
        BFMY = self.get_BFMY(sinThetaIDY,beamIDX,files=files)
        DMdels = (4148.8*(u.s*u.MHz**2) * DM * (1/self.freq**2 - 1/self.fRef**2))
        for i in range(len(files)):
            fbin = files[i]
            try:
                sz = os.path.getsize(fbin)   # follows symlink; raises if target missing
            except (OSError, FileNotFoundError) as e:
                print(f"Skipping broken/missing symlink: {fbin}  ({e})")
                continue
            
            # skip zero-length target files
            if sz == 0:
                print(f"Skipping (size=0): {fbin}")
                continue
            packMax = (sz - self.byteMeta) // self.bytePack
            with open(fbin, 'rb') as fh:
                mt = fh.read(64)
                hd = fh.read(64)
                tmp = decHeader2(hd)
                od = tmp[4]
                print(f'\n--- Processing File {i}: {fbin} (order {od}) ---')
        
            epBegin = tmp[2]*u.s + 2*u.s + (tmp[0] - tmp[4]) // self.nOrder * self.fpp * self.timeFrame
            epEnd = epBegin + packMax // self.nRow * self.fpp * self.timeFrame
            chLow = self.nSubCh * (od-self.orderOff)
            chHigh = self.nSubCh * (od - self.orderOff + 1) - 1
    
            fpga = np.memmap(fbin, dtype='u1', mode='r')
            v1 = np.zeros((self.nSubCh, self.nAnt, 2), dtype='<i2')
            v1[:, beamIDX, 0] = self.fl2int
            BFMYvec = np.copy(BFMY[0, :, chLow:chLow + self.nSubCh].T * self.fl2int)
            v2 = BFMYvec.ravel().view('<f8').reshape((self.nSubCh, self.nRow, 2)).astype('<i2')
            nChunks = packMax//(int(nSample/self.fpp)*self.nRow)
            for cID in range(nChunks):
                p0 = cID*int(nSample/self.fpp)*self.nRow
                dataOff = self.byteMeta + self.bytePack * p0
                res = np.ones((self.nSubCh, nSample), dtype='<c8')
                self.beamform(res, fpga[dataOff:], v1, v2, nSample, self.scale)
                mask = (np.nanstd(res,0)>0)
                times = -DMdels[chLow:chLow+self.nSubCh][:,np.newaxis]+np.linspace(0,nSample,nSample,endpoint=False)*self.timeFrame
                times += epBegin-ep0T.to_value('unix')*u.s + cID*nSample*self.timeFrame
                phase = np.mod((times/period).value+.5,1)-.5
                for chi in range(self.nSubCh):
                    ch = chi + chLow
                    p,_ = np.histogram(phase[chi][mask],weights=np.abs(res[chi][mask])**2,bins = np.linspace(-.5,.5,nBins+1))
                    c,_ = np.histogram(phase[chi][mask],bins = np.linspace(-.5,.5,nBins+1))
                    profile[ch]+=p
                    counts[ch]+=c
        profile[counts==0]=0
        return(profile/counts)

    def find_overlap(self,files,pad=.01*u.s):
        epEarly = -np.inf*u.s
        epLate = np.inf*u.us
        for i in range(len(files)):
            fbin = files[i]
            try:
                sz = os.path.getsize(fbin)   # follows symlink; raises if target missing
            except (OSError, FileNotFoundError) as e:
                continue
            
            # skip zero-length target files
            if sz == 0:
                continue
            packMax = (sz - self.byteMeta) // self.bytePack
        
            with open(fbin, 'rb') as fh:
                mt = fh.read(64)
                hd = fh.read(64)
                tmp = decHeader2(hd)
                od = tmp[4]
            epBegin = tmp[2]*u.s + 2*u.s + (tmp[0] - tmp[4]) // self.nOrder * self.fpp * self.timeFrame
            epEarly = max(epBegin,epEarly)
            epEnd = epBegin + packMax // self.nRow * self.fpp * self.timeFrame
            epLate = min(epEnd,epLate)
        secWin = ((epLate-epEarly) - 2*pad)/2
        epPulse =  Time((epEarly + pad + secWin),format='unix')
        return(epPulse.isot.replace('T',' '),secWin.to_value(u.s))

    def calibrator_beamform(self,files,beamIDF,find_cal=True,epPulse=None,secWin=None,pad=.01*u.s):
        epPulseMax,secWinMax = self.find_overlap(files,pad)
        if epPulse == None:
            epPulse = epPulseMax
            if secWin == None:
                secWin = secWinMax
            else:
                secWin = min(secWin,secWinMax)
        elif secWin==None:
            secWin = secWinMax - np.abs((Time(epPulseMax)-Time(epPulse)).to_value(u.s))
        else:
            secWin = min(secWin,
                         secWinMax - np.abs((Time(epPulseMax)-Time(epPulse)).to_value(u.s)))
        print(f'epPulse: {epPulse}')
        print(f'secWin: {secWin}')
        ddSpec,startTime,beamIDX,beamIDY,hasData = self.beamform_event(files,beamIDF,
                                                                  0,epPulse,
                                                                  secWin = secWin*u.s,
                                                                  find_cal=find_cal)
        return(ddSpec,startTime,beamIDX,beamIDY)

    def beamform_event_first(self,files,beamIDX,beamIDY,DM,ep0,secWin = 0.03*u.s,timeOffset = 0*u.s,skipOrder=[],verbose=False): 
        hasData = False
        epPulse = (Time(ep0, scale='utc').to_value('unix')*u.s+(4148.8*(u.s*u.MHz**2) * DM * (1/self.freq**2 - 1/self.fRef**2))).to(u.s)
        epPulse+=timeOffset
        
        pixWin = int(secWin * 2 / self.timeFrame)
        pixWin = (pixWin // 4) * 4
        
        ddSpec = np.zeros((self.nChan, pixWin), dtype=np.complex64)
        startTimes = np.zeros(self.nChan)*u.s

        combined_dd_inten = []
        combinedFreq = []      # To store frequencies per file
        winLenList = []  

        
        for i in range(len(files)):
            fbin = files[i]
            try:
                sz = os.path.getsize(fbin)   # follows symlink; raises if target missing
            except (OSError, FileNotFoundError) as e:
                if verbose:
                    print(f"Skipping broken/missing symlink: {fbin}  ({e})")
                continue
            
            # skip zero-length target files
            if sz == 0:
                if verbose:
                    print(f"Skipping (size=0): {fbin}")
                continue
            packMax = (sz - self.byteMeta) // self.bytePack
        
            with open(fbin, 'rb') as fh:
                mt = fh.read(64)
                hd = fh.read(64)
                tmp = decHeader2(hd)
                od = tmp[4]
                if verbose:
                    print(f'\n--- Processing File {i}: {fbin} (order {od}) ---')
            if od in skipOrder:
                continue
            epBegin = tmp[2]*u.s + 2*u.s + (tmp[0] - tmp[4]) // self.nOrder * self.fpp * self.timeFrame
            epEnd = epBegin + packMax // self.nRow * self.fpp * self.timeFrame
            chLow = self.nSubCh * (od-self.orderOff)
            chHigh = self.nSubCh * (od-self.orderOff + 1) - 1
            epLow = epPulse[chLow]
            epHigh = epPulse[chHigh]
    
            # p1, p2 are nearest packet to the requested frame
            ## fix: increase margin to make sure [p1,p2] convers the window
            p1 = int((epHigh - secWin - epBegin) / (self.fpp * self.timeFrame) - 1) * self.nRow
            p2 = int((epLow + secWin - epBegin) / (self.fpp * self.timeFrame) + 1) * self.nRow
            if p1<0 or p2>=packMax:
                continue
            # p1 = max(p1,0)
            # p2 = min(p2,packMax-1)
            if p2<p1:
                continue
            #p1 = int((ep_h - secWin - ep_begin) / (frame_per_pack * timeFrame)) * nRow
            #p2 = int((ep_l + secWin - ep_begin) / (frame_per_pack * timeFrame)) * nRow
            nPack = p2 - p1
            nFrame = int(nPack / self.nRow * self.fpp)
            nPack = nFrame * int(self.nRow / self.fpp)
            p2 = p1 + nPack

            if verbose:
                print(f'File {i}: packMax={packMax}, ep_begin={epBegin:.4f}, ep_end={epEnd:.4f}')
                print(f'File {i}: p1={p1}, p2={p2}, nPack={nPack}, nFrame={nFrame}')
    
            ## nFrame should be multiple of 16
            nPack = int(nFrame / 16) * 16 * 4
            nFrame = nPack / 4
            nSample = int(nFrame)

    
            fpga = np.memmap(fbin, dtype='u1', mode='r')
            p0 = p1
            data_off = self.byteMeta + self.bytePack * p0
            data = fpga[data_off:]
    
            v1 = np.zeros((self.nSubCh, self.nAnt, 2), dtype='<i2')
            v1[:, beamIDX, 0] = self.fl2int
    
            v2 = np.zeros((self.nSubCh, self.nRow, 2),dtype = '<i2')
            v2[:,beamIDY,0]=1

    
            res = np.ones((self.nSubCh, nSample), dtype='<c8')
            self.beamform(res, data, v1, v2, nSample, self.scale)
    
            ep1 = epBegin + (p1//self.nRow)*self.fpp*self.timeFrame
            pixStart = ((epPulse[chLow:(chLow+self.nSubCh)] - secWin - ep1) / self.timeFrame).to_value(u.dimensionless_unscaled).astype(int)
            startTimes[chLow:chLow+self.nSubCh] = ep1
            startTimes[chLow:chLow+self.nSubCh] += pixStart*self.timeFrame
            if verbose:
                print('first pix:', pixStart[-1])
    
            for chi in range(self.nSubCh):
                ch = chi + chLow
                if pixStart[chi]>=0:
                    pp = pixStart[chi]
                    tmp = res[chi, pp:pp+pixWin]
                    max_len = tmp.size
                    ddSpec[ch,:max_len] = tmp[:]
                else:
                    pp = pixStart[chi]
                    if pp+pixWin>0:
                        tmp = res[chi, :pixWin+pp]
                        max_len = tmp.size
                        ddSpec[ch,-pp:-pp+max_len] = tmp[:]
            hasData=True
        return(ddSpec,startTimes,beamIDX,beamIDY,hasData)
        
    def beamform_event_fixed(self,BFM2,files,beamIDF,DM,ep0,secWin = 0.03*u.s,timeOffset = 0*u.s,skipOrder=[],verbose=False): 
        hasData = False
        epPulse = (Time(ep0, scale='utc').to_value('unix')*u.s+(4148.8*(u.s*u.MHz**2) * DM * (1/self.freq**2 - 1/self.fRef**2))).to(u.s)
        epPulse+=timeOffset
        
        beamIDX,sinThetaIDY,beamIDY =  self.beamIDs_from_FUS(beamIDF)

        pixWin = int(secWin * 2 / self.timeFrame)
        pixWin = (pixWin // 4) * 4
        
        ddSpec = np.zeros((self.nChan, pixWin), dtype=np.complex64)
        startTimes = np.zeros(self.nChan)*u.s

        combined_dd_inten = []
        combinedFreq = []      # To store frequencies per file
        winLenList = []  

        
        for i in range(len(files)):
            fbin = files[i]
            try:
                sz = os.path.getsize(fbin)   # follows symlink; raises if target missing
            except (OSError, FileNotFoundError) as e:
                if verbose:
                    print(f"Skipping broken/missing symlink: {fbin}  ({e})")
                continue
            
            # skip zero-length target files
            if sz == 0:
                if verbose:
                    print(f"Skipping (size=0): {fbin}")
                continue
            packMax = (sz - self.byteMeta) // self.bytePack
        
            with open(fbin, 'rb') as fh:
                mt = fh.read(64)
                hd = fh.read(64)
                tmp = decHeader2(hd)
                od = tmp[4]
                if verbose:
                    print(f'\n--- Processing File {i}: {fbin} (order {od}) ---')
            if od in skipOrder:
                continue
            epBegin = tmp[2]*u.s + 2*u.s + (tmp[0] - tmp[4]) // self.nOrder * self.fpp * self.timeFrame
            epEnd = epBegin + packMax // self.nRow * self.fpp * self.timeFrame
            chLow = self.nSubCh * (od-self.orderOff)
            chHigh = self.nSubCh * (od-self.orderOff + 1) - 1
            epLow = epPulse[chLow]
            epHigh = epPulse[chHigh]
    
            # p1, p2 are nearest packet to the requested frame
            ## fix: increase margin to make sure [p1,p2] convers the window
            p1 = int((epHigh - secWin - epBegin) / (self.fpp * self.timeFrame) - 1) * self.nRow
            p2 = int((epLow + secWin - epBegin) / (self.fpp * self.timeFrame) + 1) * self.nRow
            if p1<0 or p2>=packMax:
                continue
            # p1 = max(p1,0)
            # p2 = min(p2,packMax-1)
            if p2<p1:
                continue
            #p1 = int((ep_h - secWin - ep_begin) / (frame_per_pack * timeFrame)) * nRow
            #p2 = int((ep_l + secWin - ep_begin) / (frame_per_pack * timeFrame)) * nRow
            nPack = p2 - p1
            nFrame = int(nPack / self.nRow * self.fpp)
            nPack = nFrame * int(self.nRow / self.fpp)
            p2 = p1 + nPack

            if verbose:
                print(f'File {i}: packMax={packMax}, ep_begin={epBegin:.4f}, ep_end={epEnd:.4f}')
                print(f'File {i}: p1={p1}, p2={p2}, nPack={nPack}, nFrame={nFrame}')
    
            ## nFrame should be multiple of 16
            nPack = int(nFrame / 16) * 16 * 4
            nFrame = nPack / 4
            nSample = int(nFrame)

    
            fpga = np.memmap(fbin, dtype='u1', mode='r')
            p0 = p1
            data_off = self.byteMeta + self.bytePack * p0
            data = fpga[data_off:]
    
            v1 = np.zeros((self.nSubCh, self.nAnt, 2), dtype='<i2')
            v1[:, beamIDX, 0] = self.fl2int
    
            # BFMYvec = BFMY[0, :, chLow:chLow + self.nSubCh].T * self.fl2int
            # v2 = BFMYvec.ravel().view('<f8').reshape((self.nSubCh, self.nRow, 2)).astype('<i2')
            v2 = BFM2[chLow:chLow + self.nSubCh,beamIDY,:,:]
    
            res = np.ones((self.nSubCh, nSample), dtype='<c8')
            self.beamform(res, data, v1, v2, nSample, self.scale)
    
            ep1 = epBegin + (p1//self.nRow)*self.fpp*self.timeFrame
            pixStart = ((epPulse[chLow:(chLow+self.nSubCh)] - secWin - ep1) / self.timeFrame).to_value(u.dimensionless_unscaled).astype(int)
            startTimes[chLow:chLow+self.nSubCh] = ep1
            startTimes[chLow:chLow+self.nSubCh] += pixStart*self.timeFrame
            if verbose:
                print('first pix:', pixStart[-1])
    
            for chi in range(self.nSubCh):
                ch = chi + chLow
                if pixStart[chi]>=0:
                    pp = pixStart[chi]
                    tmp = res[chi, pp:pp+pixWin]
                    max_len = tmp.size
                    ddSpec[ch,:max_len] = tmp[:]
                else:
                    pp = pixStart[chi]
                    if pp+pixWin>0:
                        tmp = res[chi, :pixWin+pp]
                        max_len = tmp.size
                        ddSpec[ch,-pp:-pp+max_len] = tmp[:]
            hasData=True
        return(ddSpec,startTimes,beamIDX,beamIDY,hasData)

    def get_center_time(self,files,DM,verbose=False,offset=0*u.s):
        epCenters = np.array([np.nan for i in files],dtype=np.float64)*u.s
        for i in range(len(files)):
            fbin = files[i]
            try:
                sz = os.path.getsize(fbin)   # follows symlink; raises if target missing
            except (OSError, FileNotFoundError) as e:
                if verbose:
                    print(f"Skipping broken/missing symlink: {fbin}  ({e})")
                continue
            
            # skip zero-length target files
            if sz == 0:
                if verbose:
                    print(f"Skipping (size=0): {fbin}")
                continue
            packMax = (sz - self.byteMeta) // self.bytePack
        
            with open(fbin, 'rb') as fh:
                mt = fh.read(64)
                hd = fh.read(64)
                tmp = decHeader2(hd)
                od = tmp[4]
                if verbose:
                    print(f'\n--- Processing File {i}: {fbin} (order {od}) ---')
            epBegin = tmp[2]*u.s + 2*u.s + (tmp[0] - tmp[4]) // self.nOrder * self.fpp * self.timeFrame
            epEnd = epBegin + packMax // self.nRow * self.fpp * self.timeFrame
            epCenters[(od-self.orderOff)] = (epBegin+(epEnd-epBegin)/2)-(4148.8*(u.s*u.MHz**2) * DM * (1/self.freq[self.nSubCh * (od-self.orderOff)]**2 - 1/self.fRef**2)).to(u.s)
            if verbose:
                print(Time(epBegin.value,format='unix').isot)
                print(Time(epEnd.value,format='unix').isot)
        if verbose:
            print('Order Centers:')
            print(Time(epCenters[np.isfinite(epCenters)].value,format='unix').isot)
        return((Time(np.nanmean(epCenters).value,format='unix')+offset).isot)
        
    @property
    def pos1(self):
        return np.arange(self.nAnt)*self.sepX

    @property
    def pos2(self):
        return np.arange(self.nRow)*self.sepY

    @property
    def lamb(self):
        return (const.c / self.freq).to(u.m)

    @property
    def lamb0(self):
        return (const.c/(400*u.MHz)).to(u.m)
        # return (const.c / self.f0).to(u.m)

    @property
    def freq(self):
        return np.linspace(self.f0.to_value(u.MHz),
                           (self.f0+self.BW).to_value(u.MHz),
                           self.nChan, endpoint=False)*u.MHz

    @property
    def nSubCh(self):
        return int(self.nChan/self.nOrder)

    @property
    def timeFrame(self):
        return (self.nChan/self.BW).to(u.us)

    @property
    def scale(self):
        return 1.0 / (self.fl2int**2)

    @classmethod
    def from_config(cls,config):
        beamform = getattr(bf,config['Format'].get('beamform','beamform256'))
        return(cls(nAnt =config['Geometry'].getint('nAnt',16),
            nRow = config['Geometry'].getint('nRow',16),
            nChan = config['Spectral'].getint('nChan',1024), nOrder = config['Spectral'].getint('nOrder',8),
            BW = config['Spectral'].getfloat('BW',400)*u.MHz, f0 = config['Spectral'].getfloat('f0',300)*u.MHz,
            fRef = config['Spectral'].getfloat('fRef',300)*u.MHz,
            xBeam0 = config['Geometry'].getfloat('xBeam0',-7.5),sepX = config['Geometry'].getfloat('sepX',1.0)*u.m,
            yBeam0 = config['Geometry'].getfloat('yBeam0',-7.5),sepY = config['Geometry'].getfloat('sepY',0.5)*u.m,
            byteMeta = config['Format'].getint('byteMeta',64),bytePack = config['Format'].getint('bytePack',8256),
            orderOff = config['Spectral'].getint('orderOff',0),
            fpp = config['Format'].getint('fpp',4),fl2int = config['Format'].getint('fl2int',256),beamform = beamform,
            calib = config['Format'].get('calFolder','calibration'),
            targetIP = config['General'].get('TargetIP','140.109.178.116'),
            stationID = config['General'].get('stationID','STN'),bwLimit=config['General'].getint('bwLimit',1),
            saveDir=config['General'].get('saveDir','./')
                ))
        
        