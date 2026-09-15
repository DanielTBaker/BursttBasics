import os
import re
import astropy.units as u
import numpy as np
from glob import glob
import astropy.constants as const
from astropy.coordinates import SkyCoord
from multiprocessing import Process, Pipe

import sys
sys.path.insert(0, "/data/kylin/bin/")
try:
    import beamform as bf
except:
    print('No Beamform')
from astropy.time import Time
from configparser import ConfigParser

import warnings
warnings.filterwarnings('ignore')
from .bfutils import *
from ..packet_func import *
from ..burstt_const import *


class Station:
    """ Station Class used to hold information for beamforming and data transfer for a single BURSTT station
    Parameters
    ----------
    nAnt : int, optional
        Number of antennas per row. Defaults to 16
    nRow : int,optional
        Number of rows at station. Defaults to 16
    nOrder : int,optional
        Number of subbands used in saved data at station. Defaults to 8
    BW : astropy.units.Quantity,optional
        Bandwidth of site. Defaults to 400MHz
    f0 : astropy.units.Quantity
        Lowest frequency channel of site. Defaults to 400MHz
    fRef : astropy.units.Quantity

    """
    def __init__(self,nAnt =16,nRow = 16,
                 nChan = 1024, nOrder = 8,
                 BW = 400*u.MHz, f0 = 400*u.MHz,
                 fRef = 400*u.MHz,
                 xBeam0 = -7.5,sepX = 1.0*u.m,
                 yBeam0 = -7.5,sepY = 0.5*u.m,
                 stationAngle = -3*u.deg,
                 byteMeta = 64,bytePack = 8256,
                 orderOff = 0,
                 fpp = 4,fl2int = 256,beamform = None,calib = 'calibration',skipRows = [],
                 targetIP = '140.109.178.116',
                 stationID = 'STN',bwLimit=30,saveDir='./',calFormat='.txt',username = 'ubuntu'
                ):
        
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
        self.stationAngle = stationAngle
    
    
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
        self.calFormat=calFormat
        self.saveDir = saveDir

        self.username = username

        if self.nRow>1:
            self.set_cal_options(calib)
        
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
                try:
                    V = tmp['tau_i'].T  # eigenvector with instrumental delay and weighting
                                        # shape (nRow, nChan), after .T
                    auto = tmp['auto2'] # relative amplitude betweeb rows
                                        # shape (nRow, nChan)
                    phiCorr = np.exp(-1.j * np.angle(V)).reshape(1,self.nRow,self.nChan)
                    normCorr = (np.abs(V)/0.25 / auto).reshape(1,self.nRow,self.nChan)
                except:
                    phiCorr = tmp['phiCorr']
                    normCorr = tmp['normCorr']
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
        
    def loadBatch(self,fh, pack0, npack, bpp, ppf, beam, hdlen=64, bitwidth=4, hdver=1, meta=0, unswap=False):
        '''
        load npack blocks of binary data from the open filehandle
        starting from pack0. each block is of pack_len bytes.
        input:
            pack0: starting packet index (starting from 0)
            npack: number of packets to load
            bpp: bytes per packet (paylod only)
        optional:
            order_off: packet order offset (buggy, keep =0 for now)
            hdlen: header length in bytes
            bitwidth: data format keyword
            hdver: header format keyword (1,2)
            meta: ring buffer or file metadata length in bytes

        output:
            (masked array)
            data0, shape=(npack, bpp)
            clock, shape=(npack,)
            order, shape=(npack,)
        '''

        pack_len = hdlen + bpp
        b0 = pack0 * pack_len + meta
        fh.seek(b0)

        if (bitwidth==4):
            nsamp = bpp
            if (ppf == 2):      # 4bit, bf16, bf64 modes
                grp = 2; nGroup = 256
                nsamp = nGroup*grp # single beam size; nAnt=1
        elif (bitwidth==16):
            nsamp = bpp//4

        data0 = np.ma.array(np.zeros((npack, nsamp), dtype=np.complex64), mask=False)
        clock = np.ma.array(np.zeros(npack, dtype=np.int64), mask=False)
        order = np.ma.array(np.zeros(npack, dtype=int), mask=False)
        for i in range(npack):
            buf = fh.read(pack_len)
            tmp = packetUnpack(buf, bpp, order_off=self.orderOff, hdlen=hdlen, bitwidth=bitwidth, hdver=hdver, unswap=unswap) # tmp=8192byte + a
            if (tmp is None):
                data0.mask[i] = True
                clock.mask[i] = True
                order.mask[i] = True
            else:
                clk, pko, spec = tmp
                antSpec = spec.reshape((nGroup, self.nAnt, grp)).transpose((1, 0, 2)).reshape((self.nAnt, nsamp)) # single beam size; nAnt=1 | beamSpec.shape=(nAnt, nGroup*grp)
                beamSpec = antSpec[beam] # single beam size; nAnt=1 | beamSpec.shape=(nGroup*grp,)
                data0[i] = beamSpec # single beam size; nAnt=1 | beamSpec.shape=(nGroup*grp,)
                clock[i] = clk
                order[i] = pko

        return data0, clock, order

    def formSpec(self,data0, clock, order, ppf, grp=2, bitmap=None, verbose=0):
        '''
        form spectra from the packet series
        input:
            data0, clock, order: as output of loadBatch
                    data0.shap = (npack, bpp)
            ppf: packet per frame (e.g. 2, 4 or 8)
        optional:
            grp: number of consecutive bytes (channels) from the same antenna input 
            bitmap: if provided, should be a boolean array of shape (npack,)
                    True for good packet, False for bad packet
                    also implies the packets have been ordered, contiguous,
                    although some packets may be invalid

                    on the other hand, if the bitmap is None,
                    it assumes all packets are valid, but maybe not contiguous
        
        output:
            data_tick, shape=(ntick,)
            antSpec, shape=(ntick,nAnt,nChan)   where nChan=bpp*ppf/nAnt e.g. 8192*2/16=1024
        '''
        npack, bpp = data0.shape    # e.g. 1000, 8192

        # pre-defined format, ignore input
        if (ppf == 2):      # 4bit, bf16, bf64 modes
            grp = 2; nGroup = 256
            nsamp = nGroup*grp # single beam size; nAnt=1
        elif (ppf == 8):    # 16bit mode
            grp = 1

        ## rearranging packets into frames, taking into account lost packets
        #clock  = np.array(clock)   # already an array

        # raw tick
        tick = clock - order        # packet of the same frame should have the same tick
                                            # i.e. tick is the frame id, delta-tick = ppf

        if (bitmap is None):
            if (verbose>1):
                print('in formSpec: bitmap is None. generate ad hoc one.')
            # new tick
            tick3 = tick - tick[0]              # tick3 starts from 0, increase by ppf (if no packet lost)
            tick3 = tick3 // ppf                # tick3 increment normalized to 1
            ntick = tick3.max() + 1             # e.g. ntick=500 for npack=1000
            ntick = tick3[-1] - tick3[0] + 1
            valid = np.ones(npack, dtype=bool)
            tick0 = tick[0]
        else:
            ntick = npack // ppf
            tmp = np.arange(npack)
            #tmp -= tmp%ppf
            tmp -= order
            tick3 = (tmp//ppf).astype(int)
            valid = bitmap
            if (verbose>1):
                print('in formSpec: npack/nvalid', npack, np.count_nonzero(valid))
            tick0 = -1
            ii = 0
            while (tick0 < 0 and ii<npack):
                if (valid[ii]):
                    tick0 = tick[ii] - ii
                else:
                    ii += 1
            if (tick0 < 0):
                print('error finding valid tick')
                return None
                
        if (verbose>0):
            print('ntick, ppf, bpp:', ntick, ppf, bpp)
            #print('tick3 min,max:', tick3.min(), tick3.max())
        data  = np.ma.array(np.zeros((ntick, ppf, nsamp), dtype=np.complex64), mask=False)
        # data  = np.ma.array(np.zeros((ntick, ppf, nsamp), dtype=np.complex64), mask=True)
        # data  = np.ma.array(np.zeros((ntick, ppf, bpp), dtype=np.complex64), mask=True)
        data_tick = np.arange(ntick) + tick0
        for i in range(npack):
            if (valid[i]):
                if (order.mask[i] == False):
                    it = int(tick3[i])
                    io = int(order[i])
                    #print('debug:', i, it, io)
                    if (io >= ppf):
                        sys.exit('incorrect ppf. abort.')
                    if (it >= 0):
                        data[it, io] = data0[i]
                        data[it, io].mask = False

        pack_flag = data.mask.all(axis=2)
        empty_frame = data.mask.all(axis=(1,2))
        fault_frame = data.mask.any(axis=(1,2))
        empty_pack  = data.mask.all(axis=(0,2))

        if (verbose>1): # debug info
            print('ntick:', ntick)
            print('nempty:', np.count_nonzero(empty_frame)) # frames that are totally empty
            print('nfault:', np.count_nonzero(fault_frame)) # frames that are partially empty

        if (ppf==2 or ppf==1):      # 4bit version
            antSpec = data.reshape((ntick,self.nChan))
            # antSpec = data.reshape((ntick,-1,nAnt,grp)).transpose((0,2,1,3)).reshape((ntick,nAnt,-1))
        elif (ppf==8):    # 16bit version
            evntmp = data[:,0:4].reshape((ntick,-1,self.nAnt)).transpose((0,2,1))
            oddtmp = data[:,4:8].reshape((ntick,-1,self.nAnt)).transpose((0,2,1))
            antSpec = np.ma.array(np.zeros((ntick,self.nAnt,self.nChan),dtype=complex), mask=True)
            antSpec[:,:,0::2] = evntmp
            antSpec[:,:,1::2] = oddtmp

        return data_tick, antSpec


    def loadSpec(self,fh, pack0, npack, beam, bpp=8192, ppf=2, grp=2, hdlen=64, bitmap=None, nBlock=0, verbose=0, bitwidth=4, hdver=1, blocklen=128000, unswap=False):
        '''
        a wrapper of loadBatch + formSpec
        input:
            fh: open binary file handle
            pack0: starting packet index (starting from 0)
            npack: number packets to load
            bpp: bytes per packet (payload length, e.g. 8192)
            ppf: packet per frame (e.g. 2)

        optional:
            nAnt: number of antenna inputs in the packet
            grp: number of consecutive channels (bytes) of the same antenna
            hdlen: header length in bytes
            order_off: (buggy) shift the packet order around, keep =0 for now
            bitmap: boolean array, shape=(npack,); True for a good packet, False for bad
            nBlock: int; number of Block in the open file (fh)
                    if bitmap is None and nBlock>0, the bitmap is automatically loaded
            verbose: int; pass to formSpec
            bitwidth: size of the I/Q data (4 or 16)
                    (automatically determines the grp and ppf)
            hdver: header version (1, 2)
            meta: ring buffer or file metadata length in bytes

        output:
            data_tick, shape=(ntick,)
            antSpec, shape=(ntick, nAnt, nChan)
        '''
        meta = self.byteMeta
        #print('debug: meta', meta)
        if (meta == 64):    # assuming a valid file header
            # override default nBlock and blocklen
            mdict = metaRead(fh)
            blocklen = mdict['packet_number']
            nBlock = mdict['block_number']
            # also override hdver
            hdver = 2
            if (verbose):
                print('nBlock', nBlock, 'blocklen', blocklen)

        if (bitmap is None and nBlock>0):
            BM = loadFullbitmap(fh, nBlock, blocklen=blocklen, meta=meta)
            bitmap = BM[pack0:pack0+npack]

        if (bitmap is not None):
            nvalid = np.count_nonzero(bitmap)
            fvalid = float(nvalid/npack)
            if (verbose>0):
                print('pack0, npack, nvalid:', pack0, npack, nvalid)
            if (nvalid == 0):
                print('no valid packet!')
                return None
        else:
            if (verbose>0):
                print('pack0, npack:', pack0, npack, 'bitmap N/A')

        if (bitwidth==16):
            grp = 1
            ppf = 8

        data0, clock, order = self.loadBatch(fh, pack0, npack, bpp, ppf, beam, hdlen=hdlen, bitwidth=bitwidth, hdver=hdver, meta=meta, unswap=unswap)
        tmp = self.formSpec(data0, clock, order, ppf, grp=grp, bitmap=bitmap, verbose=verbose)

        if (tmp is not None):
            data_tick, antSpec = tmp
        else:
            return None

        return data_tick, antSpec

    #fh = fh
    #start1 = p0
    #sec1_take = nPack
    #args.beam = beamIDx
    #bitmap1 = ...
    # scfg["bitwidth"] = 
    #order_off = self.orderOff
    #hdver = 
    #meta =. self.byteMeta
    # loadSpec(fh, start1, sec1_take, args.beam, bitmap=bitmap1, bitwidth=scfg["bitwidth"], order_off=scfg["order_off"], hdver=scfg["hdver"], verbose=1, meta=meta)

    def single_row_event(self,files,beamIDF,DM,ep0,secWin = 0.03*u.s,timeOffset = 0*u.s,find_cal=True,skipOrder=[],verbose=True):
        hasData = False
        epPulse = (Time(ep0, scale='utc').to_value('unix')*u.s+(4148.8*(u.s*u.MHz**2) * DM * (1/self.freq**2 - 1/self.fRef**2))).to(u.s)
        epPulse+=timeOffset
        beamIDX,sinThetaIDY,beamIDY =  self.beamIDs_from_FUS(beamIDF)
        pixWin = int(secWin * 2 / self.timeFrame)
        pixWin = (pixWin // 4) * 4
        
        ddSpec = np.zeros((self.nChan, pixWin), dtype=np.complex64)
        startTimes = np.zeros(self.nChan)*u.s

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
                epBegin = tmp[2]*u.s + 2*u.s + (tmp[0] - tmp[4]) // self.nOrder * (1./self.fpp) * self.timeFrame
                epEnd = epBegin + packMax // self.nRow *  (1./self.fpp) * self.timeFrame
                chLow = self.nSubCh * (od-self.orderOff)
                chHigh = self.nSubCh * (od-self.orderOff + 1) - 1
                epLow = epPulse[chLow]
                epHigh = epPulse[chHigh]
        
                # p1, p2 are nearest packet to the requested frame
                ## fix: increase margin to make sure [p1,p2] convers the window
                p1 = int((epHigh - secWin - epBegin) / ( (1./self.fpp) * self.timeFrame) - 1) * self.nRow
                p2 = int((epLow + secWin - epBegin) / ( (1./self.fpp) * self.timeFrame) + 1) * self.nRow
                if p1<0 or p2>=packMax:
                    if verbose:
                        print(f'Packet out of range {p1}-{p2} ({packMax})')
                    continue
                # p1 = max(p1,0)
                # p2 = min(p2,packMax-1)
                if p2<p1:
                    print(f'Packet order wrong {p1},{p2}')
                    continue
                #p1 = int((ep_h - secWin - ep_begin) / (frame_per_pack * timeFrame)) * nRow
                #p2 = int((ep_l + secWin - ep_begin) / (frame_per_pack * timeFrame)) * nRow
                nPack = p2 - p1
                nFrame = int(nPack / self.nRow * (1./self.fpp))
                nPack = nFrame * int(self.nRow / (1./self.fpp))
                p2 = p1 + nPack

                if verbose:
                    print(f'File {i}: packMax={packMax}, ep_begin={epBegin:.4f}, ep_end={epEnd:.4f}')
                    print(f'File {i}: p1={p1}, p2={p2}, nPack={nPack}, nFrame={nFrame}')
        
                ## nFrame should be multiple of 16
                nPack = int(nPack / 16) * 16

                bitmap = np.ones(nPack, dtype=bool)
    
                p0 = p1
                tick,res = self.loadSpec(fh, p0, nPack, beamIDX, bitmap=bitmap, bitwidth = 4, hdver=2)
        
                ep1 = epBegin + (p1//self.nRow)*(1./self.fpp)*self.timeFrame
                pixStart = ((epPulse[chLow:(chLow+self.nSubCh)] - secWin - ep1) / self.timeFrame).to_value(u.dimensionless_unscaled).astype(int)
                startTimes[chLow:chLow+self.nSubCh] = ep1
                startTimes[chLow:chLow+self.nSubCh] += pixStart*self.timeFrame
                if verbose:
                    print('first pix:', pixStart[-1])
        
                for chi in range(self.nSubCh):
                    ch = chi + chLow
                    if pixStart[chi]>=0:
                        pp = pixStart[chi]
                        tmp = res[pp:pp+pixWin,chi]
                        max_len = tmp.size
                        ddSpec[ch,:max_len] = tmp[:]
                    else:
                        pp = pixStart[chi]
                        if pp+pixWin>0:
                            tmp = res[:pixWin+pp,chi]
                            max_len = tmp.size
                            ddSpec[ch,-pp:-pp+max_len] = tmp[:]
                hasData=True
        return(ddSpec,startTimes,beamIDX,beamIDY,hasData)
    
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
    
    def beamIDs_from_FUS(self,beamIDF,thetaYOff=0):
        beamIDFX = beamIDF%16
        beamIDFY = beamIDF//16
        Ybeam0F = -7.5 #for nantou ##-7.5 for fushan
        sepYF = 0.5*u.m
        nRowF = 16
        nYBeamF = nRowF
        sinThetaYF = (self.lamb0/sepYF/nRowF*(np.arange(nYBeamF)+Ybeam0F)).to_value(u.dimensionless_unscaled)
        sinThetaIDY = sinThetaYF[beamIDFY]
        thetaYF = np.arcsin(sinThetaIDY) # rad
        if not thetaYOff == 0:
            thetaYF+=(thetaYOff<<u.rad).value
            sinThetaIDY = np.sin(thetaYF)
        return(beamIDFX,sinThetaIDY,beamIDFY)

    def beamform_event(self,files,beamIDF,DM,ep0,secWin = 0.03*u.s,timeOffset = 0*u.s,find_cal=True,skipOrder=[],thetaYOff = 0,verbose=False): 
        if self.nRow == 1:
            return(self.single_row_event(files,beamIDF,DM,ep0,secWin = secWin,timeOffset = timeOffset,find_cal=find_cal,skipOrder=skipOrder,verbose=verbose))
        hasData = False
        epPulse = (Time(ep0, scale='utc').to_value('unix')*u.s+(4148.8*(u.s*u.MHz**2) * DM * (1/self.freq**2 - 1/self.fRef**2))).to(u.s)
        epPulse+=timeOffset
        if find_cal:
            self.find_best_cal(Time(ep0, scale='utc'))
        
        beamIDX,sinThetaIDY,beamIDY =  self.beamIDs_from_FUS(beamIDF,thetaYOff=thetaYOff)
        BFMY = self.get_BFMY(sinThetaIDY,beamIDX,files=files,verbose=verbose)

        pixWin = int(secWin * 2 / self.timeFrame)
        pixWin = (pixWin // 4) * 4
        
        ddSpec = np.zeros((self.nChan, pixWin), dtype=np.complex64)
        startTimes = np.zeros(self.nChan)*u.s

        combined_dd_inten = []
        combinedFreq = []      # To store frequencies per file
        winLenList = []  

        procs = []
        # print(files)
        try:
            for i in range(len(files)):
                # print(f'Create Proc {i}',flush=True)
                procs.append(Reader(i))
            print(len(procs))
            for i,p in enumerate(procs):
                # print(f'Start Proc {i}',flush=True)
                p.start()
            for i,p in enumerate(procs):
                p.single_event([files[i],self.byteMeta,self.bytePack,skipOrder,self.nOrder,self.fpp,
                                        self.timeFrame,self.nRow,self.nSubCh,self.orderOff,epPulse,secWin,
                                        self.nAnt,beamIDX,self.fl2int,BFMY,self.scale,pixWin,self.beamform])
            for p in procs:
                res = p.front_pipe.recv()
                if res[0] == 'Success':
                    hasData=True
                    message,ddSpec2,startTimes2,chLow = res
                    ddSpec[chLow:chLow+self.nSubCh] = ddSpec2
                    startTimes[chLow:chLow+self.nSubCh] = startTimes2
                else:
                    print(res[0])
        finally:
            for p in procs:
                p.stop()
         
        return(ddSpec,startTimes,beamIDX,beamIDY,hasData)

    def beamform_event_onsource(self,files,srceName,DM,ep0,secWin = 0.03*u.s,timeOffset = 0*u.s,find_cal=True,skipOrder=[],thetaYOff = 0,verbose=False,**kwargs): 
            ep0Time = Time(ep0, scale='utc')
            if 'ra' in kwargs and 'dec' in kwargs:
                thEW,thNS = get_source_angles(ep0Time + np.ones(1)*timeOffset,srceName=srceName,
                                                            stationAngle = self.stationAngle,stationLoc = globals()[f'{self.stationID}Center'],ra=kwargs['ra'],dec=kwargs['dec'])
            else:
                thEW,thNS = get_source_angles(ep0Time + np.ones(1)*timeOffset,srceName=srceName,
                                            stationAngle = self.stationAngle,stationLoc = globals()[f'{self.stationID}Center'])
            hasData = False
            epPulse = ep0Time+(4148.8*(u.s*u.MHz**2) * DM * (1/self.freq**2 - 1/self.fRef**2))
            epPulse+=timeOffset
            if find_cal:
                self.find_best_cal(Time(ep0, scale='utc'))
            
            BFMY = self.get_BFMY(np.sin(thNS[0]),0,files=files,verbose=verbose)
            Binv = calc_Binv(freq=self.freq,spacing=self.sepX,beam0=self.xBeam0,f0=400*u.MHz,nAnt=self.nAnt,nBeam=self.nAnt,flipPos=False)
            print(BFMY.shape)
            pixWin = int(secWin * 2 / self.timeFrame)
            pixWin = (pixWin // 4) * 4
            
            ddSpec = np.zeros((self.nChan, pixWin), dtype=np.complex64)
            startTimes = Time(np.zeros(self.nChan),format='unix')
    
            combined_dd_inten = []
            combinedFreq = []      # To store frequencies per file
            winLenList = []  
    
            procs = []
            # print(files)
            try:
                for i in range(len(files)):
                    # print(f'Create Proc {i}',flush=True)
                    procs.append(Reader(i))
                print(len(procs))
                for i,p in enumerate(procs):
                    # print(f'Start Proc {i}',flush=True)
                    p.start()
                for i,p in enumerate(procs):
                    p.redo_first([files[i],self.byteMeta,self.bytePack,skipOrder,self.nOrder,self.fpp,
                                            self.timeFrame,self.nRow,self.nSubCh,self.orderOff,epPulse,secWin,
                                            self.nAnt,thEW,Binv,self.fl2int,BFMY,self.scale,pixWin,self.beamform,self.freq,self.sepX])
                for p in procs:
                    res = p.front_pipe.recv()
                    if res[0] == 'Success':
                        hasData=True
                        message,ddSpec2,startTimes2,chLow = res
                        ddSpec[chLow:chLow+self.nSubCh] = ddSpec2
                        startTimes[chLow:chLow+self.nSubCh] = startTimes2
                    else:
                        print(res[0])
            finally:
                for p in procs:
                    p.stop()
             
            return(ddSpec,startTimes,thEW[0],thNS[0],hasData)
        
    def fold_on_target(self,files,ep0,nSample=20000,period = 1*u.s,DM = 0*u.pc/u.cm**3,nBins = 256,find_cal=True,useSource=True,**kwargs):
        Binv = calc_Binv(freq=self.freq,spacing=self.sepX,beam0=self.xBeam0,f0=400*u.MHz,nAnt=self.nAnt,nBeam=self.nAnt,flipPos=False)
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
            if useSource:
                thEW,thNS = get_source_angles(ep0T+ np.zeros(1)*u.s,kwargs['sourceName'],
                                            stationAngle = self.stationAngle,
                                            stationLoc = globals()[f'{self.stationID}Center'])
                sinThetaIDY = np.sin(thNS).value
        elif 'ra' in kwargs and 'dec' in kwargs:
            srceCoord = SkyCoord(ra = kwargs['ra']<<u.deg,dec=kwargs['dec']<<u.deg)
            beamIDF = determine_beam(srceCoord,ep0T)
            beamIDX,sinThetaIDY,beamIDY =  self.beamIDs_from_FUS(beamIDF)
            if useSource:
                alaz = AltAz(location=globals()[f'{self.stationID}Center'],obstime=ep0T+ np.zeros(1)*u.s)
                srceCoord = srceCoord.transform_to(alaz)
                thEW,thNS = alaz_to_thetas(srceCoord.alt,srceCoord.az,stationAngle = self.stationAngle)
                sinThetaIDY = np.sin(thNS).value
        else:
            print('No Target Specified')
            return(profile)
        BFMY = self.get_BFMY(sinThetaIDY,beamIDX,files=files)
        procs = []
        # print(files)
        profile = np.zeros((self.nChan,nBins))
        try:
            for i in range(len(files)):
                # print(f'Create Proc {i}',flush=True)
                procs.append(Reader(i))
            print(len(procs))
            for i,p in enumerate(procs):
                # print(f'Start Proc {i}',flush=True)
                p.start()
            for i,p in enumerate(procs):
                p.fold_on_source([files[i],self.byteMeta,self.bytePack,[],self.nOrder,self.fpp,
                                        self.timeFrame,self.nRow,self.nSubCh,self.orderOff,ep0T,nSample,nBins,period,
                                        self.nAnt,thEW,Binv,self.fl2int,BFMY,self.scale,self.beamform,self.freq,self.sepX,DM,self.fRef])
            for p in procs:
                res = p.front_pipe.recv()
                if res[0] == 'Success':
                    hasData=True
                    message,foldedSpec,chLow = res
                    profile[chLow:chLow+self.nSubCh]=foldedSpec
                else:
                    print(res[0])
        finally:
            for p in procs:
                p.stop()
        
        return(profile)

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
        try:
            beamform = getattr(bf,config['Format'].get('beamform','beamform256'))
        except:
            beamform=None
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
            saveDir=config['General'].get('saveDir','./'),calFormat=config['Format'].get('calFormat','.txt'),
            username = config['General'].get('username',os.getlogin()),stationAngle = config['Geometry'].getfloat('stationAngle',-3.)*u.deg)
                )
        
        

class Reader(Process):
    def __init__(self,poolID,verbose=False):
        self.poolID = poolID
        self.verbose = verbose

        super().__init__()
        self.front_pipe, self.back_pipe = Pipe()

    def run(self):
        self.active = True
        while self.active:
            self.active = self.listenFront__()

    def listenFront__(self):
        message = self.back_pipe.recv()
        if message[0]== "stop":
            return False
        elif message[0] == "read":
            self.single_event__(message[1])
        elif message[0] == "redo first":
            self.redo_first__(message[1])
        elif message[0] == "fold":
            self.fold_on_source__(message[1])
        else:
            print(f"listenFront__ : unknown message in process {self.poolID} : {message}", flush=True)
        return True

    ## Call in Main
    def stop(self):
        self.front_pipe.send(["stop"])
        self.join()

    def single_event(self,x):
        self.front_pipe.send(["read",x])
    def redo_first(self,x):
            self.front_pipe.send(["redo first",x])
    def fold_on_source(self,x):
        self.front_pipe.send(["fold",x])

    def file_size_check(self,fbin):
        try:
            sz = os.path.getsize(fbin)   # follows symlink; raises if target missing
        except (OSError, FileNotFoundError) as e:
            if self.verbose:
                print(f"Skipping broken/missing symlink: {fbin}  ({e})")
            sz = -1
        return(sz)

    def get_header(self,fbin):
        with open(fbin, 'rb') as fh:
            mt = fh.read(64)
            hd = fh.read(64)
            tmp = decHeader2(hd)
            od = tmp[4]
        return(od,tmp)

    def NS_beam_from_px(self,nAnt,nRow,nOrder,timeFrame,nSubCh,
                    byteMeta,bytePack,fpp,orderOff,fl2int,beamform,scale,
                    fbin,skipOrder,beamIDX,BFMY,frame0,nFrame):
        sz = self.file_size_check(fbin)
        if sz<0:
            self.back_pipe.send(['No File'])
            return
        # skip zero-length target files
        if sz == 0:
            if self.verbose:
                print(f"Skipping (size=0): {fbin}")
            self.back_pipe.send(['File Size=0'])
            return
        packMax = (sz - byteMeta) // bytePack

        od,tmp = self.get_header(fbin)
        if self.verbose:
            print(f'\n--- Processing File {i}: {fbin} (order {od}) ---')
        
        if od in skipOrder:
            self.back_pipe.send(['Order Skipped'])
            return
        epBegin = Time(tmp[2] + 2,format='unix') + (tmp[0] - tmp[4]) // nOrder * fpp * timeFrame
        chLow = nSubCh * (od-orderOff)
        chHigh = nSubCh * (od-orderOff + 1) - 1

        p0 = int(frame0 * nRow / fpp)
        px0 = int(np.round(frame0 - p0*fpp/nRow))
        p1 = int((frame0+nFrame) * nRow / fpp)
        if p1>=packMax:
            return(0,0,0,False)
        nP = (p1-p0)+1
        nSample = int(np.round(nP * fpp / nRow))

        fpga = np.memmap(fbin, dtype='u1', mode='r')
        data_off = byteMeta + bytePack * p0
        data = fpga[data_off:]

        v1 = np.zeros((nSubCh, nAnt, 2), dtype='<i2')
        v1[:, beamIDX, 0] = fl2int

        BFMYvec = BFMY[0, :, chLow:chLow + nSubCh].T * fl2int
        v2 = BFMYvec.ravel().view('<f8').reshape((nSubCh, nRow, 2)).astype('<i2')

        res = np.ones((nSubCh, nSample), dtype='<c8')
        # print(f'Read File {fbin}',flush=True)
        beamform(res, data, v1, v2, nSample, scale)
        ddSpec = np.zeros((nSubCh, nFrame), dtype=np.complex64)
        ddSpec[:,:] = res[:,px0:px0+nFrame]
        return(ddSpec,epBegin+((p0*fpp/nRow)+px0)*timeFrame,chLow,True)


    def NS_beamform(self,nAnt,nRow,nOrder,timeFrame,nSubCh,
                    byteMeta,bytePack,fpp,orderOff,fl2int,beamform,scale,
                    fbin,skipOrder,epPulse,secWin,beamIDX,BFMY,pixWin):
        # print(f'Start File {fbin}',flush=True)
        sz = self.file_size_check(fbin)
        if sz<0:
            self.back_pipe.send(['No File'])
            return
        # skip zero-length target files
        if sz == 0:
            if self.verbose:
                print(f"Skipping (size=0): {fbin}")
            self.back_pipe.send(['File Size=0'])
            return
        packMax = (sz - byteMeta) // bytePack
    
        if not os.access(fbin, os.R_OK):
            self.back_pipe.send(['Cannot Open File'])
            return

        od,tmp = self.get_header(fbin)
        if self.verbose:
            print(f'\n--- Processing File {i}: {fbin} (order {od}) ---')
        
        if od in skipOrder:
            self.back_pipe.send(['Order Skipped'])
            return
        epBegin = Time(tmp[2],format='unix') + 2*u.s + (tmp[0] - tmp[4]) // nOrder * fpp * timeFrame
        epEnd = epBegin + packMax // nRow * fpp * timeFrame
        chLow = nSubCh * (od-orderOff)
        chHigh = nSubCh * (od-orderOff + 1) - 1
        epLow = epPulse[chLow]
        epHigh = epPulse[chHigh]
        
        # p1, p2 are nearest packet to the requested frame
        ## fix: increase margin to make sure [p1,p2] convers the window
        p1 = int((epHigh - secWin - epBegin) / (fpp * timeFrame) - 1) * nRow
        p2 = int((epLow + secWin - epBegin) / (fpp * timeFrame) + 1) * nRow
        if p1<0 or p2>=packMax:
            self.back_pipe.send(['Packet Out of Bounds'])
            return
        # p1 = max(p1,0)
        # p2 = min(p2,packMax-1)
        if p2<p1:
            self.back_pipe.send(['Packet Order Error'])
            return
        #p1 = int((ep_h - secWin - ep_begin) / (frame_per_pack * timeFrame)) * nRow
        #p2 = int((ep_l + secWin - ep_begin) / (frame_per_pack * timeFrame)) * nRow
        nPack = p2 - p1
        nFrame = int(nPack / nRow * fpp)
        nPack = nFrame * int(nRow / fpp)
        p2 = p1 + nPack

        if self.verbose:
            print(f'File {i}: packMax={packMax}, ep_begin={epBegin:.4f}, ep_end={epEnd:.4f}')
            print(f'File {i}: p1={p1}, p2={p2}, nPack={nPack}, nFrame={nFrame}')

        ## nFrame should be multiple of 16
        nPack = int(nFrame / 16) * 16 * 4
        nFrame = nPack / 4
        nSample = int(nFrame)


        fpga = np.memmap(fbin, dtype='u1', mode='r')
        p0 = p1
        data_off = byteMeta + bytePack * p0
        data = fpga[data_off:]

        v1 = np.zeros((nSubCh, nAnt, 2), dtype='<i2')
        v1[:, beamIDX, 0] = fl2int
        BFMYvec = BFMY[0, :, chLow:chLow + nSubCh].T * fl2int
        v2 = BFMYvec.ravel().view('<f8').reshape((nSubCh, nRow, 2)).astype('<i2')

        res = np.ones((nSubCh, nSample), dtype='<c8')
        # print(f'Read File {fbin}',flush=True)
        beamform(res, data, v1, v2, nSample, scale)
        # print(f'Dedisperse File {fbin}',flush=True)

        ep1 = epBegin + (p1//nRow)*fpp*timeFrame
        pixStart = ((epPulse[chLow:(chLow+nSubCh)] - secWin - ep1) / timeFrame).to_value(u.dimensionless_unscaled).astype(int)
        
        startTimes = ep1 + pixStart*timeFrame

        if self.verbose:
            print('first pix:', pixStart[-1])

        ddSpec = np.zeros((nSubCh, pixWin), dtype=np.complex64)

        for chi in range(nSubCh):
            if pixStart[chi]>=0:
                pp = pixStart[chi]
                tmp = res[chi, pp:pp+pixWin]
                max_len = tmp.size
                ddSpec[chi,:max_len] = tmp[:]
            else:
                pp = pixStart[chi]
                if pp+pixWin>0:
                    tmp = res[chi, :pixWin+pp]
                    max_len = tmp.size
                    ddSpec[chi,-pp:-pp+max_len] = tmp[:]
        return(ddSpec,startTimes,chLow)

    def single_event__(self,x):
        fbin,byteMeta,bytePack,skipOrder,nOrder,fpp,timeFrame,nRow,nSubCh,orderOff,epPulse,secWin,nAnt,beamIDX,fl2int,BFMY,scale,pixWin,beamform = x
        ddSpec,startTimes,chLow = self.NS_beamform(nAnt,nRow,nOrder,timeFrame,nSubCh,
                    byteMeta,bytePack,fpp,orderOff,fl2int,beamform,scale,
                    fbin,skipOrder,epPulse,secWin,beamIDX,BFMY,pixWin)
        self.back_pipe.send(['Success',ddSpec,startTimes,chLow])
        return

    def invert_first(self,specs,Binv):
        specs2 = np.einsum('jik,ikl -> jkl',Binv,np.nan_to_num(specs))
        return(specs2)

    def beamform_line(self,specs,cal,theta,freq=np.linspace(300,700,1024,endpoint=False)*u.MHz,spacing=.5*u.m,skipCal=False,flipPos=False):
        pos = np.arange(specs.shape[0]) * spacing
        pos-=pos.mean()
        if flipPos:
            pos*=-1
        lamb = const.c/freq
        th2 = theta << u.rad
        BFMY = np.exp(2.j*np.pi*pos[:,np.newaxis,np.newaxis]*np.sin(th2)[np.newaxis,np.newaxis,:]/lamb[np.newaxis,:,np.newaxis])
        if skipCal:
            final = np.einsum('ijk,ijl->ljk',np.nan_to_num(specs),BFMY)
        else:
            final = specs*cal[:,:,np.newaxis]
            final = np.einsum('ijk,ijl->ljk',np.nan_to_num(final),BFMY)
        return(final)

    def redo_first__(self,x):
        fbin,byteMeta,bytePack,skipOrder,nOrder,fpp,timeFrame,nRow,nSubCh,orderOff,epPulse,secWin,nAnt,thetaX,Binv,fl2int,BFMY,scale,pixWin,beamform,freq,spacing = x
        rowSpec = np.zeros((nAnt,nSubCh, pixWin), dtype=np.complex64)
        for column in range(nAnt):
            rowSpec[column],startTimes,chLow = self.NS_beamform(nAnt,nRow,nOrder,timeFrame,nSubCh,
                        byteMeta,bytePack,fpp,orderOff,fl2int,beamform,scale,
                        fbin,skipOrder,epPulse,secWin,column,BFMY,pixWin)
        rowSpec = self.invert_first(rowSpec,Binv[:,:,chLow:chLow+nSubCh])
        ddSpec = self.beamform_line(rowSpec,None,thetaX,freq[chLow:chLow+nSubCh],skipCal=True,spacing=spacing)
        self.back_pipe.send(['Success',ddSpec,startTimes,chLow])

    def fold_on_source__(self,x):
        fbin,byteMeta,bytePack,skipOrder,nOrder,fpp,timeFrame,nRow,nSubCh,orderOff,epPulse,nFrame,nBins,period,nAnt,thetaX,Binv,fl2int,BFMY,scale,beamform,freq,spacing,DM,fref = x
        sz = self.file_size_check(fbin)
        if sz<0:
            self.back_pipe.send(['No File'])
            return
        # skip zero-length target files
        if sz == 0:
            if self.verbose:
                print(f"Skipping (size=0): {fbin}")
            self.back_pipe.send(['File Size=0'])
            return
        packMax = (sz - byteMeta) // bytePack
        frameMax =  int(packMax*fpp/nRow)
        nChunks = frameMax//nFrame
        foldedSpec = np.zeros((nSubCh,nBins))
        counts = np.zeros((nSubCh,nBins),dtype=int)
        DMt = (DM/K)*(1/freq**2 - 1/fref**2)
        internalPhases = (((np.linspace(0,nFrame,nFrame,endpoint=False)*timeFrame)[np.newaxis,:]-DMt[:,np.newaxis])/period).to_value(u.dimensionless_unscaled)
        # eID = fbin.split('/')[-1].split('.')[1]
        # order = fbin.split('/')[-1].split('.')[3]
        # tempSpec = foldedSpec*0
        # tempCounts = counts*0
        for chunk in range(nChunks):
            rowSpec = np.zeros((nAnt,nSubCh, nFrame), dtype=np.complex64)
            frame0 = nFrame*chunk
            worked = True
            for column in range(nAnt):
                rowSpec[column],startTime,chLow,column_worked = self.NS_beam_from_px(nAnt,nRow,nOrder,timeFrame,nSubCh,
                                                                       byteMeta,bytePack,fpp,orderOff,fl2int,beamform,scale,
                                                                       fbin,skipOrder,column,BFMY,frame0,nFrame)
                worked*=column_worked
                if not worked:
                    break
            if not worked:
                break
            if chunk==0:
                refTime = startTime+0*u.s
            # print(chLow,chunk,(startTime-refTime).to(timeFrame.unit)//timeFrame,flush=True)
            rowSpec = self.invert_first(rowSpec,Binv[:,:,chLow:chLow+nSubCh])
            I = self.beamform_line(rowSpec,None,thetaX,freq[chLow:chLow+nSubCh],skipCal=True,spacing=spacing)[0].value
            I = np.abs(I)**2
            phases = np.mod(((startTime-epPulse)/period).to_value(u.dimensionless_unscaled)+internalPhases[chLow:chLow+nSubCh]+.5,1)-.5
            if np.all((I[:,0]==0)+(~np.isfinite(I[:,0]))) or np.all((I[:,-1]==0)+(~np.isfinite(I[:,-1]))):
                m = np.all((I==0)+(~np.isfinite(I)),axis=0)
                keep = np.argwhere(~m)[:,0]
                if keep.shape[0]==0:
                    continue
                I=I[:,keep.min():keep.max()+1]
                phases=phases[:,keep.min():keep.max()+1]
            # tempCounts*=0
            # tempSpec*=0
            for chan in range(nSubCh):
                s,_ = np.histogram(phases[chan],weights=I[chan],bins = np.linspace(-.5,.5,nBins+1))
                c,_ = np.histogram(phases[chan],bins = np.linspace(-.5,.5,nBins+1))
                foldedSpec[chan]+=s
                counts[chan]+=c
            # np.savez(f'{eID}_{order}_{chunk}.npz',spec = tempSpec, counts = tempCounts,phasesT = phases[0],phasesF = phases[:,0])
        foldedSpec/=counts
        foldedSpec[counts==0]=np.nan
        self.back_pipe.send(['Success',foldedSpec,chLow])

            

            
            
def alaz_to_thetas(alt,az,stationAngle = 3*u.deg):
    """
    Determine the East-West and North-South angles for a given Altitude and Azimuth
    Parameters
    ----------
    alt : astropy.Quantity
        Altitute of target.
    az : String (optional)
        Azimuth of target.
    stationAngle : astropy.Quantity (optional)
        Angle offset of station NS from true NS. Defaults to 3 deg (FUS value)
    Returns
    -------
    thEW : astropy.Quantity
        Angular position of source in beamform East-West direction
    thNS : astropy.Quantity
        Angular position of source in beamform North-South direction
    """
    ## convert into cartesian coordinates
    ## correct Az for rotation of the array relative to NS
    x = np.cos(az+stationAngle)
    y = np.sin(az+stationAngle)
    z = np.sin(alt)
    x *= np.cos(alt)
    y *=np.cos(alt)
    hx = np.sqrt(x**2+z**2)
    hy = np.sqrt(y**2+z**2)
    ## Convert into the two angles
    thNS = np.arctan2(x,hy).to(u.deg)
    thEW = np.arctan2(y,hx).to(u.deg)
    return(thEW,thNS)

def get_source_angles(t,srceName='PSR B0531+21',stationAngle = 3*u.deg,stationLoc = EarthLocation(x=-3035409.9764314634*u.m,y=4937543.332691993*u.m,z=2654856.3500984423*u.m),**kwargs):
    """
    Determine the East-West and North-South angles towards the source from a given station
    Parameters
    ----------
    t : astropy.Time
        Time to calculate source position
    srceName : String (optional)
        Name of source (readable by SkyCoord). Defaults to the Crab Pulsar (PSR B0531+21).
    stationAngle : astropy.Quantity (optional)
        Angle offset of station NS from true NS. Defaults to 3 deg (FUS value)
    stationLoc : EarthLocation (optional)
        Coordinated of the station. Defaults to Fushan
    Returns
    -------
    thEW : astropy.Quantity
        Angular position of source in beamform East-West direction
    thNS : astropy.Quantity
        Angular position of source in beamform North-South direction
    """
    ## Get Alt-Az frame at station
    alaz = AltAz(location=stationLoc,obstime=t)
    ## Convert source into frame
    if 'ra' in kwargs and 'dec' in kwargs:
        print('given')
        srce = SkyCoord(ra=kwargs['ra']<<u.deg,dec=kwargs['dec']<<u.deg,obstime=t).transform_to(alaz)
    else:
        srce = SkyCoord.from_name(srceName).transform_to(alaz)
    ## Convert to NS-EW angles
    thEW,thNS = alaz_to_thetas(srce.alt,srce.az,stationAngle)
    return(thEW,thNS)

def calc_Binv(freq=np.linspace(300,700,1024,endpoint=False)*u.MHz,spacing=1*u.m,beam0=-7.5,f0=400*u.MHz,nAnt=16,nBeam=16,flipPos=False):
    pos = np.arange(nAnt) * spacing
    pos-=pos.mean()
    if flipPos:
        pos*=-1
    lamb = const.c/freq
    lamb0 = const.c/f0
    nChan = freq.shape[0]
    sinTheta = (lamb0 / spacing / nAnt * (np.arange(nBeam)+beam0)).to(u.dimensionless_unscaled)
    BFM0 = np.exp(2.j*np.pi*pos.reshape((1,nAnt,1))/lamb.reshape((1,1,nChan))*sinTheta.reshape((nBeam,1,1)))
    Binv = np.zeros_like(BFM0)
    for ch in range(nChan):
        Binv[:,:,ch] = np.linalg.inv(BFM0[:,:,ch])
    return(Binv)