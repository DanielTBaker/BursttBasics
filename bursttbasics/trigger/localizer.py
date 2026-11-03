from pycalc11 import Calc
from multiprocessing import Process,Pipe
import astropy.units as u
import numpy as np
from astropy.coordinates import Angle,SkyCoord
import os
from .correlator import CorrelationResult
import pandas as pd
from ..burstt_const import *


class ParallelCalc(Process):
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
        elif message[0] == "setup":
            self.setup__(message[1])
        elif message[0] == "run":
            self.run_calc__(message[1])
        elif message[0] == "runL":
                    self.run_linear__(message[1])
        elif message[0] == "runLR":
                    self.run_linear_rot__(message[1])
        elif message[0] == "fitmask":
                    self.set_fitmask__(message[1])
        elif message[0] == "calcDirs":
                    self.calc_dir__(False)
        else:
            print(f"listenFront__ : unknown message in process {self.poolID} : {message}", flush=True)
        return True

    ## Call in Main
    def stop(self):
        self.front_pipe.send(["stop"])
        self.join()

    def setup(self,x):
        self.front_pipe.send(["setup",x])

    def calc_dirs(self):
         self.front_pipe.send(["calcDirs"])

    def set_fitmask(self,x):
            self.front_pipe.send(["fitmask",x])

    def set_fitmask__(self,x):
        try:
            self.fitmask = np.ones((len(self.stations),len(self.sources)),dtype=bool)
            if isinstance(x,dict):
                if 'maskStations' in x:
                    for station in x['maskStations']:
                        if station in self.stations:
                            self.fitmask[self.stations.index(station),:]=False
                if 'maskSources' in x:
                    for source in x['maskSources']:
                        if source in self.sources:
                            self.fitmask[self.sources.index(source),:]=False
                if 'SNRMask' in x:
                    self.fitmask*=self.SNRs>x['SNRMask']
                    self.fitmask*=(self.targetSNR>x['SNRMask'])[:,np.newaxis]
            if hasattr(self,'delays'):
                 self.fitmask*=np.isfinite(self.delays)
            self.back_pipe.send(['success',self.fitmask])
        except Exception as e:
            self.back_pipe.send(['error',e])
        

    def run_calc(self,x):
        self.front_pipe.send(['run',x])
    def run_linear(self,x):
            self.front_pipe.send(['runL',x])
    def run_linear_rot(self,x):
                self.front_pipe.send(['runLR',x])
    def setup__(self,x):
        try:
            # importlib.reload(pycalc11)
            from pycalc11 import Calc
            dataDir = x[0]
            self.eventID = x[1]
            self.targetName = x[2]
            self.sources = list(np.unique(np.array([x.split('_')[2] for x in os.listdir(dataDir) if self.eventID in x and x.endswith('.json')])))
            self.stations = list(np.unique(np.array([x.split('_')[3].removesuffix('.json').split('-') for x in os.listdir(dataDir) if self.eventID in x and x.endswith('.json')])))
            self.stations.remove('FUS')
            self.sources.remove(self.targetName)
        
            sourceCSV = pd.read_csv('sources_updated.csv')
            RAs = [Angle(f'%sh%sm%ss' %tuple(x.split(' '))) for x in sourceCSV['Right ascension'].values]
            decs =  [Angle(f'%sd%sm%ss' %tuple(x.split(' '))) for x in sourceCSV['Declination'].values]
            names = list(sourceCSV['Name'].values)
            self.RAs = [RAs[names.index(n)] for n in self.sources]
            self.decs = [decs[names.index(n)] for n in self.sources]
            self.refRA = RAs[names.index(self.targetName)]
            self.refDec = decs[names.index(self.targetName)]

            self.delays = np.zeros((len(self.stations),len(self.sources)))*u.ns
            self.delaysErr = np.zeros((len(self.stations),len(self.sources)))*u.ns
            self.SNRs = np.zeros((len(self.stations),len(self.sources)))
            self.targetSNR = np.zeros(len(self.stations))
            fRef = np.inf*u.MHz
            for stationID,station in enumerate(self.stations):
                filename = f'{self.eventID}_{self.targetName}_FUS-{station}.json'
                if not os.path.exists(os.path.join(dataDir,filename)):
                    self.delays[stationID,:]*=np.nan
                    self.delaysErr[stationID,:]*=np.nan
                    continue
                targetData = CorrelationResult.from_json(os.path.join(dataDir,filename))
                if targetData.tauRef == None:
                    self.delays[stationID,:]*=np.nan
                    self.delaysErr[stationID,:]*=np.nan
                    continue
                self.targetSNR[stationID] = targetData.SNR
                targetDelay =targetData.tauRef + targetData.coarseDelay
                targetDelay -=  (targetData.dDM/K)*(1/targetData.nuRef**2 - 1/fRef**2)
                targetDelayErr = targetData.tauRefErr
                targetDelaydDMErr = (targetData.dDMErr/K)*(1/targetData.nuRef**2 - 1/fRef**2)
                targetDelayErr = np.sqrt(targetDelayErr**2 + targetDelaydDMErr**2)
                self.targetTime = targetData.tRef
                for sourceID,source in enumerate(self.sources):
                    filename = f'{self.eventID}_{source}_FUS-{station}.json'
                    if not os.path.exists(os.path.join(dataDir,filename)):
                        self.delays[stationID,sourceID]*=np.nan
                        self.delaysErr[stationID,sourceID]*=np.nan
                        continue
                    
                    calData = CorrelationResult.from_json(os.path.join(dataDir,filename))
                    if calData.tauRef == None:
                        self.delays[stationID,sourceID]*=np.nan
                        self.delaysErr[stationID,sourceID]*=np.nan
                        continue
                    calDelay =calData.tauRef + calData.coarseDelay
                    calDelay -=  (calData.dDM/K)*(1/calData.nuRef**2 - 1/fRef**2)
                    calDelay += (self.targetTime - calData.tRef)*calData.modelRate
                    calDelayErr = calData.tauRefErr
                    calDelaydDMErr = (calData.dDMErr/K)*(1/calData.nuRef**2 - 1/fRef**2)
                    calDelayErr = np.sqrt(calDelayErr**2 + calDelaydDMErr**2)
                    self.delays[stationID,sourceID] = calDelay - targetDelay
                    self.delaysErr[stationID,sourceID] = np.sqrt(calDelayErr**2+targetDelayErr**2)
                    self.SNRs[stationID,sourceID] = calData.SNR
            self.fitmask = np.isfinite(self.delays)
            self.back_pipe.send(['complete'])
        except Exception as e:
            self.back_pipe.send(['error',e])

    def calc_dir__(self,silent = True):
        ra  = np.concatenate([(self.refRA + np.array([-1,1,0,0,0])*u.arcsec/np.cos(self.refDec)).to(u.deg), u.Quantity(self.RAs).to(u.deg)])
        dec = np.concatenate([(self.refDec + np.array([0,0,-1,1,0])*u.arcsec).to(u.deg), u.Quantity(self.decs).to(u.deg)])
        ci = Calc(
            station_names = ['FUS']+self.stations,
            station_coords=[FUSCenter]+[eval(f'{station}Center') for station in self.stations],
            source_coords=SkyCoord(ra=ra,dec=dec).icrs,
            start_time=self.targetTime-.5*u.min,
            duration_min=1,
            d_interval=1,base_mode='baseline'
        )
        ci.run_driver()
        model = ci.interpolate_delays(self.targetTime).to(u.ns)[0,1:]
        targetDelays = model[:,:5]
        calDelays =  model[:,5:]
        diffDelays = calDelays[:,np.newaxis,:] - targetDelays[:,:,np.newaxis]
        self.dDeldRA = (diffDelays[:,1,:]-diffDelays[:,0,:])/2
        self.dDeldDec = (diffDelays[:,3,:]-diffDelays[:,2,:])/2
        self.refModel = diffDelays[:,-1,:]
        if not silent:
            self.back_pipe.send(['complete',self.dDeldRA,self.dDeldDec])

    def run_calc__(self,x):
        try:
            dRaas,dDecas = x
            ra  = np.concatenate([(self.refRA + dRaas*u.arcsec/np.cos(self.refDec)).to(u.deg), u.Quantity(self.RAs).to(u.deg)])
            dec = np.concatenate([(self.refDec + dDecas*u.arcsec).to(u.deg), u.Quantity(self.decs).to(u.deg)])
            ci = Calc(
                station_names = ['FUS']+self.stations,
                station_coords=[FUSCenter]+[eval(f'{station}Center') for station in self.stations],
                source_coords=SkyCoord(ra=ra,dec=dec).icrs,
                start_time=self.targetTime-.5*u.min,
                duration_min=1,
                d_interval=1,base_mode='baseline'
            )
            ci.run_driver()
            model = ci.interpolate_delays(self.targetTime).to(u.ns)[0,1:]
            targetDelays = model[:,:dRaas.shape[0]]
            calDelays =  model[:,dRaas.shape[0]:]
            diffDelays = calDelays[:,np.newaxis,:] - targetDelays[:,:,np.newaxis]
            lnL = -.5*np.nansum(self.fitmask[:,np.newaxis,:]*((diffDelays-self.delays[:,np.newaxis,:])/self.delaysErr[:,np.newaxis,:])**2,(0,2))
            self.back_pipe.send(['complete',lnL.to_value(u.dimensionless_unscaled)])
        except:
            self.back_pipe.send(['complete',-np.inf])

    def run_linear__(self,x):
        if not hasattr(self,'dDeldRA'):
            self.calc_dir__()
        diffDelays = self.refModel[:,:,np.newaxis]+x[0]*self.dDeldRA[:,:,np.newaxis]+x[1]*self.dDeldDec[:,:,np.newaxis]
        lnL = -.5*np.nansum(self.fitmask[:,:,np.newaxis]*((diffDelays-self.delays[:,:,np.newaxis])/self.delaysErr[:,:,np.newaxis])**2,(0,1))
        self.back_pipe.send(['complete',lnL.to_value(u.dimensionless_unscaled)])

    def run_linear_rot__(self,x):
            if not hasattr(self,'dDeldRA'):
                self.calc_dir__()
            rot = x[2]
            dRA = np.cos(rot)*x[0]+np.sin(rot)*x[1]
            dDec = np.sin(rot)*x[0]-np.cos(rot)*x[1]
            diffDelays = self.refModel[:,:,np.newaxis]+dRA*self.dDeldRA[:,:,np.newaxis]+dDec*self.dDeldDec[:,:,np.newaxis]
            lnL = -.5*np.nansum(self.fitmask[:,:,np.newaxis]*((diffDelays-self.delays[:,:,np.newaxis])/self.delaysErr[:,:,np.newaxis])**2,(0,1))
            self.back_pipe.send(['complete',lnL.to_value(u.dimensionless_unscaled)])