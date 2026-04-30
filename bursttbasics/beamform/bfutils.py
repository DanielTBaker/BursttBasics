import sys
import os
from glob import glob
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import EarthLocation,AltAz
import contextlib
from astropy.stats import sigma_clip
import subprocess

def find_files(eventID,searchDirs):
    matchedFiles = []
    for searchDir in searchDirs:
        pattern = os.path.join(searchDir, f"*{eventID}*")
        match = glob(pattern)
        matchedFiles.extend(match)
    if not matchedFiles:
        decType = False
        try:
            eventT = datetime.strptime(eventID, "%Y%m%d_%H%M%S.%fZ")
            for searchDir in searchDirs:
                pattern = os.path.join(searchDir, f"*{eventID.split('.')[0]}Z*")
                match = glob(pattern)
                matchedFiles.extend(match)
            if len(matchedFiles)>0:
                return(matchedFiles)
        except:
            try:
                eventT = datetime.strptime(eventID, "%Y%m%d_%H%M%SZ")
            except:
                return(matchedFiles)
        print(f"[Info] No exact match, trying forward shifts:")
        for shiftSec in range(1, 6):  # try +1s to +5s
            shiftedDt = eventT + timedelta(seconds=shiftSec)
            shiftedStr = shiftedDt.strftime("%Y%m%d_%H%M%SZ")
            for searchDir in searchDirs:
                pattern = os.path.join(searchDir, f"*{shiftedStr}*")
                match = glob(pattern)
                matchedFiles.extend(match)
            if matchedFiles:
                break
            shiftedStr = shiftedDt.strftime("%Y%m%d_%H%M%S.%fZ")
            for searchDir in searchDirs:
                pattern = os.path.join(searchDir, f"*{shiftedStr}*")
                match = glob(pattern)
                matchedFiles.extend(match)
            if matchedFiles:
                break
        if not matchedFiles:
            eventT = datetime.strptime(f"{eventID.split('.')[0]}Z", "%Y%m%d_%H%M%SZ")
            for shiftSec in range(1, 6):
                shiftedDt = eventT + timedelta(seconds=shiftSec)
                shiftedStr = shiftedDt.strftime("%Y%m%d_%H%M%SZ")
                for searchDir in searchDirs:
                    pattern = os.path.join(searchDir, f"*{shiftedStr}*")
                    match = glob(pattern)
                    matchedFiles.extend(match)
                if matchedFiles:
                    break
    return(matchedFiles)

def determine_beam(srce,time,plot=True):
    codeDir = os.path.dirname(os.path.realpath(__file__))
    fileName = "../beams_altaz_fus_16x16_b-7.5_-7.5_flim400_800_rot-3.0_sky400x700_ew-60_60_ns-70_70_att30n60_cal251027.npy"
    contour_altaz = np.load(os.path.join(codeDir,fileName), allow_pickle=True).item()
    FUS = EarthLocation(lon = 121*u.deg+34*u.arcmin+53.92087*u.arcsec,
                          lat = 24*u.deg+45*u.arcmin+23.37659*u.arcsec,
                          height = 642.341*u.m)
    aaFrame = AltAz(obstime=time,location=FUS)
    aas = srce.transform_to(aaFrame)
    mins = np.array([min(aas.separation(AltAz(alt=contour_altaz[k][:,0]*u.deg,
                             az=contour_altaz[k][:,1]*u.rad,
                             obstime=time,location=FUS))).value for k in contour_altaz.keys()])
    beamID = np.argmin(mins)
    offset = (mins.min()*u.deg).to_value(u.arcsecond)
    print(f'Using beam {beamID} offset by {offset:.1f}"')
    azp = np.mod(contour_altaz[beamID][:,1]-aas.az.to_value(u.rad)+np.pi,2*np.pi)+aas.az.to_value(u.rad)-np.pi
    if plot:
        plt.figure()
        plt.plot(azp,contour_altaz[beamID][:,0]*u.deg)
        plt.plot(aas.az.to_value(u.rad),aas.alt.to_value(u.deg),'*')
        plt.xlabel('Az (rad)')
        plt.ylabel('Alt (deg)')
        plt.show()
    return(beamID)

def in_beam_test(srce,time,findClosest=False,plot=True):
    # print('In Beam Test')
    FUS = EarthLocation(lon = 121*u.deg+34*u.arcmin+53.92087*u.arcsec,
                          lat = 24*u.deg+45*u.arcmin+23.37659*u.arcsec,
                          height = 642.341*u.m)
    aaFrame = AltAz(obstime=time,location=FUS)
    aas = srce.transform_to(aaFrame)
    codeDir = os.path.dirname(os.path.realpath(__file__))
    beamFile = 'beams_altaz_fus_16x16_b-7.5_-7.5_flim400_800_rot-3.0_sky400x700_ew-60_60_ns-70_70_att30n60_cal251027.npy'
    beamBounds = np.load(os.path.join(codeDir,'..',beamFile), allow_pickle=True).item()
    minCross=np.inf
    minBeam = 0
    for k in range(len(beamBounds)):
        offs = beamBounds[k]-np.array([aas.alt.to_value(u.deg),aas.az.to_value(u.rad)])
        offs[:,1]=(np.mod(offs[:,1]+np.pi,2*np.pi)-np.pi)*180/np.pi
        edges = np.diff(offs,axis=0)
        cross = (offs[:-1,0]*edges[:,1]-offs[:-1,1]*edges[:,0])
        if np.all(cross<0):
            # print(f'Beam {k}')
            if plot:
                plt.figure()
                plt.plot(np.mod(beamBounds[k][:,1]-aas.az.to_value(u.rad)+np.pi,2*np.pi)-np.pi+aas.az.to_value(u.rad),
                         beamBounds[k][:,0])
                plt.plot(aas.az.to_value(u.rad),aas.alt.to_value(u.deg),'r*')
                plt.show()
            return(k)
        if cross.max()<minCross:
            minCross = cross.max()
            minBeam = k
    if plot and minCross>0:
        plt.figure()
        plt.plot(np.mod(beamBounds[minBeam][:,1]-aas.az.to_value(u.rad)+np.pi,2*np.pi)-np.pi+aas.az.to_value(u.rad),
                 beamBounds[minBeam][:,0])
        plt.plot(aas.az.to_value(u.rad),aas.alt.to_value(u.deg),'r*')
        plt.show()
    if findClosest:
        return(minBeam)
    return(-1)

def time_from_event(eventID):
    return(f"{eventID[:4]}-{eventID[4:6]}-{eventID[6:8]} {eventID[9:11]}:{eventID[11:13]}:{eventID[13:-1]}")

def plot_spectrum(spec,f,t,out=contextlib.nullcontext(),rbt=400,rbf=32,onWidth=5,**kwargs):
    with out:
        I = np.abs(spec)**2
        I[np.abs(f-370*u.MHz)<10*u.MHz]=np.nan
        I/=np.nanmean(I,1)[:,np.newaxis]
        v = I.std(1)
        v2 = sigma_clip(v)
        I[v2.mask]=np.nan
        I2 = np.nanmean(np.reshape(I[:,:rbt*(I.shape[1]//rbt)],(1024,-1,rbt)),2)
        t2 = np.reshape(t[:rbt*I2.shape[1]],(-1,rbt)).mean(1)
        f2 = np.reshape(f,(-1,rbf)).mean(1)
        SNR = np.nansum(I2,0)
        on = np.abs(t2-t2[np.argmax(SNR)])<=(onWidth+.1)*(t2[1]-t2[0])
        SNR -= np.mean(SNR[np.invert(on)])
        SNR/=np.nanstd(SNR[np.invert(on)])
        wts = np.copy(SNR)
        wts[np.invert(on)]=0
        wts/=np.sum(wts)
        SF = np.nansum(I2*wts,1)
        NF = np.nansum(I2*np.roll(wts,wts.shape[0]//2),1)
        SNRF = (SF-NF)/NF
        I2 = np.nanmean(np.reshape(I2,(1024//rbf,rbf,-1)),1)
        dt = t2[1]-t2[0]
        df = f2[1]-f2[0]
        ext = [(t2[0]-dt/2).value,(t2[-1]+dt/2).value,
               (f2[0]-df/2).value,(f2[-1]+df/2).value,
              ]
        grid=plt.GridSpec(nrows=2,ncols=2,height_ratios=[1,4],width_ratios=[4,1],wspace=0,hspace=0)
        plt.figure(figsize=(8,8))
        plt.subplot(grid[1,0])
        plt.imshow((I2-1)/np.nanstd(I2),origin='lower',aspect='auto',vmin=-2,vmax=4,extent=ext,cmap='cividis')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (MHz)')
        plt.subplot(grid[0,0])
        plt.plot(t2,SNR,'k')
        plt.axhline(0,ls='--',c='k',alpha=.5)
        plt.text(t2.min().value,SNR.max()*.9,f'SNR = {SNR.max():.2f}')
        plt.axvspan(t2[on].min().value,t2[on].max().value, alpha=0.5, color='green')
        plt.axhspan(-1,1,alpha=.5,color='grey')
        plt.xlim(ext[:2])
        plt.xticks([])
        plt.ylabel('SNR')
        plt.subplot(grid[1,1])
        plt.plot(SNRF,f,'k')
        plt.ylim(ext[2:])
        plt.yticks([])
        plt.axvline(0,ls='--',c='k',alpha=.5)
        if 'title' in kwargs:
            plt.suptitle(kwargs['title'])
            plt.tight_layout()
        plt.show()
    return(SNR.max())

def transfer_data(transfer,station):
    for source in transfer.keys():
        if len(transfer[source])>0:
            bwLimit = f'{station.bwLimit}M'
            targetIP = station.targetIP
            targetDir = f'/burstt_home/shared/dbaker/{source.lower()}'
            destination = f'ubuntu@{targetIP}:{targetDir}'
            command = ["rsync", "-avh", '--info=progress2', '--remove-source-files'] + transfer[source] + [destination, f'--bwlimit={bwLimit}']
            print(f'Start transfer of {len(transfer[source])} {source} files')
            process = subprocess.Popen(command,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        text=True,bufsize=1)
            try:
                # iter(lambda: ...) reads the output character by character or line by line
                for line in iter(process.stdout.readline, ""):
                    # Display the raw progress line from rsync
                    if '%' in line:
                        sys.stdout.write(f"\r{line.strip()}"+"\x1b[K")
                        sys.stdout.flush()
            finally:
                process.stdout.close()
                return_code = process.wait()
                if return_code == 0:
                    print("\nTransfer Complete.")
                else:
                    print(f"\nError: Rsync exited with code {return_code}")