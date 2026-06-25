#!/usr/bin/env python

#-----------------------------------------------------------
# The following is a wrapper that calls several functions
# written in python (and other languages) in order to estimate 
# the RM produced by the Ionosphere for a given date/time and 
# a line of sight.
# @version 1.0
# @author carlos <sotomayor@astro.rub.de>
# Updates of IGRF and angles by Charlotte Sobey 
# The source transits the sky, changing its coordinates of
# latitude and longitude every sec, min, hr, etc. As a 
# consequence of this the coordinates of the intersection 
# of the line of sight direction with thin Ionospheric shell 
# also changes.
#
# This program calculates the coordinates at the piercing 
# point during 24 hr, one point per every hour.
# Each coordinate is then used to calulate the TEC and B values 
# for every hour. Remember that there should be 25 coordinates,
# each one corresponding to every hour during a day (from 00~24).
#
# NOTE: Actually due to problems with the 'sidereal' package we only
# obtain 24 RM values (from 00~23)
#-----------------------------------------------------------

# `path` is the variable describing where the ionFR code is. Determine this
# automatically.
#-----------------------------------------------------------

import sys
import math
from numpy import pi,sin,cos,sign,arcsin,linspace,mod

from . import rdalaz
from .rdalaz import usage
from . import ippcoor_v1 as ippcoor
from . import teccalc
from . import tecrmscalc
from . import ionheight
from ..pyIGRF14.pyIGRFFunc import get_IGRF
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from scipy.interpolate import interp1d

##python ionFRM.py 3h32m59.3s+4d34m45.0s 45d57m20.8sn 78d04m16.9sw 2024-01-01T00:00:00 IGS0OPSFIN_20261260000_01D_02H_GIM.INX 


def ionFRM(source : SkyCoord, site : EarthLocation, time : Time, nameIONEX,igrf_gen=14):
	# Defining some variables for further use
	TECU = 1e16/(u.m**2)
	AltIon = ionheight.calcionheight(nameIONEX)*u.km
	aa = AltAz(location=site, obstime=time)
	AzS = source.transform_to(aa).az
	AlS = source.transform_to(aa).alt
	if AlS<0:
		return(0*u.rad/u.m**2,0*u.rad/u.m**2)
	
	ZenS = 90*u.deg - AlS
	psi = 90*u.deg - AlS - arcsin(cos(AlS)*(1*u.earthRad)/(1*u.earthRad+AltIon))
	latP = arcsin(sin(site.lat)*cos(psi)+cos(site.lat)*sin(psi)*cos(AzS))
	lonP = site.lon +arcsin((sin(psi)*sin(AzS))/cos(latP))
	aa2 = AltAz(location=EarthLocation(lon=lonP,lat=latP,height=AltIon), obstime=time)
	AzP = source.transform_to(aa2).az
	AlP = source.transform_to(aa2).alt
	# AlP = AlS + psi
	ZenP = 90*u.deg - AlP
	
	TECarr = teccalc.calcTEC(latP.to_value(u.deg),lonP.to_value(u.deg),nameIONEX)
	TECInt = interp1d(linspace(0,24,25),TECarr)
	VTEC = TECInt(mod(time.mjd,24))*TECU
	TECpath = VTEC/cos(ZenP)

	RMSTECarr = tecrmscalc.calcRMSTEC(latP.to_value(u.deg),lonP.to_value(u.deg),nameIONEX)*TECU
	VRMSTEC = RMSTECarr[time.datetime.hour]
	RMSTECpath = VRMSTEC/cos(ZenP) # from vertical RMS TEC to line of sight RMS TEC
	Xfield, Yfield, Zfield = get_IGRF(time,latP,lonP,1*u.earthRad + AltIon,igrf_gen=igrf_gen)
	Totfield = Zfield*cos(ZenP) + Yfield*sin(ZenP)*sin(AzP) - Xfield*sin(ZenP)*cos(AzP)
	IFR = ((2.6e-13*u.rad/u.T)*Totfield*TECpath).to(u.rad/u.m**2)
	RMSIFR = ((2.6e-13*u.rad/u.T)*Totfield*RMSTECpath).to(u.rad/u.m**2)
	print('------')
	print(f'Hour: {time.datetime.hour}')
	print(f'Lon: {site.lon.to(u.deg)} | Lat: {site.lat.to(u.deg)}')
	print(f'Alt: {AlS.to(u.deg)} | Az: {AzS.to(u.deg)}')
	print(f'Lon Offset: {(lonP-site.lon).to(u.deg)} | Lat Offset: {(latP-site.lat).to(u.deg)}')
	print(f'Lon P: {lonP.to(u.deg)} | Lat P: {latP.to(u.deg)}')
	print(f'Alt P: {AlP.to(u.deg)} | Az P: {AzP.to(u.deg)}')
	print(f'X: {Xfield.to(u.nT)} | Y: {Yfield.to(u.nT)} | Z: {Zfield.to(u.nT)}')
	print(f'TEC: {TECpath} | B: {Totfield.to(u.nT)}')
	print(f'Ion Height: {AltIon}')
	return(str(time.datetime.hour).zfill(2),TECpath,Totfield.to(u.nT),IFR,RMSIFR)
	
	

	# predict the ionospheric RM for every hour within a day 
	for h in range(24):
		if h < 10:
			rawtime = str(rawDTime.split('T')[0]+'T0'+str(h)+':00:00')
		else:
			rawtime = str(rawDTime.split('T')[0]+'T'+str(h)+':00:00')
		
		hour = rawtime.split('T')[1].split(':')[0]
		date = rawtime.split('T')[0].split('-')
		year = rawtime.split('T')[0].split('-')[0]
		month = rawtime.split('T')[0].split('-')[1]
		day = rawtime.split('T')[0].split('-')[2]

		# RA and Dec (of the source in degrees) to Alt and Az (radians)
		AzS,AlS,HA,LatO,LonO = rdalaz.alaz(rawtime)
		ZenS = (pi/2.0) - AlS

		# output data only when the altitude of the source is above 0 degrees
		if AlS*(180.0/pi) > 0: 

			# Reading the altitude of the Ionosphere in km (from IONEX file)
			AltIon = ionheight.calcionheight(nameIONEX)
			AltIon = AltIon*1000.0 # km to m

			# Alt and AZ coordinates of the Ionospheric piercing point
			# Lon and Lat distances wrt the location of the antenna are also 
			# calculated (radians)
			offLat,offLon,AzPunct,ZenPunct = ippcoor.PuncIonOffset(LatO,AzS,ZenS,AltIon)
			AlSPunct = (pi/2.0) - ZenPunct

			# Calculation of TEC path value for the indicated 'hour' and therefore 
			# at the IPP
			if rawLatitude[-1] == 's':
				if rawLongitude[-1] == 'e':
					TECarr = teccalc.calcTEC(-(LatO + offLat)*180.0/pi,(LonO + offLon)*180.0/pi,nameIONEX)
				if rawLongitude[-1] == 'w':
					TECarr = teccalc.calcTEC(-(LatO + offLat)*180.0/pi,-(LonO + offLon)*180.0/pi,nameIONEX)
			if rawLatitude[-1] == 'n':
				if rawLongitude[-1] == 'e':
					TECarr = teccalc.calcTEC((LatO + offLat)*180.0/pi,(LonO + offLon)*180.0/pi,nameIONEX)
				if rawLongitude[-1] == 'w':
					TECarr = teccalc.calcTEC((LatO + offLat)*180.0/pi,-(LonO + offLon)*180.0/pi,nameIONEX)
			VTEC = TECarr[int(hour)]
			TECpath = VTEC*TEC2m2/math.cos(ZenPunct) # from vertical TEC to line of sight TEC

			# Calculation of RMS TEC path value (same as the step above)
			if rawLatitude[-1] == 's':
				if rawLongitude[-1] == 'e':
					RMSTECarr = tecrmscalc.calcRMSTEC(-(LatO + offLat)*180.0/pi,(LonO + offLon)*180.0/pi,nameIONEX)
				if rawLongitude[-1] == 'w':
					RMSTECarr = tecrmscalc.calcRMSTEC(-(LatO + offLat)*180.0/pi,-(LonO + offLon)*180.0/pi,nameIONEX)
			if rawLatitude[-1] == 'n':
				if rawLongitude[-1] == 'e':
					RMSTECarr = tecrmscalc.calcRMSTEC((LatO + offLat)*180.0/pi,(LonO + offLon)*180.0/pi,nameIONEX)
				if rawLongitude[-1] == 'w':
					RMSTECarr = tecrmscalc.calcRMSTEC((LatO + offLat)*180.0/pi,-(LonO + offLon)*180.0/pi,nameIONEX)
			VRMSTEC = RMSTECarr[int(hour)]
			RMSTECpath = VRMSTEC*TEC2m2/math.cos(ZenPunct) # from vertical RMS TEC to line of sight RMS TEC


			if rawLatitude[-1] == 's':
				if rawLongitude[-1] == 'e':
					Xfield, Yfield, Zfield = get_IGRF(time,-(LatO + offLat)*180.0/pi,(LonO + offLon)*180.0/pi,(EarthRadius+AltIon)/1000.0)
				if rawLongitude[-1] == 'w':
					Xfield, Yfield, Zfield = get_IGRF(time,-(LatO + offLat)*180.0/pi,-(LonO + offLon)*180.0/pi,(EarthRadius+AltIon)/1000.0)
			if rawLatitude[-1] == 'n':
				if rawLongitude[-1] == 'e': 
					Xfield, Yfield, Zfield = get_IGRF(time,(LatO + offLat)*180.0/pi,(LonO + offLon)*180.0/pi,(EarthRadius+AltIon)/1000.0)
				if rawLongitude[-1] == 'w':
					Xfield, Yfield, Zfield = get_IGRF(time,(LatO + offLat)*180.0/pi,-(LonO + offLon)*180.0/pi,(EarthRadius+AltIon)/1000.0)
			Xfield = Xfield*pow(10,-9)*Tesla2Gauss
			Yfield = Yfield*pow(10,-9)*Tesla2Gauss
			Zfield = Zfield*pow(10,-9)*Tesla2Gauss
			Totfield = Zfield*cos(ZenPunct) + Yfield*sin(ZenPunct)*sin(AzPunct) - Xfield*sin(ZenPunct)*cos(AzPunct)

			# Saving the Ionosheric RM and its corresponding
			# rms value to a file for the given 'hour' value
			IFR = 2.6*pow(10,-17)*Totfield*TECpath
			RMSIFR = 2.6*pow(10,-17)*Totfield*RMSTECpath
			return(IFR,RMSIFR)



def ipp_coords(lat,lon,alt,az,h):
	psi = 90*u.deg - alt - np.arcsin(np.cos(alt)*(1*u.earthRad)/(1*u.earthRad+h))
	latP = np.arcsin(np.sin(lat)*np.cos(psi)+np.cos(lat)*np.sin(psi)*np.cos(az))
	lonP = lon +np.arcsin((np.sin(psi)*np.sin(az))/np.cos(latP))
	return(latP.to(u.deg),lonP)