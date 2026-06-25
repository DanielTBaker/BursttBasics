#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyIGRF: code to synthesise magnetic field values from any generation of the
        International Geomagnetic Reference Field (IGRF),

 @author: Ciaran Beggan (British Geological Survey)
 
 See https://www.ngdc.noaa.gov/IAGA/vmod/ for information on the IGRF
 
 Based on existing codes: igrf13.f (FORTRAN) and chaosmagpy (Python3)
 
 With acknowledgements to: Clemens Kloss (DTU Space), David Kerridge (BGS),
      William Brown and Grace Cox.
 
     This is a program for synthesising geomagnetic field values from the 
     International Geomagnetic Reference Field series of models 
     
    
     The main-field models for 1900.0, 1905.0,..1940.0 and 2025.0 are 
     non-definitive, those for 1945.0, 1950.0,...2020.0 are definitive and
     the secular-variation model for 2025.0 to 2030.0 is non-definitive.

     Main-field models are to degree and order 10 (i.e. 120 coefficients)
     for 1900.0-1995.0 and to 13 (i.e. 195 coefficients) for 2000.0 onwards. 
     The predictive secular-variation model is to degree and order 8 (i.e. 80
     coefficients).

     Older versions of IGRF from generation 1 to 12 can be used. These have 
     differing spatial resolution. 

 Inputs:
 -------
     Inputs are via the command line:    
     
     Enter IGRF version or generation: 
	default IGRF-14

     Options include: 
         Write to (1) screen or (2) filename          
    
    Load in a particular generation of IGRF v1 to v14
    
    Type of computation:
         (1) values at a single locations at one time (spot value)
         (2) values at same location at one year intervals (time series), 
         (3) grid of values at one time (grid); 
     
        Positions can be in:  
         (1) geodetic (WGS-84)
         (2) geocentric coordinates
         
     Latitude & longitude entered as: 
         (1) decimal degrees or 
         (2) degrees & minutes (not in grid)
    
     Altitude can be entered as:
         (1) Geodetic: height above the ellipsoid in km
         (2) Geocentric: distance from the Earth's centre in km
        
     Date: in decimal years (e.g. 2020.25)
 
 Outputs: 
 -----------
      : main field (in nanoTesla) and secular variation (in nanoTesla/year)
 
 
 Dependencies: 
 -------------
     : numpy, scipy
 
 Recent history of code:
 -----------------------
     Initial release: April 2020 (Ciaran Beggan, BGS)
     Corrections in 2020/2021
     Read in IGRFX.SHC files: Jun 2024
     Added IGRF14.SHC file: Nov 2024
 
    
"""
from scipy import interpolate
from . import igrf_utils as iut
from astropy.time import Time
import os
import astropy.units as u

def frac_year(time):
    dt = time.to_datetime()
    t0 = Time(f'{dt.year}-01-01T00:00:00')
    t1 = Time(f'{dt.year+1}-01-01T00:00:00')
    frac = dt.year + (time-t0)/(t1-t0)
    return(frac)

def get_IGRF(time,lat,lon,alt,igrf_gen=14):

    sd = 0
    cd = 0
    itype = 2
    # Load in the file of coefficients
    IGRF_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)),'SHC_files/IGRF' + str(igrf_gen) + '.SHC')
    igrf = iut.load_shcfile(IGRF_FILE, None)
  
    lat, lon = iut.check_lat_lon_bounds(lat.to_value(u.deg),0,lon.to_value(u.deg),0)
    lat*=u.deg
    lon*=u.deg
    colat = 90*u.deg-lat  
    date = frac_year(time)
        
        
    # Interpolate the geomagnetic coefficients to the desired date(s)
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    f = interpolate.interp1d(igrf.time, igrf.coeffs, fill_value='extrapolate')
    coeffs = f(date)    
    
    # Compute the main field B_r, B_theta and B_phi value for the location(s) 
    Br, Bt, Bp = iut.synth_values(coeffs.T, alt.to_value(u.km), colat.to_value(u.deg), lon.to_value(u.deg),
                              igrf.parameters['nmax'])
    
    # For the SV, find the 5 year period in which the date lies and compute
    # the SV within that period. IGRF has constant SV between each 5 year period
    # We don't need to subtract 1900 but it makes it clearer:
    epoch = (date-1900)//5    
    epoch_start = epoch*5
    # Add 1900 back on plus 1 year to account for SV in nT per year (nT/yr):
    coeffs_sv = f(1900+epoch_start+1) - f(1900+epoch_start)   
    Brs, Bts, Bps = iut.synth_values(coeffs_sv.T, alt.to_value(u.km), colat.to_value(u.deg), lon.to_value(u.deg),
                              igrf.parameters['nmax'])
    
    # Use the main field coefficients from the start of each five epoch
    # to compute the SV for Dec, Inc, Hor and Total Field (F) 
    # [Note: these are non-linear components of X, Y and Z so treat separately]
    coeffsm = f(1900+epoch_start);
    Brm, Btm, Bpm = iut.synth_values(coeffsm.T, alt.to_value(u.km), colat.to_value(u.deg), lon.to_value(u.deg),
                              igrf.parameters['nmax'])
    
    
    # Rearrange to X, Y, Z components 
    X = -Bt; Y = Bp; Z = -Br
    # t = X; X = X*cd + Z*sd;  Z = Z*cd - t*sd
    # t = dX; dX = dX*cd + dZ*sd;  dZ = dZ*cd - t*sd
    # t = Xm; Xm = Xm*cd + Zm*sd;  Zm = Zm*cd - t*sd
    return(X*u.nT, Y*u.nT, Z*u.nT)
   
    
