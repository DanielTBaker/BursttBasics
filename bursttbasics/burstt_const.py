from astropy.coordinates import EarthLocation
import astropy.units as u
import numpy as np
import os

K = 241.0331786*u.GHz**-2*u.cm**-3*u.pc/u.s

def get_offset_vecs(location):
    locationN = EarthLocation(lon=location.lon,
                              lat = location.lat+1*u.mas,
                              height = location.height)
    locationE = EarthLocation(lon=location.lon+1*u.mas,
                              lat = location.lat,
                              height = location.height)
    locationUp = EarthLocation(lon=location.lon,
                              lat = location.lat,
                              height = location.height+1*u.m)
    upVec = np.array([locationUp.x.value-location.x.value,
                      locationUp.y.value-location.y.value,
                      locationUp.z.value-location.z.value,
                     ])
    northVec = np.array([locationN.x.value-location.x.value,
                      locationN.y.value-location.y.value,
                      locationN.z.value-location.z.value,
                     ])
    eastVec = np.array([locationE.x.value-location.x.value,
                      locationE.y.value-location.y.value,
                      locationE.z.value-location.z.value,
                     ])
    northVec/=np.sqrt(np.sum(northVec**2))
    eastVec/=np.sqrt(np.sum(eastVec**2))
    upVec/=np.sqrt(np.sum(upVec**2))
    return(northVec,eastVec,upVec)

def offset_location(location,N,E,up):
    northVec,eastVec,upVec = get_offset_vecs(location)
    xyz = np.array([location.x.to_value(u.m),
           location.y.to_value(u.m),
           location.z.to_value(u.m)
          ])*u.m
    xyz += N*northVec + E*eastVec + up*upVec
    offset = EarthLocation(x=xyz[0],y=xyz[1],z=xyz[2])
    return(offset)
                              

FUSGNSS = EarthLocation(lon = 121*u.deg+34*u.arcmin+53.92087*u.arcsec,
                          lat = 24*u.deg+45*u.arcmin+23.37659*u.arcsec,
                          height = 642.341*u.m)
LTNGNSS = EarthLocation(lon = 120*u.deg+49*u.arcmin+27.85643*u.arcsec,
                          lat = 23*u.deg+42*u.arcmin+52.48231*u.arcsec,
                          height = 882.641*u.m)

## Fushan Offsets
xLocalF = -4.955*u.m
yLocalF = -17.97*u.m
zLocalF = -1.475*u.m
NF = np.cos(3*u.deg)*yLocalF - np.sin(3*u.deg)*xLocalF
EF = np.cos(3*u.deg)*xLocalF + np.sin(3*u.deg)*yLocalF
FUSCenter = offset_location(FUSGNSS,NF,EF,zLocalF)

## Nantou Offsets
xLocalN  = 3.254*u.m
yLocalN = 6.359*u.m
zLocalN = -2.195*u.m
NN = yLocalN
EN = xLocalN
LTNCenter = offset_location(LTNGNSS,NN,EN,zLocalN)
LTNCenter = EarthLocation(x=LTNCenter.x +16.38823593*u.m,
                          y=LTNCenter.y-13.27957413*u.m,
                          z=LTNCenter.z-16.47346749*u.m)

# GRNCenter = EarthLocation(lon = 121.5007*u.deg,lat = 22.6750*u.deg,height = 15.0*u.m)
GRNCenter = EarthLocation(lon = 121*u.deg + 30*u.arcmin+4.29131*u.arcsecond,
                          lat = 22*u.deg + 40*u.arcmin + 32.20767*u.arcsecond,
                          height = 32.677*u.m)

OGWCenter = EarthLocation(lon = 142*u.deg + 13*u.arcmin+0.87471*u.arcsecond,
                          lat = 27*u.deg + 5*u.arcmin + 31.14852*u.arcsecond,
                          height = 32.677*u.m)

beamFile = 'beams_altaz_fus_16x16_b-7.5_-7.5_flim400_800_rot-3.0_sky400x700_ew-60_60_ns-70_70_att30n60_cal251027.npy'
beamBounds = np.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),beamFile), allow_pickle=True).item()