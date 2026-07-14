from pycalc11 import Calc
from astropy.coordinates import SkyCoord
from astropy.time import Time
import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u

def dump_delay(stations,source,mode='center',decOff=0,raOff=0,**kwargs):
    """
    Determine rate and delay of source signal between multiple stations
    """
    srce = SkyCoord(ra=[source.ra+raOff],dec=[source.dec+decOff])
    if 'time' in kwargs:
        epoch=kwargs['time']
    else:
        if isinstance(stations[0].ep0,np.ndarray):
            epoch = Time(Time(stations[0].ep0[0],format='unix').isot)
        else:
            epoch = Time(Time(stations[0].ep0,format='unix').isot)
    coords = []
    for station in stations:
        if hasattr(station,'location'):
            coords.append(station.location)
        else:
            coords.append(eval(f'{station.station}Center'))
    ci = Calc(
            station_names = [station.station for station in stations],
            station_coords=coords,
            source_coords=srce.icrs,
            start_time=epoch-2*u.min,
            duration_min=4,
            d_interval=1,base_mode='baseline'
        )
    ci.run_driver()
    if mode=='center':
        delay = ci.interpolate_delays(epoch).to(u.us)[0,1,0]
    else:
        delay = ci.interpolate_delays(epoch).to(u.us)[0,1,0]
    rates = ci.delay_rate[:,0,1,0]
    interp = interp1d(ci.times.mjd,rates.value)
    rate = interp(epoch.mjd)*rates.unit
    return(delay,rate)