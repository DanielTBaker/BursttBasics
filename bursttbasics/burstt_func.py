from astropy.time import Time
import astropy.units as u
import astropy.constants as const
import numpy as np

from .packet_func import *
from .burstt_const import *
from scipy.special import erfcx
try:
    import emcee
    hasEmcee = True
except:
    hasEmcee = False


def gau1(x,A,x0,s,C):
    """ A simple 1D gaussian.

    Parameters
    ----------
    x : array-like
        x coordinate for gaussian.
    A : float
        Amplitude of gaussian.
    x0 : float
        x coordinate of peak.
    s : float
        Gaussian width.
    C: float
        Background level
    
    Returns
    -------
    gaus : array-like
        Gaussian of the form A e^(-(x-x0)^2/(2s^2))+C
    
    """
    gaus = A*np.exp(-np.power((x-x0)/s,2)/2)+C
    return(gaus)

def exp_gau(x,A,x0,s,t,C):
    return(A*np.exp(np.power(s/t,2)/2-(x-x0)/t)*erfc(((s/t)-((x-x0)/s))/np.sqrt(2))+C)

def power(x,A,gamma):
    """ Power law for fitting

    Parameters
    ----------
    x : array-like
        x coordinate for gaussian.
    A : float
        Amplitude of power law.
    gamma : float
        Power low exponent.

    Returns
    -------
    pwr: array-like

    """
    pwr = A*np.power(x,gamma)
    return(pwr)

def frequency_model_polarized(f,A,n,RM,phi0,pf,N):
    """
    Pulse Spectral profile with RM fluctuations

    Parameters
    ===========
    f - numpy array
        Observing Frequency
    A - float
        Pulse Amplitude
    n - float
        Pulse spectral index
    RM -  float
        Pulse Rotation Measure
    phi0 - float
        RM Phase at 400MHz
    pf - float
        Polarization Fraction
    """
    I0 = A*np.power(f/400,n)
    lam = (const.c/(f*u.MHz)).to(u.m)
    I=I0*(1 + pf*np.cos(2*RM*lam.value**2+2*phi0))
    return(I+N)

def frequency_model_polarized2(f,A,n,n2,RM,phi0,pf,N):
    """
    Pulse Spectral profile with RM fluctuations

    Parameters
    ===========
    f - numpy array
        Observing Frequency
    A - float
        Pulse Amplitude
    n - float
        Pulse spectral index
    RM -  float
        Pulse Rotation Measure
    phi0 - float
        RM Phase at 400MHz
    pf - float
        Polarization Fraction
    """
    I0 = A*np.exp(n*np.log(f/400)+n2*np.log(f/400)**2)
    lam = (const.c/(f*u.MHz)).to(u.m)
    I=I0*(1 + pf*np.cos(2*RM*lam.value**2+2*phi0))
    return(I+N)

def frequency_model(f,A,n,N):
    """
    Pulse Spectral profile with RM fluctuations

    Parameters
    ===========
    f - numpy array
        Observing Frequency
    A - float
        Pulse Amplitude
    n - float
        Pulse spectral index
    RM -  float
        Pulse Rotation Measure
    phi0 - float
        RM Phase at 400MHz
    pf - float
        Polarization Fraction
    """
    I0 = A*np.power(f/400,n)
    return(I0+N)

def calculate_uvw(source, time, s1, s2):
    '''
    Calculates the projected uvw track for the baseline s2-s1.
    Negative u points West, Negative v points South.
    Parameters:
    ----------
    source: astropy.coordinates.SkyCoord object,
            the phase center of the observation
    time:   astropy.time.Time object, the time of the
            observation
    s1:     astropy.coordinates.EarthLocation object,
            station 1
    s2:     astropy.coordinates.EarthLocation object,
            station 2
    Returns:
    -------
    uvw, a list of the u, v and w tracks
    '''
    gw = EarthLocation(lon=0, lat=0)
    b = np.array([
        s2.x.value - s1.x.value, s2.y.value - s1.y.value,
        s2.z.value - s1.z.value
    ]) * u.m
    #     altaz = source.transform_to(AltAz(obstime=time,location=gw))
    H = time.sidereal_time('mean', longitude=gw.lon) - source.ra
    d = source.dec
    trans = np.array(
        [[np.sin(H).value, np.cos(H).value, np.zeros(H.shape)],
         [-np.sin(d).value * np.cos(H).value,
          np.sin(d).value * np.sin(H).value,
          np.cos(d).value*np.ones(H.shape)],
         [np.cos(d).value * np.cos(H).value, -np.cos(d).value * np.sin(H).value,
          np.sin(d).value*np.ones(H.shape)]])
    return np.sum(trans*b[np.newaxis,:,np.newaxis],1)    


def unwrap(y,hatLength=100):
    yOut = np.copy(y)
    hat = np.ones(hatLength)
    comp = np.exp(1j*yOut)
    correction = yOut -np.angle(comp)
    correction-=correction[0]
    comp2 = np.convolve(comp,hat,mode='same')
    smoothedAngs = np.angle(comp2)
    smoothedAngs+=correction
    smoothedDiff = np.diff(smoothedAngs)
    while np.abs(smoothedDiff).max()>np.pi:
        idx = np.argmax(np.abs(smoothedDiff))
        if smoothedDiff[idx]>0:
            smoothedAngs[idx+1:]-=2*np.pi
        else:
            smoothedAngs[idx+1:]+=2*np.pi
        smoothedDiff = np.diff(smoothedAngs)
    
    while (yOut-smoothedAngs).max()>np.pi:
        yOut[(yOut-smoothedAngs)>np.pi] -=2*np.pi
    while (yOut-smoothedAngs).min()<-np.pi:
        yOut[(yOut-smoothedAngs)<-np.pi] +=2*np.pi
    return(yOut)

def fft_axis(x, unit, pad=0):
    """
    Calculates fourier space coordinates from data space coordinates.

    Parameters
    ----------
    x : `~astropy.units.Quantity`
        Coordinate array to find Fourier conjugate of
    unit : `~astropy.units.Unit`
        Desired unit for fourier coordinates
    pad : int, optional
        Integer giving how many additional copies of the data are padded in
        this direction
    """
    fx = (
        np.fft.fftshift(
            np.fft.fftfreq((pad + 1) * x.shape[0], x[1] - x[0]).to_value(unit)
        )
        * unit
    )
    return fx

def ext_find(x, y):
    """
    Determine extent for imshow to center bins at given coordinates

    Parameters
    ----------
    x : `~astropy.units.Quantity`
        X coordinates of data
    y : `~astropy.units.Quantity`
        Y coordiantes of data

    """
    dx = np.diff(x).mean()
    dy = np.diff(y).mean()
    ext = [
        (x[0] - dx / 2).value,
        (x[-1] + dx / 2).value,
        (y[0] - dy / 2).value,
        (y[-1] + dy / 2).value,
    ]
    return ext

def shiftArr(arr,x,xExpected,coher = False):
    if coher:
        Farr = np.fft.fftshift(np.fft.fft(arr,axis=1),axes=1)
        fx = fft_axis(x,1/x.unit)
        x2 = fft_axis(fx,x.unit)
        Farr*=np.exp(1j*2*np.pi*((xExpected[:,np.newaxis]-x[0])*fx[np.newaxis,:]).to(u.dimensionless_unscaled))
        shiftedArr = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Farr,axes=1),axis=1).real,axes=1)
    else:
        shifts = x.shape[0]//2-((xExpected-x[0])//(x[1]-x[0])).astype(int)
        shiftedArr = np.copy(arr)
        for i in range(shiftedArr.shape[0]):
            shiftedArr[i]=np.roll(shiftedArr[i],shifts[i])
        x2 = np.linspace(0,x.shape[0]-1,x.shape[0])*(x[1]-x[0])
        x2 -= x2[x2.shape[0]//2]
    return(shiftedArr,x2)


def format_sci_with_error(val, err,unit=u.dimensionless_unscaled, **kwargs):
    # Determine the exponent of the value
    
    exponent = int(np.floor(np.log10(np.abs(val)))) if val != 0 else 0
    # Scale both values by the exponent
    scaled_val = val / (10 ** exponent)
    scaled_err = err / (10 ** exponent)
    if "decimals" in kwargs:
        decimals = kwargs['decimals']
    else:
        decimals = int(np.floor(np.log10(np.abs(val)))-np.floor(np.log10(err)))
        decimals=max(decimals,0)
        if f'{scaled_err:.{decimals}f}'[-1]=="1":
            decimals+=1
    while np.abs(scaled_val)>=10-5*10**(-decimals-1):
            scaled_val/=10
            scaled_err/=10
            exponent+=1
    if "exponent" in kwargs:
        decimals += kwargs['exponent'] - exponent
        scaled_val*=10**(-kwargs['exponent'] + exponent)
        scaled_err*=10**(-kwargs['exponent'] + exponent)
        exponent = kwargs['exponent']
    # Return formatted string
    decimals=max(decimals,0)
    if exponent == 0:
        return f'{scaled_val:.{decimals}f} ± {scaled_err:.{decimals}f} {unit}'
    else:
        return f'{scaled_val:.{decimals}f} ± {scaled_err:.{decimals}f} x 10^{exponent} {unit}'

def ddm_delay_phase(freq,dDM,tau,phi0):
    fu = freq<<u.MHz
    dDMu = dDM<<(u.pc/u.cm**3)
    delay = (-(dDMu/K)*(1/fu**2) + (tau<<u.ns)).to(u.ns)
    phi = 2*np.pi*((fu*delay)).to_value(u.dimensionless_unscaled)+phi0
    phi -= (2*np.pi*300*u.MHz*(-(dDMu/K)*(1/(300*u.MHz)**2) + (tau<<u.ns)).to(u.ns)).to_value(u.dimensionless_unscaled)
    return(phi)

def flat_fit(freq,dDM,tau,phi0):
    phi = ddm_delay_phase(freq,dDM,tau,phi0)
    model = np.exp(1j*phi)
    model = np.hstack([model.real, model.imag])
    return(model)

def smart_sampler(nwalkers,ndim,logL,pos,args=(),maxLoops=100,checkInterval=100,minBurn=500,vectorize=False,ignoreTau = []):
    oldTau = np.inf
    converged = False

    ## Start sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, logL, args=args,vectorize=vectorize,
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)])
    
    
    state = sampler.run_mcmc(pos, nsteps=checkInterval)
    use = [i for i in range(ndim)]
    for id in ignoreTau:
        use.remove(id)
    for loop in range(maxLoops):
        state = sampler.run_mcmc(state, nsteps=checkInterval)
        tau = sampler.get_autocorr_time(quiet=True)[use]
                
        
        
        converged0 = np.all(tau * 50 + minBurn < sampler.iteration)
        converged1 = np.all(np.abs(oldTau - tau) <= tau*0.01)
        
        if converged1*converged0:
            converged=True
            break
            
        oldTau = tau
    
    # 4. Determine Burn-In Length & Thinning
    # The standard recommendation is to discard 2 * max(tau) as burn-in
    finalTau = sampler.get_autocorr_time(quiet=True)[use]
    burnIn = max(int(2 * np.max(finalTau)),minBurn)
    thinning = int(0.5 * np.min(finalTau))
    lp = np.median(sampler.get_log_prob(discard=burnIn), axis=0)
    good = lp > np.median(lp) - 10
    flat = sampler.get_chain(discard=burnIn)[:, good, :].reshape(-1, ndim)
    return(sampler,flat,converged)

def lnp_phase(d, rho):
    """log p(phase error d | channel S/N rho) for complex Gaussian noise; stable for any rho."""
    c = rho*np.cos(d)/np.sqrt(2)
    small = c < 5
    out = np.empty_like(c)
    # e^{c^2}(1+erf c) = 2e^{c^2} - erfcx(c); evaluate directly where it can't overflow
    cs = c[small]
    out[small] = -rho[small]**2/2 + np.log1p(np.sqrt(np.pi)*cs*erfcx(-cs))
    # large c: the 1 and erfcx terms are negligible -> Gaussian-like limit
    cl, rl, dl = c[~small], rho[~small], d[~small]
    out[~small] = -0.5*(rl*np.sin(dl))**2 + np.log(2*np.sqrt(np.pi)*cl)
    return out - np.log(2*np.pi)

def lnL_rician(theta, f, phi_obs, rho):
    dDM,tau,phiRef = theta
    if not (np.isfinite(dDM) and np.isfinite(tau) and np.isfinite(phiRef)):
        print(f'failed ddM: {dDM}, tau: {tau}, phiRef: {phiRef}')
        return -np.inf
    if np.abs(tau) > 1280 or np.abs(phiRef) > np.pi or np.abs(dDM)>1e-4:
        return(-np.inf)
    return np.sum(lnp_phase(phi_obs - ddm_delay_phase(f, dDM,tau,phiRef),rho))

def rho_from_projection(theta, V, sigC, f, ok):
    # 3) unbiased amplitude estimate per channel (error = sigC)
    A = (V*np.exp(-1j*ddm_delay_phase(f, *theta))).real
    # 4) weighted power-law fit; for fixed alpha, A0 is linear least squares
    x, w, Ao = f[ok]/400, 1/sigC[ok]**2, A[ok]
    best = (np.inf,)
    for al in np.linspace(-6, 2, 201):
        m = x**al
        A0 = np.sum(w*Ao*m)/np.sum(w*m*m)
        chi = np.sum(w*(Ao - A0*m)**2)
        if chi < best[0]:
            best = (chi, al, A0)
    _, al, A0 = best
    # 5) rho per channel
    rho = np.clip(A0*(f/400)**al, 0, None)/sigC
    rho[~ok] = np.nan
    return rho, A0, al


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_dir(x):
    s = sigmoid(x)
    return(s*(1-s))

def sigmoid_dir2(x):
    s = sigmoid(x)
    return(s*(2*s**2-3*s+1))

def sigmoid_inv(s):
    return(-np.log(1./s - 1.))