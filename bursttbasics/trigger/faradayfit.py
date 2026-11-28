import numpy as np
import fitburst as fb
from fitburst.analysis.model import SpectrumModeler
import fitburst.routines.derivative as deriv
import fitburst.routines as rt
from fitburst.backend import general
import fitburst.utilities as ut
from fitburst.utilities import bases
import astropy.units as u

from bursttbasics.burstt_func import sigmoid,sigmoid_dir,sigmoid_dir2

import fitburst.utilities.bases as _fb_bases
import fitburst.analysis.fitter as _fb_fitter
import fitburst.analysis.model as _fb_model
from scipy.optimize import least_squares

_FB_MODULES = (_fb_bases, _fb_fitter, _fb_model)

def fitburst_quiet(quiet=True):
    """Silence (or restore) fitburst's print output: NaN warnings, 'bad data value', INFO lines."""
    for m in _FB_MODULES:
        if quiet:
            m.print = lambda *a, **k: None
        elif 'print' in vars(m):
            del m.print

fitburst_quiet(True) 



class BURSTTDataReader(bases.ReaderBaseClass):
    def __init__(self,data, burstParameters, rbt = 1,fCenter=False):
        self.burst_parameters = {}
        for k, v in burstParameters.items():
            if not isinstance(v, list):
                self.burst_parameters.update({k :[v]})
            else:
                self.burst_parameters.update({k : v})
        self.data_full = np.abs(data.spec)**2
        self.num_freq = self.data_full.shape[0]
        self.data_full = np.reshape(self.data_full[:,:rbt*(self.data_full.shape[1]//rbt)],(self.num_freq,-1,rbt)).mean(2)
        self.num_time = self.data_full.shape[1]
        self.data_weights = np.ones((self.num_freq, self.num_time), dtype=float)
        self.res_time = rbt*(data.time[1]-data.time[0]).to_value(u.s)
        self.times = np.linspace(0,self.num_time,self.num_time,endpoint=False)*self.res_time
        self.res_freq = (data.freq[1]-data.freq[0]).to_value(u.MHz)
        self.freqs = data.freq.to_value(u.MHz)
        if fCenter:
            self.freqs+=self.res_freq/2
        self.is_dedispersed = True
        self.times_bin0 = 0.0

class FaradaySpectrumModeler(SpectrumModeler):
    """
    A Python structure that contains all information regarding parameters that
    describe and are used to compute models of dynamic spectra.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, freqs: float, times: float, dm_incoherent: float = 0.,
        factor_freq_upsample: int = 1, factor_time_upsample: int = 1, num_components: int = 1,
        is_dedispersed: bool = False, is_folded: bool = False, scintillation: bool = False,
        verbose: bool = False) -> None:
        """
        Instantiates the model object and sets relevant parameters, depending on
        desired model for spectral energy distribution.

        Parameters
        ----------

        freqs: float
            The frequency channels in the spectrum (including masked ones)

        times: float
            The time samples in the spectrum

        dm_incoherent : float, optional
            The DM used to incoherently dedisperse input data; this is only used if the
            'is_dedispersed' argument is set to True

        factor_freq_upsample : int, optional
            The factor to upsample each frequency label into an array of subbands

        factor_time_upsample : int, optional
            The factor to upsample the array of timestamps

        is_dedispersed : bool, optional
            If true, then assume that the dispersion measure is an 'offset' parameter
            and computes the relative dispersion for non-zero offset values

        is_folded : bool, optional
            If true, then the temporal profile is computed over two realizations and then
            averaged down to one (in order to allow for wrapping of a folded pulse shape)

        num_components : int, optional
            The number of distinct burst components in the model

        scintillation : bool, optional
            if true, the compute per-channel amplitudes using input data.

        verbose : bool, optional
            If true, then print parameter values during each function call
            (This is mainly useful to gauge least-squares fitting algorithms.)

        """

        # pylint: disable=too-many-arguments,too-many-locals

        # first define model-configuration parameters that are not fittable.
        self.dm_incoherent = dm_incoherent
        self.factor_freq_upsample = factor_freq_upsample
        self.factor_time_upsample = factor_time_upsample
        self.freqs = freqs
        self.is_dedispersed = is_dedispersed
        self.is_folded = is_folded
        self.num_components = num_components
        self.scintillation = scintillation
        self.times = times
        self.verbose = verbose

        # derive some additional data needed for downstream fitting.
        self.num_freq = len(self.freqs)
        self.num_time = self.times.shape[-1]
        self.res_freq = np.fabs(self.freqs[1] - self.freqs[0])
        self.res_time = np.fabs(self.times[..., 1] - self.times[..., 0])

        # determine if the times array is frequency dependent
        if self.times.ndim == 1:
            self.has_freq_dependent_times = False

        elif self.times.ndim == 2:
            if self.times.shape[0] != self.num_freq:
                raise RuntimeError("times array has more than one dimension, "
                                   "but first dimension does not match frequency axis.")

            self.has_freq_dependent_times = True

        else:
            raise RuntimeError("Do not recognize shape of times array. "
                               "Must be either (ntime,) or (nfreq, ntime).")

        # define all *fittable* model parameters first.
        # NOTE: 'ref_freq' is not listed here as it's a parameter that is always held fixed.
        self.parameters = [
            "amplitude",
            "arrival_time",
            "burst_width",
            "dm",
            "dm_index",
            "scattering_timescale",
            "scattering_index",
            "spectral_index",
            "spectral_running",
            'rm',
            'phi0',
            "polarization_fraction"
        ]

        # now instantiate parameter attributes and set initially to NoneType.
        for current_parameter in self.parameters:
            setattr(self, current_parameter, None)

        # now instantiate the structures for per-component models, time differences, and temporal profiles.
        # (the following are used for computing derivatives and/or per-channel amplitudes.)
        self.amplitude_per_component = np.zeros(
            (self.num_freq, self.num_time, self.num_components), dtype=float
        )

        self.spectrum_per_component = np.zeros(
            (self.num_freq, self.num_time, self.num_components), dtype=float
        )

        self.timediff_per_component = np.zeros(
            (self.num_freq, self.num_time, self.num_components), dtype=float
        )

        self.timeprof_per_component = np.zeros(
            (self.num_freq, self.num_time, self.num_components), dtype=float
        )

        self.faraday_free_per_component = np.zeros_like(self.spectrum_per_component)

    def compute_model(self, data: float = None) -> float:
        """
        Computes the model dynamic spectrum based on model parameters (set as class
        attributes) and input values of times and frequencies.

        Parameters
        ----------
        data : float, optional
            an array of time_averaged data values to be used for per-channel normalization
        """

        # pylint: disable=no-member,too-many-locals

        # if scintillation modeling is desired but data aren't provide, then exit.
        if self.scintillation and data is None:
            sys.exit("ERROR: scintillation modelling is desired by data are missing!")

        # initialize model matrix and size of temporal window.
        num_window_bins = self.num_time // 2

        # loop over all components.
        for current_freq in range(self.num_freq):

            if self.has_freq_dependent_times:
                times = self.times[current_freq]
                res_time = self.res_time[current_freq]
            else:
                times = self.times
                res_time = self.res_time

            # now loop over bandpass.
            for current_component in range(self.num_components):

                # extract parameter values for current component.
                current_amplitude = self.amplitude[current_component]
                current_arrival_time = self.arrival_time[current_component]
                current_dm = self.dm[0]
                current_dm_index = self.dm_index[0]
                current_ref_freq = self.ref_freq[current_component]
                current_sc_idx = self.scattering_index[0]
                current_sc_time = self.scattering_timescale[0]
                current_sp_idx = self.spectral_index[current_component]
                current_sp_run = self.spectral_running[current_component]
                current_width = self.burst_width[current_component]
                current_rm = self.rm[0]
                current_phi0 = self.phi0[current_component]
                current_pol_frac = self.polarization_fraction[current_component]
                

                if self.verbose and current_freq == 0:
                    if self.scintillation:
                         print(
                            f"{current_dm:.5f} {current_arrival_time:.5f} ",
                            f"{current_sc_idx:.5f}  {current_sc_time:.5f}  {current_width:.5f}", end=" ")
                    else:
                         print(
                            f"{current_dm:.5f}  {current_amplitude:.5f}  {current_arrival_time:.5f}  ",
                            f"{current_sc_idx:.5f}  {current_sc_time:.5f}  {current_width:.5f}", end=" ")

                # create an upsampled version of the current frequency label.
                # even if no upsampling is desired, this will return an array
                # of length 1.
                current_freq_arr = rt.manipulate.upsample_1d(
                    [self.freqs[current_freq]],
                    self.res_freq,
                    self.factor_freq_upsample
                )

                # create an upsampled version of the times label
                current_times = rt.manipulate.upsample_1d(
                    times,
                    res_time,
                    self.factor_time_upsample
                )

                # first, compute arrival time for all upsampled frequency labels.
                arrival_time_vs_freq = current_arrival_time + rt.ism.compute_time_dm_delay(
                    self.dm_incoherent + current_dm,
                    general["constants"]["dispersion"],
                    current_dm_index,
                    current_freq_arr,
                    freq2=current_ref_freq,
                )

                if not self.has_freq_dependent_times:

                    # if the time axis is shared across all freqs,
                    # incoherent dedispersion has already aligned to
                    # `ref_freq`. Remove that alignment here so the
                    # residual DM term is applied consistently per channel.
                    arrival_time_vs_freq -= rt.ism.compute_time_dm_delay(
                        self.dm_incoherent,
                        general["constants"]["dispersion"],
                        current_dm_index,
                        self.freqs[current_freq],
                        freq2=current_ref_freq,
                    )

                # calculate time relative to the arrival time at each frequency
                current_times_arr = current_times - arrival_time_vs_freq[:, None]

                # before proceeding, compute and save the per-component time difference map.
                self.timediff_per_component[current_freq, :, current_component] = \
                    rt.manipulate.downsample_1d(
                        current_times_arr.mean(axis=0),
                        self.factor_time_upsample
                    )

                # next, compute and store raw temporal profile.
                current_profile = self.compute_profile(
                    current_times_arr,
                    0.0, # since 'current_times' is already corrected for DM.
                    current_sc_time,
                    current_sc_idx,
                    current_width,
                    current_freq_arr[:, None],
                    current_ref_freq,
                    is_folded = self.is_folded,
                )

                self.timeprof_per_component[current_freq, :, current_component] = rt.manipulate.downsample_1d(
                    current_profile.mean(axis=0),
                    self.factor_time_upsample
                )

                # next, compute and scale profile by the spectral energy distribution.
                current_profile *= rt.spectrum.compute_spectrum_rpl(
                    current_freq_arr,
                    current_ref_freq,
                    current_sp_idx,
                    current_sp_run,
                )[:, None]

                current_profile_faraday_free = current_profile.copy()

                current_profile *= (1+ sigmoid(current_pol_frac) *np.cos(2*current_rm*(299792458.0**2)/(current_freq_arr*1e6)**2 + 2*current_phi0))[:,None]

                # before writing, downsize upsampled array to original size.
                current_profile = rt.manipulate.downsample_1d(
                    current_profile.mean(axis=0),
                    self.factor_time_upsample
                )

                # before exiting the loop, save different snapshots of the model.
                self.amplitude_per_component[current_freq, :, current_component] = rt.spectrum.compute_spectrum_rpl(
                    self.freqs[current_freq],
                    current_ref_freq,
                    current_sp_idx,
                    current_sp_run
                ) * (10 ** current_amplitude) * (1+ sigmoid(current_pol_frac) *np.cos(2*current_rm*(299792458**2)/(self.freqs[current_freq]*1e6)**2 + 2*current_phi0))
                self.spectrum_per_component[current_freq, :, current_component] = (10 ** current_amplitude) * current_profile
                self.faraday_free_per_component[current_freq, :, current_component] = (10 ** current_amplitude) * rt.manipulate.downsample_1d(current_profile_faraday_free.mean(axis=0), self.factor_time_upsample)

                # print spectral index/running for current component.
                if current_freq == 0:
                    if self.verbose and not self.scintillation:
                        print(f"{current_sp_idx:.5f}  {current_sp_run:.5f}")

                    else:
                        print()

        # if desired, then compute per-channel amplitudes in cases where scintillation is significant.
        if self.scintillation:

            for freq in range(self.num_freq):
                current_amplitudes = rt.ism.compute_amplitude_per_channel(
                    data[freq], self.timeprof_per_component[freq, :, :]
                )
                # now compute model with per-channel amplitudes determined.
                for component in range(self.num_components):
                    current_profile = self.timeprof_per_component[freq, :, component]
                    self.amplitude_per_component[freq, :, component] = current_amplitudes[component]
                    self.spectrum_per_component[freq, :, component] = current_amplitudes[component] * current_profile

        return np.sum(self.spectrum_per_component, axis=2)

class ExtendedFitter(fb.analysis.fitter.LSFitter):
    def __init__(self, data: float, model: object, good_freq: bool, weighted_fit: bool = True, 
                 weight_range: list = None, x_scale: str = 'jac'):
        super().__init__(data,model,good_freq,weighted_fit,weight_range)
        self.x_scale = x_scale

    def fit(self, exact_jacobian: bool = True) -> None:
        """
        Executes least-squares fitting of the model spectrum to data, and stores
        results to child class.

        Parameters
        ----------
        exact_jacobian : bool, optional
            If set, then use the exact formulation of the Jacobian and supply it 
            as method for the solver to use.

        Returns
        -------
        None
            A dictionary attribute is defined that contains the best-fit results from
            the scipy.optimize.least_aquares solver, as well as estimates of the
            covariance matrix and parameter uncertainties.
        """

        # pylint: disable=broad-except

        # convert loaded parameter dictionary/entries into a list for scipy object.
        parameter_list = self.get_fit_parameters_list()

        # if desired, use exact formulation of Jacobian.
        jac = "2-point"

        if exact_jacobian:
            jac = self.compute_jacobian

        # do fit!
        try:
            results = least_squares(
                self.compute_residuals, 
                parameter_list,
                jac = jac,
                x_scale = self.x_scale
            )

            self.results = results

            if self.results.success:
                print("INFO: fit successful!")

            else:
                print("INFO: fit didn't work!")

            # now try computing uncertainties and fit statistics.
            self._compute_fit_statistics(self.data, results)

            if self.success:
                print("INFO: derived uncertainties and fit statistics")

        except Exception as exc:
            print("ERROR: solver encountered a failure! Debug!")
            print(_fb_fitter.traceback.format_exc())

def _faraday_terms(parameters, model, component):
    lam2 = 299792458.0**2 / (model.freqs * 1e6)**2                        # (nfreq,)
    rm   = parameters["rm"][0]                                # global
    p    = parameters["polarization_fraction"][component]
    phi0 = parameters["phi0"][component]
    theta = 2*rm*lam2 + 2*phi0
    M0 = model.faraday_free_per_component[:, :, component]    # (nfreq, ntime), no Faraday factor
    return lam2, p, theta, M0

def deriv_model_wrt_rm(parameters: dict, model: object, component: int = 0) -> float:
    """dM/dRM = sum over components of -M0_c * p_c * sin(theta_c) * 2 lam^2  (rm is global)."""
    out = 0.
    for c in range(model.num_components):
        lam2, p, theta, M0 = _faraday_terms(parameters, model, c)
        out = out + M0 * (-2 * sigmoid(p) * lam2 * np.sin(theta))[:, None]
    return out

def deriv_model_wrt_phi0(parameters: dict, model: object, component: int = 0) -> float:
    """dM/dRM = sum over components of -M0_c * p_c * sin(theta_c) * 2 lam^2  (rm is global)."""
    lam2, p, theta, M0 = _faraday_terms(parameters, model, component)
    return M0 * (-2 * sigmoid(p) * np.sin(theta))[:, None]

def deriv_model_wrt_polarization_fraction(parameters: dict, model: object, component: int = 0) -> float:
    """dM/dp = M0 * cos(theta)  (per component)."""
    lam2, p, theta, M0 = _faraday_terms(parameters, model, component)
    return M0 * (sigmoid_dir(p) * np.cos(theta))[:, None]


def deriv2_model_wrt_rm_rm(parameters: dict, model: object, component: int = 0) -> float:
    out = 0.
    for c in range(model.num_components):
        lam2, p, theta, M0 = _faraday_terms(parameters, model, c)
        out = out + M0 * (-4 * sigmoid(p) * lam2**2 * np.cos(theta))[:, None]
    return out

def deriv2_model_wrt_rm_phi0(parameters: dict, model: object, component: int = 0) -> float:
    lam2, p, theta, M0 = _faraday_terms(parameters, model, component)
    return M0 * (-4 * sigmoid(p) * lam2 * np.cos(theta))[:, None]

def deriv2_model_wrt_rm_polarization_fraction(parameters: dict, model: object, component: int = 0) -> float:
    lam2, p, theta, M0 = _faraday_terms(parameters, model, component)
    return  M0 * (-2 * sigmoid_dir(p) * lam2 * np.sin(theta))[:, None]

def deriv2_model_wrt_phi0_phi0(parameters: dict, model: object, component: int = 0) -> float:
    lam2, p, theta, M0 = _faraday_terms(parameters, model, component)
    return M0 * (-4 * sigmoid(p) * np.cos(theta))[:, None]

def deriv2_model_wrt_phi0_polarization_fraction(parameters: dict, model: object, component: int = 0) -> float:
    lam2, p, theta, M0 = _faraday_terms(parameters, model, component)
    return M0 * (-2 * sigmoid_dir(p) * np.sin(theta))[:, None]

def deriv2_model_wrt_polarization_fraction_polarization_fraction(parameters: dict, model: object, component: int = 0) -> float:
    lam2, p, theta, M0 = _faraday_terms(parameters, model, component)
    return M0 * (sigmoid_dir2(p) *np.cos(theta))[:,None]

deriv.deriv_model_wrt_rm = deriv_model_wrt_rm
deriv.deriv_model_wrt_phi0 = deriv_model_wrt_phi0
deriv.deriv_model_wrt_polarization_fraction = deriv_model_wrt_polarization_fraction
deriv.deriv2_model_wrt_rm_rm = deriv2_model_wrt_rm_rm
deriv.deriv2_model_wrt_rm_phi0 = deriv2_model_wrt_rm_phi0
deriv.deriv2_model_wrt_rm_polarization_fraction = deriv2_model_wrt_rm_polarization_fraction
deriv.deriv2_model_wrt_phi0_phi0 = deriv2_model_wrt_phi0_phi0
deriv.deriv2_model_wrt_phi0_polarization_fraction = deriv2_model_wrt_phi0_polarization_fraction
deriv.deriv2_model_wrt_polarization_fraction_polarization_fraction = deriv2_model_wrt_polarization_fraction_polarization_fraction

def make_old_new(old,new):
    d1 = getattr(deriv,f'deriv_model_wrt_{old}')
    if new == 'rm':
        def f(parameters: dict, model: object, component: int = 0) -> float:
            lam2, p, theta, M0 = _faraday_terms(parameters, model, component)
            F = 1 + sigmoid(p)*np.cos(theta)
            return(d1(parameters,model,component) * (-2*sigmoid(p)*lam2*np.sin(theta)/F)[:,None])
    if new == 'phi0':
        def f(parameters: dict, model: object, component: int = 0) -> float:
            lam2, p, theta, M0 = _faraday_terms(parameters, model, component)
            F = 1 + sigmoid(p)*np.cos(theta)
            return(d1(parameters,model,component) * (-2*sigmoid(p)*np.sin(theta)/F)[:,None])
    if new == 'polarization_fraction':
        def f(parameters: dict, model: object, component: int = 0) -> float:
            lam2, p, theta, M0 = _faraday_terms(parameters, model, component)
            F = 1 + sigmoid(p)*np.cos(theta)
            return(d1(parameters,model,component) * (sigmoid_dir(p) * np.cos(theta)/F)[:,None])
    f.__name__ = f'deriv2_model_wrt_{old}_{new}'
    return(f)
for O in [ "amplitude","arrival_time","burst_width","dm","dm_index","scattering_index", "scattering_timescale","spectral_index","spectral_running"]:
    for N in ['rm','phi0','polarization_fraction']:
        setattr(deriv,f'deriv2_model_wrt_{O}_{N}',make_old_new(O,N))
            