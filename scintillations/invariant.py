"""
Invariant
=========

Atmospheric turbulence causes fluctuations in the sound speed which in effects causes fluctuations in the amplitude and phase of the sound pressure.
"""
import numpy as np
from scipy.signal import resample, fftconvolve
from scipy.special import erf

#from turbulence.vonkarman import covariance_wind as _covariance_vonkarman_wind

def variance_gaussian(distance, wavenumber, scale, mean_mu_squared):
    """Variance of Gaussian fluctuations.

    :param spatial_separation: Spatial separation.
    :param distance: Distance.
    :param wavenumber: Wavenumber.
    :param mean_mu_squared: Mean mu squared.
    :param scale: Correlation length
    :returns: Variance

    .. math:: \\langle \\chi^2 \\rangle = \\langle S^2 \\rangle = \\frac{\\sqrt{\\pi}}{2} \\langle \\mu^2 \\rangle k^2 r L

    """
    return np.sqrt(np.pi)/2.0 * mean_mu_squared * wavenumber*wavenumber * distance * scale


def variance_gaussian_with_saturation(distance, wavenumber, scale, mean_mu_squared, include_saturation=True):

    variance = variance_gaussian(distance, wavenumber, scale, mean_mu_squared)
    if include_saturation:
        variance *= saturation_factor(distance, wavenumber, scale, mean_mu_squared)
    return variance


def correlation_spherical_wave(spatial_separation, correlation_length):
    """Correlation of spherical waves.

    :param spatial_separation: Spatial separation.
    :param correlation_length: Correlation length.
    :returns: Correlation

    .. math:: \\frac{\\sqrt{\\pi}}{2} \\frac{\\erf{x}}{x}

    .. note:: Instead of spatial separation and correlation length, you can also use time lag and correlation time.

    """
    x = np.atleast_1d(spatial_separation/correlation_length)
    cor = np.sqrt(np.pi) / 2.0 * erf(x)/x
    cor[x==0.0] = 1.0
    return cor


def covariance_gaussian(spatial_separation, distance, wavenumber, scale, mean_mu_squared):
    """Calculate the covariance of a Gaussian turbulence spectrum and spherical waves.

    See Daigle, 1987: equation 2 and 3.

    :param spatial_separation: Spatial separation.
    :param distance: Distance.
    :param wavenumber: Wavenumber.
    :param mean_mu_squared: Mean mu squared.
    :param scale: Outer length scale.
    :returns: Covariance

    .. math:: B{\chi} (\rho) = B{S}(\rho) = \\frac{\\sqrt{\\pi}}{2} \\langle \\mu^2 \\rangle k^2 r L \\frac{\\Phi(\\rho/L) }{\\rho / L}

    """
    #covariance = 0.0
    #covariance += (spatial_separation!=0.0) * \
                  #np.nan_to_num( ( np.pi/4.0 * mean_mu_squared * (wavenumber*wavenumber) * \
                  #distance * scale * (erf(spatial_separation/scale) / \
                  #(spatial_separation/scale) ) ) )

    #covariance += (spatial_separation==0.0) * np.sqrt(np.pi)/2.0 * \
                  #mean_mu_squared * (wavenumber*wavenumber) * distance * scale
    cor = correlation_spherical_wave(spatial_separation, scale)
    var = variance_gaussian(distance, wavenumber, scale, mean_mu_squared)
    covariance = cor * var
    return covariance


def transverse_coherence_expected(variance, correlation):
    """Transverse coherence of a spherical waves and Gaussian fluctuations.

    See Daigle, equation 11.
    """
    return np.exp(-2.0*variance * (1.0 - correlation))


def transverse_coherence_expected_large_spatial_separation(variance):
    """Transverse coherence of a spherical waves and Gaussian fluctuations in case the spatial separation is much larger than the correlation length.

    See Daigle, equation 12.
    """
    return np.exp(-2.0 * variance)


def transverse_coherence(logamp_structure, phase_structure):
    """Transverse coherence as function of structure functions.

    See Daigle, equation 6.
    """
    return np.exp(-0.5 * (logamp_structure+phase_structure))


def longitudinal_coherence(logamp_variance, phase_variance):
    """longitudinal coherence.

    See Daigle, equation 13.
    """
    return np.exp(-logamp_variance - phase_variance)


def logamp_structure(logamp_a, logamp_b, axis=-1):
    """Structure function for log-amplitude fluctuations.

    See Daigle, equation 17.
    """
    return ((logamp_a-logamp_b)**2.0).mean(axis=axis)


def phase_structure(phase_a, phase_b, axis=-1):
    """Structure function for phase fluctuations.

    See Daigle, equation 18.

    The last term accounts for the fact that the mean phase difference of the signals may be nonzero.
    """
    return ((phase_a-phase_b)**2.0).mean(axis=axis) - ((phase_a-phase_b).mean(axis=axis))**2.0


def logamp_variance(logamp, axis=-1):
    """Logamp variance.

    See Daigle,  equation 19.
    """
    amp = np.exp(logamp)
    logamp_normalized = np.log(amp/amp.mean(axis=axis))
    return logamp_normalized.var(axis=axis)


def phase_variance(phase, axis=-1):
    """Phase variance.

    See Daigle, equation 20.

    The last term accounts for the fact that the mean phase difference of the signals may be nonzero.
    """
    return phase.var(axis=axis) - (phase-phase.mean(axis=axis)).mean(axis=axis)**2.0


def saturation_distance(mean_mu_squared, wavenumber, scale):
    """Saturation distance according to Wenzel.

    :param mean_mu_squared: Mean mu squared.
    :param wavenumber: Wavenumber.
    :param scale: Outer length scale.

    See Daigle, 1987: equation 5

    .. math:: r_s = \\frac{1}{2 \\langle \mu^2 \\rangle k^2 L}

    """
    return 1.0 / (2.0 * mean_mu_squared * wavenumber*wavenumber * scale)


def saturation_factor(distance, wavenumber, scale, mean_mu_squared):
    """Factor to multiply log-amplitude (co-)variance with to include log-amplitude saturation.

    ..math:: x = \\frac{1}{1 + r/r_s}
    """
    sat_distance = saturation_distance(mean_mu_squared, wavenumber, scale)
    factor = ( 1.0 / (1.0 + distance/sat_distance) )
    return factor


def impulse_response_fluctuations(covariance, fs, window=None):
    """Impulse response describing fluctuations.

    :param covariance: Covariance vector.
    :param fs: Sample frequency
    :param window: Window to apply to impulse response. If passed `None`, no window is applied.
    :returns: Impulse response of fluctuations filter.

    """
    nsamples = covariance.shape[-1]
    df = fs / nsamples

    if window is not None:
        covariance = covariance * window(nsamples)[...,:] # Not inplace!
    # The covariance is a symmetric, real function.
    autospectrum = np.abs(np.fft.rfft(covariance))#/df**2.0 # Autospectrum
    autospectrum[..., 0] = 0.0 # Remove DC component from spectrum.
    del covariance

    # The autospectrum is real-valued. Taking the square root given an amplitude spectrum.
    # Because we have a symmetric spectrum, taking the inverse DFT results in an even, real-valued
    # impulse response. Furthermore, because we have zero phase the impulse response is even as well.
    ir = np.fft.ifftshift((np.fft.irfft(np.sqrt(autospectrum), n=nsamples)).real)
    del autospectrum
    #ir[...,1:] *= 2.0 # Conservation of power. Needed, checked with asymptote of transverse coherence

    return ir


def tau(ntaps, fs):
    """Time lag $\\tau$ for autocorrelation $B\\tau$.

    :param ntaps: Amount of taps.
    :param fs: Sample frequency.

    """
    return np.fft.fftfreq(ntaps, fs/ntaps)


def generate_gaussian_fluctuations_standard(nsamples, ntaps, fs, correlation_time, state=None, window=None):
    """Generate Gaussian fluctuations with variance 1.

    :param nsamples: Length of the sequence in samples.
    :param ntaps: Length of the filter used to shape the PSD of the sequence.
    :param fs: Sample frequency.
    :param correlation_time: Correlation time.
    :param speed: Speed.
    :param state: State of random number generator.
    :param window: Window used in filter design.
    :returns: Fluctuations.

    .. seealso:: :func:`generate_gaussian_fluctuations`
    """
    correlation = correlation_spherical_wave(tau(ntaps, fs), correlation_time)
    # Impulse response of fluctuations
    ir = impulse_response_fluctuations(correlation, fs, window=window)

    # Gaussian white noise
    state = state if state else np.random.RandomState()
    noise = state.randn(nsamples)

    # Generate sequences of fluctuations with variance 1.
    # This is our modulation signal.
    fluctuations = fftconvolve(noise, ir, mode='same')
    fluctuations /= fluctuations.std()

    return fluctuations

def generate_gaussian_fluctuations(nsamples, ntaps, fs, correlation_length, speed, distance,
                                   frequency, soundspeed, mean_mu_squared,
                                   window=None, include_saturation=False,
                                   state=None, factor=5.0):
    """Generate Gaussian fluctuations.

    :param factor: To resolve spatial field you need a sufficient resolution.

    .. warning:: You likely do not want to use a window as it will dramatically alter the frequency response of the fluctuations.
    .. seealso:: :func:`generate_gaussian_fluctuations_standard`
    """
    correlation_time = correlation_length / speed
    # Low resolution parameters
    fs_low = factor / correlation_time
    times = np.arange(nsamples)/fs
    ##correlation = correlation_spherical_wave(tau(ntaps, fs_low), correlation_time)
    upsample_factor = fs / fs_low
    nsamples_low = np.ceil(nsamples / upsample_factor)
    times_low = np.arange(nsamples_low) / fs_low

    # Modulation signal with variance 1
    fluctuations = generate_gaussian_fluctuations_standard(nsamples_low, ntaps, fs_low, correlation_time, state, window)

    wavenumber = 2.*np.pi*frequency / soundspeed

    # Variances of the fluctuations.
    variance_logamp = variance_gaussian_with_saturation(distance, wavenumber, correlation_length,
                                                        mean_mu_squared, include_saturation=include_saturation)
    variance_phase = variance_gaussian(distance, wavenumber, correlation_length, mean_mu_squared)

    # Apply correct variance
    logamp = fluctuations * np.sqrt(variance_logamp)
    phase  = fluctuations * np.sqrt(variance_phase)

    # Upsampled modulation signals
    logamp = np.interp(times, times_low, logamp)
    phase = np.interp(times, times_low, phase)

    return logamp, phase



def covariance(covariance_func, **kwargs):
    if covariance_func == 'gaussian':
        return covariance_gaussian(kwargs['spatial_separation'], kwargs['distance'],
                                   kwargs['wavenumber'], kwargs['scale'], kwargs['mean_mu_squared'])
    elif covariance_func == 'vonkarman_wind':
        return _covariance_vonkarman_wind(kwargs['spatial_separation'], kwargs['distance'],
                                          kwargs['wavenumber'], kwargs['scale'], kwargs['soundspeed'],
                                          kwargs['wind_speed_variance'], kwargs['steps'], kwargs['initial'])
    else:
        raise ValueError("Unknown covariance function {}".format(covariance_func))


def get_covariance(covariance):

    if covariance == 'gaussian':
        def wrapped(**kwargs):
            return covariance_gaussian(kwargs['spatial_separation'], kwargs['distance'],
                                       kwargs['wavenumber'], kwargs['scale'], kwargs['mean_mu_squared'])
    elif covariance == 'vonkarman_wind':
        def wrapped(**kwargs):
            return _covariance_vonkarman_wind(kwargs['spatial_separation'], kwargs['distance'],
                                              kwargs['wavenumber'], kwargs['scale'], kwargs['soundspeed'],
                                              kwargs['wind_speed_variance'], kwargs['steps'], kwargs['initial'])
    else:
        raise ValueError("Covariance unavailable.")

    return wrapped

#def covariance_gaussian(**kwargs):
    #return covariance(spatial_separation, distance, wavenumber, scale, mean_mu_squared)

#def covariance_vonkarman_wind(**kwargs):
    #return _covariance_vonkarman_wind(spatial_separation, distance, wavenumber, scale, soundspeed, wind_speed_variance, steps, initial)

#COVARIANCES = {
        #'gaussian' : covariance,
        #'vonkarman_wind' : covariance_vonkarman_wind,
    #}


###def generate_fluctuations(nsamples, ntaps, fs, speed, distance,
                          ###frequency, soundspeed, scale, state=None,
                          ###window=None, model='gaussian', **kwargs):

    ###logging.debug("generate_fluctuations: covariance model {}".format(model))

    ###try:
        ###include_saturation = kwargs.pop('include_saturation')
    ###except KeyError:
        ###include_saturation = False

    #### Determine the covariance
    ###spatial_separation = tau(ntaps, fs) * speed
    ###wavenumber = 2.*np.pi*frequency / soundspeed

    ###cov = covariance(model, spatial_separation=spatial_separation,
                                 ###distance=distance,
                                 ###wavenumber=wavenumber,
                                 ###scale=scale, **kwargs)

    ####cov0 = covariance_func(spatial_separation=0.0,
                                 ####distance=distance,
                                 ####wavenumber=wavenumber,
                                 ####scale=scale, **kwargs)
    #### Create an impulse response using this covariance
    ###ir = impulse_response_fluctuations(cov, window=window)

    #### We need random numbers.
    ###state = state if state else np.random.RandomState()

    #### Calculate log-amplitude fluctuations
    ####noise = state.randn(samples*2-1)
    ####log_amplitude = fftconvolve(noise, ir, mode='valid')
    ###noise = state.randn(nsamples)
    ###log_amplitude = fftconvolve(noise, ir, mode='same')
    ####log_amplitude -= cov[0]


    ####log_amplitude -= (log_amplitude.mean() - logamp_variance(np.exp(log_amplitude)))

    #### Include log-amplitude saturation
    ###if include_saturation:
        ###if model == 'gaussian':
            ###mean_mu_squared = kwargs['mean_mu_squared']
            ###sat_distance = saturation_distance(mean_mu_squared, wavenumber, scale)
            ###log_amplitude *=  (np.sqrt( 1.0 / (1.0 + distance/sat_distance) ) )
        ###else:
            ###raise ValueError("Cannot include saturation for given covariance function.")

    #### Calculate phase fluctuations
    ####noise = state.randn(samples*2-1)
    ####phase = fftconvolve(noise, ir, mode='valid')
    ###noise = state.randn(nsamples)
    ###phase = fftconvolve(noise, ir, mode='same')

    ###return log_amplitude, phase

#def fluctuations_logamp(ir, noise):
    #log_amplitude = fftconvolve(noise, ir, mode='same')
    #return log_amplitude

#def fluctuations_phase(ir, noise):
    #phase = fftconvolve(noise, ir, mode='same')
    #return phase

def apply_log_amplitude(signal, log_amplitude):
    """Apply log-amplitude fluctuations.

    :param signal: Pressure signal.
    :param log_amplitude: Log-amplitude fluctuations.

    .. math:: p_m = p \\exp{\\chi}

    """
    return signal * np.exp(log_amplitude) # exp(2*log_amplitude) if intensity


def apply_phase(signal, phase, frequency, fs):
    """Apply phase fluctuations.

    :param signal: Pressure signal.
    :param phase: Phase fluctuations.
    :param frequency: Frequency of tone.
    :param fs: Sample frequency.

    Phase fluctuations are applied through a resampling.

    """
    delay = phase/(2.0*np.pi*frequency)
    signal = _apply_delay_turbulence(signal, delay, fs)
    return signal

def apply_fluctuations(signal, fs, frequency=None, log_amplitude=None, phase=None):
    """Apply log-amplitude and/or phase fluctuations.
    """
    if log_amplitude is not None:
        signal = apply_log_amplitude(signal, log_amplitude)
    if phase is not None:
        signal = apply_phase(signal, phase, frequency, fs)
    return signal


def _apply_delay_turbulence(signal, delay, fs):
    """Apply phase delay due to turbulence.

    :param signal: Signal
    :param delay: Delay
    :param fs: Sample frequency
    """

    k_r = np.arange(0, len(signal), 1)          # Create vector of indices
    k = k_r - delay * fs                      # Create vector of warped indices

    kf = np.floor(k).astype(int)       # Floor the warped indices. Convert to integers so we can use them as indices.
    dk = kf - k
    ko = np.copy(kf)
    kf[ko<0] = 0
    kf[ko+1>=len(ko)] = 0
    R = ( (1.0 + dk) * signal[kf] + (-dk) * signal[kf+1] ) * (ko >= 0) * (ko+1 < len(k)) #+ 0.0 * (kf<0)
    return R



#def logamp_variance(logamp, axis=-1):
    #amp = np.exp(logamp)
    #logamp_normalized = np.log(amp/amp.mean(axis=axis))
    #return logamp_normalized.var(axis=axis)

#def logamp_variance(amp):
    #"""Variance of log-amplitude fluctuations.

    #:param amp: Time-series of amplitude fluctuations, NOT log-amplitude.

    #See Daigle, 1983: equation 15, 16 and 19.
    #"""
    #return (np.log(amp/(amp.mean(axis=-1)[...,None]))**2.0).mean(axis=-1)
    ##return (( np.log(amp) - np.log(amp.mean(axis=-1) )[...,None])**2.0).mean(axis=-1)


#def generate_many_gaussian_fluctuations(samples, spatial_separation, distance, wavenumber,
                                        #mean_mu_squared, scale, window=np.hamming,
                                        #include_saturation=False, seed=None):
    #"""Generate time series of log-amplitude and phase fluctuations.

    #:param samples: Length of series of fluctuations.
    #:param spatial_separation: Spatial separation.
    #:param distance: Distance.
    #:param wavenumber: Wavenumber
    #:param mean_mu_squared: Mean mu squared.
    #:param scale: Outer length scale.
    #:param window: Window function.
    #:param include_saturation: Include saturation of log-amplitude.
    #:param seed: Seed.
    #:returns: Log-amplitude array and phase array.

    #This function performs better when many series need to be generated.

    #"""

    ## Calculate correlation
    ##B = (spatial_separation!=0.0) * np.nan_to_num( ( np.pi/4.0 * mean_mu_squared * (k*k)[:,None] * r[None,:] * L * (erf(spatial_separation/L) / (spatial_separation/L))[None,:] ) )
    ##B = (spatial_separation==0.0)[None,:] * (np.sqrt(np.pi)/2.0 * mean_mu_squared * (k*k)* r * L )[:,None]

    #spatial_separation = np.atleast_1d(spatial_separation)
    #distance = np.atleast_1d(distance)
    #wavenumber = np.atleast_1d(wavenumber)
    #mean_mu_squared = np.atleast_1d(mean_mu_squared)
    #scale = np.atleast_1d(scale)
    #covariance = covariance_gaussian(spatial_separation[None,:], distance[:,None],
                                          #wavenumber[:,None], mean_mu_squared[:,None], scale[:,None])

    #if covariance.ndim==2:
        #N = covariance.shape[-2]
    #elif covariance.ndim==1:
        #N = 1
    #else:
        #raise ValueError("Unsupported amount of dimensions.")

    ## Seed random numbers generator.
    #np.random.seed(seed)
    #n = samples * 2 - 1

    #ir = impulse_response_fluctuations(covariance, window=window)

    #noise = np.random.randn(N,n)
    #log_amplitude = fftconvolve1D(noise, ir, mode='valid') # Log-amplitude fluctuations
    #del noise

    #if include_saturation:
        #sat_distance = saturation_distance(mean_mu_squared, wavenumber, scale)
        #log_amplitude *=  (np.sqrt( 1.0 / (1.0 + distance/sat_distance) ) )[...,None]
        #del sat_distance

    #noise = np.random.randn(N,n)
    #phase = fftconvolve1D(noise, ir, mode='valid')           # Phase fluctuations

    #return log_amplitude, phase




#def gaussian_fluctuations_variances(samples, f0, fs, mean_mu_squared,
                                    #distance, scale,
                                    #spatial_separation, soundspeed,
                                    #include_saturation=True, state=None):
    #"""Calculate the variances of fluctuations in the time series of amplitude and phase fluctuations.

    #:param samples: Amount of samples to take.
    #:param f0: Frequency for which the fluctuations should be calculated.
    #:param fs: Sample frequency.
    #:param mean_mu_squared: Mean of refractive-index squared.
    #:param r: Distance.
    #:param L: Outer length scale.
    #:param rho: Spatial separation.
    #:param soundspeed: Speed of sound.
    #:param include_phase: Include phase fluctuations.
    #:param state: State of numpy random number generator.

    #"""
    #spatial_separation *= np.ones(samples)
    #wavenumber = 2.0 * np.pi * f0 / soundspeed
    #a, p = generate_gaussian_fluctuations(samples, spatial_separation, distance,
                                               #wavenumber, mean_mu_squared, scale,
                                               #include_saturation=include_saturation,
                                               #state=state)

    #return logamp_variance(np.exp(a)), phase_variance(p)


#def fluctuations_variance(signals, fs, N=None):
    #"""
    #Determine the variance in log-amplitude and phase by ensemble averaging.

    #:param signals: List of signals or samples.
    #:param fs: Sample frequency

    #The single-sided spectrum is calculated for each signal/sample.

    #The log-amplitude of the :math:`n`th sample is given by

    #.. math:: \\chi^2 = \\ln{\\frac{A_{n}}{A_{0}}}

    #where :math:`A_{n}` is the amplitude of sample :math:`n` and :math:`A_{0}` is the ensemble average

    #.. math:: A_{0} = \\frac{1}{N} \\sum_{n=1}^{N} \\chi_{n}^2

    #"""

    #s = np.array(signals) # Array of signals

    ##print s

    #f, fr = ir2fr(s, fs, N) # Single sided spectrum
    #amp = np.abs(fr)
    #phase = np.angle(fr)
    #logamp_squared_variance = (np.log(amp/amp.mean(axis=0))**2.0).mean(axis=0)
    #phase_squared_variance = ((phase - phase.mean(axis=0))**2.0).mean(axis=0)

    #return f, logamp_squared_variance, phase_squared_variance


#def plot_variance(frequency, logamp, phase):
    #"""
    #Plot variance.
    #"""
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.scatter(frequency, logamp, label=r"$\langle \chi^2 \rangle$", color='b')
    #ax.scatter(frequency, phase, label=r"$\langle S^2 \rangle$", color='r')
    #ax.set_xlim(100.0, frequency.max())
    #ax.set_ylim(0.001, 10.0)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #ax.legend()
    #ax.grid()
    #ax.set_xlabel(r"$f$ in Hz")
    #ax.set_ylabel(r"$\langle X \rangle$")

    #return fig


# -----------------------Streaming version ------------------------


