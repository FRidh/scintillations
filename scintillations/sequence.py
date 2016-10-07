"""
Sequence
=========

Generate a sequence of scintillations.

Note that this implementation takes into account varying correlation time by updating
the impulse response, and not resampling.

"""
import numpy as np
from scipy.signal import resample, fftconvolve

import collections
from scintillations.common import *
from scintillations.common import _fluctuations_with_variance

#from turbulence.vonkarman import covariance_wind as _covariance_vonkarman_wind

def variance_gaussian(distance, wavenumber, scale, mean_mu_squared, include_saturation=False):
    """Variance of Gaussian fluctuations.

    :param distance: Distance.
    :param wavenumber: Wavenumber.
    :param scale: Correlation length
    :param mean_mu_squared: Mean mu squared.

    :param include_saturation: Whether to include log-amplitude saturation. In this case the variance is multiplied with the saturation factor, :func:`saturation_factor`.
    :returns: Variance

    .. math:: \\langle \\chi^2 \\rangle = \\langle S^2 \\rangle = \\frac{\\sqrt{\\pi}}{2} \\langle \\mu^2 \\rangle k^2 r L

    """
    variance = np.sqrt(np.pi)/2.0 * mean_mu_squared * wavenumber*wavenumber * distance * scale
    if include_saturation:
        variance *= saturation_factor(distance, wavenumber, scale, mean_mu_squared)
    return variance


def fluctuations_with_variance(fluctuations, wavenumber, distance, correlation_length, mean_mu_squared, include_saturation=False):
    """Generate fluctuations with correct variance.
    """
    return _fluctuations_with_variance(variance_gaussian, fluctuations, wavenumber, distance, correlation_length, mean_mu_squared, include_saturation)


def covariance_gaussian(spatial_separation, distance, wavenumber, scale, mean_mu_squared):
    """Calculate the covariance of a Gaussian turbulence spectrum and spherical waves.

    See Daigle, 1987: equation 2 and 3.

    :param spatial_separation: Spatial separation.
    :param distance: Distance.
    :param wavenumber: Wavenumber.
    :param mean_mu_squared: Mean mu squared.
    :param scale: Outer length scale.
    :returns: Covariance

    .. math:: B{\\chi} (\\rho) = B{S}(\\rho) = \\frac{\\sqrt{\\pi}}{2} \\langle \\mu^2 \\rangle k^2 r L \\frac{\\Phi(\\rho/L) }{\\rho / L}

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


def generate_gaussian_fluctuations_standard(nsamples, ntaps, fs, correlation_time, state=None, window=None):
    """Generate Gaussian fluctuations with variance 1.

    :param nsamples: Length of the sequence in samples.
    :param ntaps: Length of the filter used to shape the PSD of the sequence.
    :param fs: Sample frequency.
    :param correlation_time: Correlation time. Single value or array with a value for each time instance.
    :param state: State of random number generator.
    :param window: Window used in filter design.
    :returns: Fluctuations.

    This function generates fluctuations that are Gaussian distributed and have a variance of 1.

    """
    # Gaussian white noise
    state = state if state else np.random.RandomState()
    nsamples_total = nsamples + ntaps - 1
    noise = state.randn(nsamples_total)

    # Time shifts for sampling temporal correlation function.
    times = tau(ntaps, fs)

    # Whether our correlation time is time-variant or not.
    # This is an optimization step.
    if isinstance(correlation_time, collections.Iterable):
        if len(set(correlation_time)) > 1:  # Unique values
            time_variant = True
        else:   # Sequence with same value
            correlation_time = correlation_time[0]
            time_variant = False
    else:   # Single value
        time_variant = False;

    if time_variant:
        # We need to convolve each sample with a unique impulse response.
        fluctuations = np.empty(nsamples)
        for i in range(nsamples):
            noise_block = noise[i:i+ntaps]
            correlation = correlation_spherical_wave(times, correlation_time[i])
            ir = impulse_response_fluctuations(correlation, ntaps, window=window)
            fluctuations[i] = fftconvolve(noise_block, ir, mode='valid')

    else:
        # If time-invariant, we can perform a single convolution.
        correlation = correlation_spherical_wave(times, correlation_time)
        ir = impulse_response_fluctuations(correlation, ntaps, window=window)
        fluctuations = fftconvolve(noise, ir, mode='valid')

    # The filtering process adjusts the spectrum of the sequence and thereby the variance.
    # One might want to normalize with the standard deviation to take this into account.
    # However, the expected value of the sequence is still one, and therefore
    # normalization should not be done.
    #fluctuations /= fluctuations.std()

    return fluctuations




def generate_fluctuations_logamp_and_phase(nsamples, fs, ntaps, correlation_length, speed, frequency, soundspeed,
                                           distance, mean_mu_squared, include_saturation=True, state=None, window=None):
    """Generate logamp and phase fluctuations.

    :returns: Logamp :math:`\chi(t)` and phase :math:`S(t)` fluctuations.

    """
    correlation_length = np.atleast_1d(correlation_length)
    speed = np.atleast_1d(speed)
    correlation_time = correlation_length / speed
    mean_mu_squared = np.atleast_1d(mean_mu_squared)
    distance = np.atleast_1d(distance)

    wavenumber = 2.*np.pi*frequency / soundspeed
    # Fluctuations with variance 1.
    fluctuations = generate_gaussian_fluctuations_standard(nsamples, ntaps, fs, correlation_time, state=state, window=window)

    # Log-amplitude and phase fluctuations with correct variance.
    logamp = fluctuations_with_variance(fluctuations[:,None], wavenumber, distance[:,None], correlation_length[:,None], mean_mu_squared[:,None], include_saturation)
    phase =  fluctuations_with_variance(fluctuations[:,None], wavenumber, distance[:,None], correlation_length[:,None], mean_mu_squared[:,None], False)

    return np.squeeze(logamp), np.squeeze(phase)


def generate_fluctuations_spectra_and_delay(nsamples, fs, ntaps, correlation_length, speed, frequency, soundspeed,
                                            distance, mean_mu_squared, include_saturation=True, state=None, window=None):
    """Generate fluctuations represented by a time-variant magnitude-only spectrum and a variable delay.

    :returns: Magnitude-only spectra as function of time, and delay as function of time
    .. seealso:: :func:`generate_complex_fluctuations`
    """
    correlation_length = np.atleast_1d(correlation_length)
    speed = np.atleast_1d(speed)
    correlation_time = correlation_length / speed
    mean_mu_squared = np.atleast_1d(mean_mu_squared)
    distance = np.atleast_1d(distance)

    wavenumber = 2.*np.pi*frequency / soundspeed
    # Fluctuations with variance 1.
    fluctuations = generate_gaussian_fluctuations_standard(nsamples, ntaps, fs, correlation_time, state=state, window=window)

    # Log-amplitude and phase fluctuations with correct variance.
    logamp = fluctuations_with_variance(fluctuations[:,None], wavenumber, distance[:,None], correlation_length[:,None], mean_mu_squared[:,None], include_saturation=include_saturation)
    # Phase fluctuations with correct variance for frequency=1
    phase =  fluctuations_with_variance(fluctuations[:,None], 2.*np.pi*1.0/soundspeed, distance[:,None], correlation_length[:,None], mean_mu_squared[:,None], False)

    # Magnitude spectra
    spectra = np.squeeze(amplitude_fluctuations(logamp))

    # Variable delay
    delay = np.squeeze(delay_fluctuations(phase, fs, 1.0))

    return spectra, delay


def apply_log_amplitude(signal, log_amplitude):
    """Apply log-amplitude fluctuations.

    :param signal: Pressure signal.
    :param log_amplitude: Log-amplitude fluctuations.

    .. math:: p_m = p \\exp{\\chi}

    """
    return signal * amplitude_fluctuations(log_amplitude)


def apply_phase(signal, phase, frequency, fs):
    """Apply phase fluctuations.

    :param signal: Pressure signal.
    :param phase: Phase fluctuations.
    :param frequency: Frequency of tone.
    :param fs: Sample frequency.

    Phase fluctuations are applied through a resampling.

    """
    delay = delay_fluctuations(phase, frequency)
    signal = apply_delay(signal, delay, fs)
    return signal


def apply_fluctuations(signal, fs, frequency=None, log_amplitude=None, phase=None):
    """Apply log-amplitude and/or phase fluctuations.
    """
    if log_amplitude is not None:
        signal = apply_log_amplitude(signal, log_amplitude)
    if phase is not None:
        signal = apply_phase(signal, phase, frequency, fs)
    return signal


def apply_delay(signal, delay, fs):
    """Apply propagation delay fluctuations.

    :param signal: Signal
    :param delay: Propagation delay fluctuations.
    :param fs: Sample frequency
    :returns: Frequency-modulated signal.
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


def modulate_tone(signal, fs, frequency, fs_low, correlation_length, speed, distance, soundspeed, mean_mu_squared,
                  ntaps_corr=8192, window=None, include_saturation=False, state=None, include_amplitude=True,
                  include_phase=True):
    """Modulate tone.

    :param signal: Signal to modulate.
    :param fs: Sample frequency of signal.
    :param fs_low: Sample frequency at which to generate the scintillations.

    .. note:: Time-dependent values like speed and correlation length are given per block.
    """

    nsamples = len(signal)
    nsamples_low = int(np.ceil(nsamples * fs_low / fs))
    times = np.arange(nsamples)/fs
    times_low = np.arange(nsamples_low)/fs_low

    logamp, phase = generate_fluctuations_logamp_and_phase(nsamples_low, fs_low, ntaps_corr, correlation_length, speed, frequency,
                                                           soundspeed, distance, mean_mu_squared, include_saturation=include_saturation,
                                                           state=state, window=window)

    if include_amplitude:
        logamp = np.interp(times, times_low, logamp)
        signal = apply_log_amplitude(signal, logamp)

    if include_phase:
        delay = delay_fluctuations(phase, fs_low, frequency)
        delay = np.interp(times, times_low, delay)
        #delay = delay * fs_low #/ frequency # FIXME: Apparently we need this factor. WHY???!
        #phase = np.interp(times, times_low, phase)
        #delay = delay_fluctuations(phase, fs_low)

        signal = apply_delay(signal, delay, fs)

    return signal


def modulate_broadband(signal, fs, nblock, correlation_length, speed, distance, soundspeed, mean_mu_squared, ntaps_corr=8192,
             ntaps_spectra=128, window=None, include_saturation=False, state=None,
             include_amplitude=True, include_phase=True):
    """Modulate tone.
    """
    return NotImplemented
    fb = fs / nblock
    frequency = np.fft.rfftfreq(ntaps_spectra, 1./ fs)
    spectrum, delay = generate_fluctuations_spectra_and_delay(nsamples, fb, ntaps_corr, correlation_length, speed,
                                                                 frequency, soundspeed, distance, mean_mu_squared,
                                                                 include_saturation, state, window)
    if include_amplitude:
        return NotImplemented
        logamp = np.interp(times, times_low, logamp)
        signal = apply_log_amplitude(signal, logamp)

    if include_phase:
        delay = np.interp(times, times_low, delay)
        signal = apply_delay(signal, delay, fs)

    return signal

#def generate_gaussian_fluctuations(nsamples, ntaps, fs, correlation_length, speed, distance,
                                   #frequency, soundspeed, mean_mu_squared,
                                   #window=None, include_saturation=False,
                                   #state=None, factor=5.0):
    #"""Generate Gaussian fluctuations.

    #:param factor: To resolve spatial field you need a sufficient resolution.

    #This function generated log-amplitude and phase fluctuations with the correct variance for the specified frequency.

    #.. warning:: You likely do not want to use a window as it will dramatically alter the frequency response of the fluctuations.
    #.. seealso:: :func:`generate_gaussian_fluctuations_standard`
    #"""
    #correlation_time = correlation_length / speed
    ## Low resolution parameters
    #fs_low = factor / correlation_time
    #times = np.arange(nsamples)/fs
    ###correlation = correlation_spherical_wave(tau(ntaps, fs_low), correlation_time)
    #upsample_factor = fs / fs_low
    #nsamples_low = np.ceil(nsamples / upsample_factor)
    #times_low = np.arange(nsamples_low) / fs_low

    ## Modulation signal with variance 1
    #fluctuations = generate_gaussian_fluctuations_standard(nsamples_low, ntaps, fs_low, correlation_time, state, window)

    #wavenumber = 2.*np.pi*frequency / soundspeed

    ## Log-amplitude fluctuations with correct variance.
    #logamp = logamp_fluctuations(fluctuations, wavenumber, distance, correlation_length, mean_mu_squared,
                                 #include_saturation=include_saturation)

    ## Phase fluctuations with correct variance.
    #phase = phase_fluctuations(fluctuations, wavenumber, distance, correlation_length, mean_mu_squared)

    ## Upsampled modulation signals
    #logamp = np.interp(times, times_low, logamp)
    #phase = np.interp(times, times_low, phase)

    #return logamp, phase

#from scipy.signal import resample

#def upsample(signal, factor, method='fourier'):
    #"""Upsample signal with `factor`.

    #:param signal: Signal to be upsampled.
    #:param factor: Upsample factor.
    #:type factor: float
    #:returns: Upsampled signal.

    #.. seealso:: :func:`scipy.signal.resample`

    #"""
    #nsamples_low = len(signal)
    #nsamples_high = nsamples_low * np.ceil(factor)

    #if method=='fourier':
        #upsampled = resample(signal, nsamples_high)
    #elif method=='linear':
        #times_low = np.linspace(0., 1., nsamples_low, endpoint=False)
        #times_high = np.linspace(0., 1., nsamples_high, endpoint=False)
        #upsampled = np.interp(times_high, times_low, signal)
    #else:
        #raise ValueError("Invalid method: {}".format(method))

    #return upsampled


    #if nsamples_upsampled:
        #times = np.arange(nsamples)
        #fs_upsampled = nsamples_upsampled / nsamples * fs
        #times_upsampled = np.arange(nsamples_upsampled) / fs_upsampled
        #logamp = np.interp(times_upsampled, times, logamp)
        #phase = np.interp(times_upsampled, times, phase)



#def modulate(signal, fs, correlation_length, speed, distance, soundspeed, mean_mu_squared, ntaps=8192,
             #nfreqs=100, window=None, include_saturation=False, state=None, factor=5.0,
             #include_amplitude=True, include_phase=True, fs_low=None):
    #"""Apply modulations to `signal`.

    #:param signal: Signal to apply modulations to.
    #:param fs: Sample frequency of signal.
    #:param correlation_length: Correlation length.
    #:param speed: Speed.
    #:param distance: Distance.
    #:param soundspeed: Speed of sound.
    #:param mean_mu_squared: Variance of refractive-index.
    #:param ntaps: Amount of taps to use to sample the correlation function.
    #:param nfreqs: Amount of frequencies to calculate the logamp variances for.
    #:param window: Window to apply when designing filter.
    #:param include_saturation: Include logamp saturation.
    #:param state: State of the PRNG.
    #:param factor: Oversampling factor.
    #:param include_amplitude: Whether to include logamp modulations.
    #:param include_phase: Whether to include phase modulations.

    #"""
    #correlation_time = correlation_length / speed
    #if fs_low is None:
        #fs_low = np.max(5.0 / correlation_time)

    #times = np.arange(nsamples)/fs

    #upsample_factor = fs / fs_low
    #nsamples_low = np.ceil(nsamples / upsample_factor)
    #times_low = np.arange(nsamples_low) / fs_low

    #fluctuations = generate_complex_fluctuations




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


#def _spatial_separation(A, B, C):
    #"""Spatial separation.

    #:param A: Source position as function of time.
    #:param B: Reference position, e.g. source position at t-1
    #:param C: Receiver position.

    #Each row is a sample and each column a spatial dimension.

    #"""
    #a = np.linalg.norm(B-C, axis=1)
    #b = np.linalg.norm(A-C, axis=1)
    #c = np.linalg.norm(A-B, axis=1)

    #gamma = np.arccos((a**2.0+b**2.0-c**2.0) / (2.0*a*b))
    #spatial_separation = 2.0 * b * np.sin(gamma/2.0)
    #L = b * np.cos(gamma/2.0)

    #return spatial_separation, L


#def transverse_speed(A, B, C, fs):
    #"""Transverse speed computed from three positions.

    #:param A: Source position as function of time.
    #:param B: Reference position, e.g. source position at t=0
    #:param C: Receiver position.
    #:param fs: Sample frequency

    #"""
    #rho, _ = _spatial_separation(A, B, C)

    #step_distance = np.linalg.norm(A-B, axis=-1)
    #speed = step_distance * fs
    #transverse = rho / step_distance * speed

    #return transverse, rho, step_distance



