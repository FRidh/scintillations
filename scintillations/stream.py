"""
Stream
======

Generate a stream of scintillations. Functions in this module are supposed to work with :mod:`streaming.stream`.

Varying correlation time is taken into account by resampling the generated fluctuations.

"""
import streaming
import itertools
from scintillations.common import * # amplitude_fluctuations, delay_fluctuations, impulse_response_fluctuations, variance_gaussian, correlation_spherical_wave
from scintillations.common import _fluctuations_with_variance


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
    variance = np.sqrt(np.pi)/2.0 * mean_mu_squared.copy() * wavenumber.copy()**2.0 * distance.copy() * scale.copy()
    if include_saturation:
        variance *= saturation_factor(distance, wavenumber, scale, mean_mu_squared)
    return variance


def fluctuations_with_variance(fluctuations, wavenumber, distance, correlation_length, mean_mu_squared, include_saturation=False):
    """Generate fluctuations with correct variance.
    """
    return _fluctuations_with_variance(variance_gaussian, fluctuations, wavenumber, distance, correlation_length, mean_mu_squared, include_saturation)


#from common import *

#def _generate_sequence(ntaps, D, D0, state):

    ## D = v / L * fs

    #x = np.arange(ntaps) #* D0
    #correlation = _correlation_spherical_wave(x)

    ##correlation = correlation_spherical_wave(tau(ntaps, fs_base), fs_base)
    #ir = impulse_response_fluctuations(correlation, ntaps)

    #noise = streaming.signal.noise(state=state)
    ## Fluctuations without taking into account the actual correlation length, speed or sample frequency
    ## Ideal block size? I don't know. Depends on the above parameters!
    #fluctuations = streaming.signal.convolve_overlap_save(noise, streaming.signal.constant(ir), nhop=8192, ntaps=ntaps)

    ## Sample times for the initial sample frequency
    #times_compressed =  streaming.signal.times(D0)

    ## Sample times at output sample frequency, taking into account the correlation time
    #times = streaming.Stream(itertools.chain([0.0], streaming.signal.cumsum(D)))

    #fluctuations = streaming.signal.interpolate(times_compressed, fluctuations, times)
    #return fluctuations


#def generate_fluctuations_resample_fluctuations(ntaps, fs_desired, correlation_time, state, f0):
    #D = 1./(correlation_time * fs_desired)
    #D0 = f0 / f_desired

    #return _generate_sequence(ntaps, D, D0, state)




def generate_fluctuations_resample_fluctuations(ntaps, fs_desired, correlation_time, state, fs_base=50., window=None):
    """Generate fluctuations.

    :param fs_desired: Sample frequency of the input and the output.
    :param fs: Sample frequency at which to compute the initial fluctuations.

    :returns: Fluctuations.

    .. note:: This function applies change of relative speed by resampling the fluctuations at a different pace.

    """
    correlation = correlation_spherical_wave(tau(ntaps, fs_base), fs_base)
    ir = impulse_response_fluctuations(correlation, ntaps)

    # We now generate fluctuations without taking into account the actual correlation length, speed or sample frequency.
    # Ideal block size? I don't know. Depends on the above parameters!
    nhop = ntaps # Arbitrary
    noise = streaming.signal.noise(nblock=nhop, state=state)
    fluctuations = streaming.signal.convolve_overlap_add(noise, streaming.signal.constant(ir), nhop=nhop, ntaps=ntaps)

    # Sample times for the initial sample frequency
    times_compressed = streaming.signal.times(1./fs_base)

    # Sample times at output sample frequency, taking into account the correlation time
    times = streaming.Stream(itertools.chain([0.0], streaming.signal.cumsum(1./correlation_time / fs_desired)))

    # Resample fluctuations to take into account the actual parameters
    fluctuations = streaming.signal.interpolate(times_compressed, fluctuations, times)

    return fluctuations


def generate_fluctuations_resample_filter(fs_desired, fs, ntaps, speed, correlation_length, state, window=None):
    """Generate fluctuations.

    .. note:: This function applies change of relative speed by updating the impulse response of the filter by resampling the initial frequency response.
    """
    return NotImplemented


def generate_fluctuations_update_filter(ntaps, fs, correlation_time, state, window=None):
    """Generate fluctuations.

    .. note:: This function applies change of relative speed by updating the impulse response of the filter by recomputing.
    """
    nblock = ntaps
    _tau = tau(ntaps, fs)
    correlation = correlation_time.map(lambda x: correlation_spherical_wave(_tau, x))

    ir = correlation.map(lambda c: impulse_response_fluctuations(c, ntaps, window=window))
    #ir = streaming.Stream(impulse_response_fluctuations(c, ntaps, window=window) for c in correlation)

    # Noise generator. Samples from the standard normal distribution.
    state = state if state is not None else np.random.RandomState()
    noise = streaming.signal.noise(nblock=None, state=state).blocks(nblock=ntaps, noverlap=ntaps-1)
    #noise = streaming.BlockStream( (state.randn(nblock) for i in itertools.count()), nblock)
    #fluctuations = streaming.Stream(fftconvolve(block, i, mode='valid')[0] for block, i in zip(noise, ir))
    fluctuations = streaming.signal.convolve_overlap_save(noise, ir, nblock, ntaps).samples()
    return fluctuations


def generate_fluctuations(fs_desired, fs, ntaps, speed, correlation_length, state, window=None, method="resample_fluctuations"):
    """Generate fluctuations.
    """
    if method == 'resample_fluctuations':
        return generate_fluctuations_resample_fluctuations(fs_desired, fs, ntaps, speed, correlation_length, state, window=None)
    elif method == 'update_filter':
        return generate_fluctuations_update_filter(fs_desired, fs, ntaps, speed, correlation_length, state, window=None)
    elif method == 'resample_filter':
        return generate_fluctuations_resample_filter(fs_desired, fs, ntaps, speed, correlation_length, state, window=None)



#def fluctuations_naf(nblock, fs, nbins, ntaps, speed, correlation_length,
                     #distance, soundspeed, mean_mu_squared, window, seed, include_saturation):
    #"""Generate gains and delays for amplitude and phase modulations.

    #:param nblock: Blocksize the NAF uses.
    #:param fs: Sample frequency the NAF uses.
    #:param nbins: Amount of frequency bins for the amplitude modulations' spectra.
    #:param ntaps: Amount of points to sample the correlation function.

    #.. note:: Single values per block.
    #"""
    ## Blocks per second in the NAF
    #fs_desired = fs / nblock
    ## Samples per second at which to compute fluctuations
    #fs_low = 1000.

    ## Warn when sample frequency is too low.

    ##------Fluctuations with variance 1---------
    ## Ideally we use the function that works at lower sample frequency. However, then it gets difficult because the
    ## amount of taps is going to be higher than the blocksize (1).
    ##fluctuations = _generate_gaussian_fluctuations(nblock, ntaps, fs, correlation_time, window=window, state=state)
    ## Instead we now consider the upsampled version, partition in blocks with the blocksize the NAF uses,
    ###fluctuations = turbulence_fluctuations(ntaps, fs, fs_low, ntaps, speed,
    ###                                       correlation_length, window=window, state=np.random.RandomState(seed))
    ## and then just pick the last item of the fluctuations.
    ###fluctuations = streaming.Stream(fluctuations.blocks(nblock).map(lambda x: x[-1]))
    #state = np.random.RandomState(seed)
    #fluctuations = generate_fluctuations_resampling(fs_desired, fs_low, ntaps, speed, correlation_length, state, window=window)

    #logamp, phase = fluctuations.tee()

    ##------Amplitude fluctuations--------------
    #frequencies = np.fft.rfftfreq(nbins, 1./fs)
    #wavenumbers = 2.*np.pi*frequencies / soundspeed
    #logamp_func = lambda d : variance_gaussian(d, wavenumbers, correlation_length, mean_mu_squared, include_saturation=include_saturation)
    ## Variances as function of time
    ##variance_logamp = distance.map(logamp_func)
    #variance_logamp = logamp_func(distance)
    ##print(variance_logamp.peek())
    ##print(logamp.peek())

    ## Spectral gains as function of time
    #spectrum = np.exp( logamp * variance_logamp.sqrt() )

    ##------Phase or delay fluctuations---------
    #omega = 2.0*np.pi*1.0
    #wavenumber = omega / soundspeed
    #phase_func = lambda d : variance_gaussian(d, wavenumber, correlation_length, mean_mu_squared)
    #variance_phase = distance.copy().map(phase_func)
    #phase = phase * variance_phase.sqrt()
    #delay = phase / omega

    #return frequencies, spectrum, delay


def _stream_or_constant(x):
    if isinstance(x, streaming.abstractstream.AbstractStream):
        return x.samples()
    else:
        return streaming.signal.constant(x)


def generate_fluctuations_spectra_and_delay(fs, ntaps, correlation_length, speed, frequency, soundspeed,
                                            distance, mean_mu_squared, fmin, include_saturation=True, state=None, window=None):
    """Generate fluctuations.

    :param fs: Sample frequency.
    :param ntaps: Taps of impulse response.
    :param correlation_length: Correlation length.

    :returns: Frequencies, magnitude spectra and delays.
    """
    correlation_length = _stream_or_constant(correlation_length)
    speed = _stream_or_constant(speed)
    soundspeed = _stream_or_constant(soundspeed)
    distance = _stream_or_constant(distance)
    mean_mu_squared = _stream_or_constant(mean_mu_squared)

    correlation_time = correlation_length.copy() / speed.copy()

    fluctuations = generate_fluctuations_resample_fluctuations(ntaps, fs, correlation_time, state, window=window, fs_base=fmin).samples()

    wavenumber_logamp = 2.*np.pi*frequency/soundspeed.copy()
    wavenumber_phase = 2.*np.pi*1.0/soundspeed.copy()
    logamp = fluctuations_with_variance(fluctuations.copy(), wavenumber_logamp, distance.copy(), correlation_length.copy(), mean_mu_squared.copy(), include_saturation)
    phase  = fluctuations_with_variance(fluctuations.copy(), wavenumber_phase , distance.copy(), correlation_length.copy(), mean_mu_squared.copy(), False)

    spectra = amplitude_fluctuations(logamp)
    delay = delay_fluctuations(phase, fs, 1.0)

    del fluctuations, distance, correlation_length, mean_mu_squared, speed, soundspeed

    return spectra, delay


def generate_fluctuations_complex_spectrum(fluctuations):
    """Generate fluctuations.

    :returns: Frequencies, complex spectra
    """
    correlation_length = _stream_or_constant(correlation_length)
    speed = _stream_or_constant(speed)
    soundspeed = _stream_or_constant(soundspeed)
    distance = _stream_or_constant(distance)
    mean_mu_squared = _stream_or_constant(mean_mu_squared)

    correlation_time = correlation_length.copy() / speed

    fluctuations = generate_fluctuations_resample_fluctuations(ntaps, fs, correlation_time, state, window=window, fs_base=fmin).samples()

    wavenumber = 2.*np.pi*frequency/soundspeed
    logamp = fluctuations_with_variance(fluctuations.copy(), wavenumber.copy(), distance.copy(), correlation_length.copy(), mean_mu_squared.copy(), include_saturation)
    phase =  fluctuations_with_variance(fluctuations, wavenumber, distance, correlation_length, mean_mu_squared, False)

    complex_spectra = complex_fluctuations(logamp, phase)
    return complex_spectra


def modulate_with_spectra_and_delay(signal, fs, nhop, spectra, delay, ntaps=None):
    """Modulate signal with amplitude and phase fluctuations.

    :param signal: Carrier signal.
    :param fs: Sample frequency of signal.
    :param nhop: Compute impulse response to describe fluctuations every `nhop` samples.
    :param spectra: Spectrum for each block. Single side magnitude only.
    :param delay: Delay for each block.
    :param ntaps: Amount of filter taps to use for IR of `spectra`.
    :returns: Modulated signal.
    :rtype: :class:`streaming.stream.Stream()`

    .. note:: The spectra describe amplitude fluctuations and the delay is the phase or propagation delay fluctuations.
    """
    #signal = signal.blocks(nhop)

    # Amplitude modulations
    if spectra is not None:
        if ntaps is None:
            raise ValueError("ntaps should be specified when including amplitude modulations.")
        # Calculate an IR for each amplitude modulations spectra
        # We apply linear-phase
        compute_ir = lambda x: np.fft.ifftshift(np.fft.irfft(x, n=ntaps))
        ir = spectra.map(compute_ir)
        #ir = streaming.Stream((np.fft.ifftshift(np.fft.irfft(2.*x, n=ntaps)) for x in spectra))

        # Apply amplitude modulations
        signal = streaming.signal.convolve_overlap_save(signal, ir, nhop, ntaps)
        # Linear-phase filter. Correct for group delay.
        signal = signal.samples().drop(int(ntaps//2))

    # Time-delay modulations
    if delay is not None:
        # Interpolate time-delay modulations
        times = streaming.signal.times(1./fs)
        fb = fs / nhop
        times_low = streaming.signal.times(1./fb)
        delay = streaming.signal.interpolate(times_low, delay, times.copy())
        # Apply time-delay modulations
        signal = streaming.signal.vdl(signal, times, delay)

    return signal


def modulate_with_complex_spectra(signal, fs, nhop, complex_spectra, ntaps):
    """Modulate signal with amplitude and phase fluctuations.

    .. note:: The complex spectra includes amplitude and phase fluctuations.

    """
    ir = spectrum.map(lambda x: np.fft.ifftshift(np.fft.irfft(x, n=ntaps)))
    # Convolve signal with complex spectra.
    # Because we add a linear-phase (ifftshift) we correct for that by dropping samples.
    signal = streaming.signal.convolve_overlap_save(signal, ir, nhop, ntaps).samples().drop(ntaps//2)
    return signal


#def apply_modulations(signal, fs, nblock, spectrum, delay, ntaps=None):
    #"""Modulate `signal` with amplitude modulations given by `spectrum` and time-delay modulations given by `delay`.

    #"""
    #return modulate_with_complex_spectra(signal, fs, nblock, spectrum, delay, ntaps=ntaps)


def modulate(signal, fs, nhop, correlation_length, speed, distance, soundspeed, mean_mu_squared, fmin, ntaps_corr=8192,
             ntaps_spectra=128, window=None, include_saturation=False, state=None,
             include_amplitude=True, include_phase=True, method="spectra_and_delay"):
    """Compute and apply scintillations to `signal`.

    :param signal: Signal to modulate.
    :type signal: :class:`Stream`
    :param fs: Sample frequency
    :param nhop: Compute new impulse response and delay every `nhop` samples.
    :param correlation: Correlation length.
    :type correlation: Either a single value, or a :class:`Stream` sampled at `fs/nhop`.
    :param speed: Transverse speed.
    :type correlation: Either a single value, or a :class:`Stream` sampled at `fs/nhop`.
    :param distance: Source-receiver distance.
    :type distance: Either a single value, or a :class:`Stream` sampled at `fs/nhop`.
    :param soundspeed: Speed of sound.
    :type soundspeed: Either a single value, or a :class:`Stream` sampled at `fs/nhop`.
    :param mean_mu_squared: Variance of the refractive-index field.
    :type mean_mu_squared: Either a single value, or a :class:`Stream` sampled at `fs/nhop`.
    :param ntaps_corr: Length of the impulse responses for the correlation filter.
    :param ntaps_spectra: Length of the impulse responses for the eventual magnitude-filter.
    :param window: Window
    :param include_saturation: Whether to include log-amplitude saturation.
    :param state: State of the PRNG.
    :param include_amplitude: Whether to include log-amplitude fluctuations.
    :param include_phase: Whether to include phase fluctuations.
    :param method: Method to use.


    """
    # Generate complex spectra or magnitude spectra and delays
    # Block rate
    fb = fs / nhop
    frequency = np.fft.rfftfreq(ntaps_spectra, 1./ fs)

    ## Downsample speed, correlation_length, distance, soundspeed, mean_mu_squared
    ## TODO: lowpass filter?
    #if isinstance(speed, streaming.abstractstream.AbstractStream):               speed = speed.samples().take_nth(nhop)
    #if isinstance(correlation_length, streaming.abstractstream.AbstractStream):  correlation_length = correlation_length.samples().take_nth(nhop)
    #if isinstance(distance, streaming.abstractstream.AbstractStream):            distance = distance.samples().take_nth(nhop)
    #if isinstance(soundspeed, streaming.abstractstream.AbstractStream):          soundspeed = soundspeed.samples().take_nth(nhop)
    #if isinstance(mean_mu_squared, streaming.abstractstream.AbstractStream):     mean_mu_squared = mean_mu_squared.samples().take_nth(nhop)

    if method=="complex_spectra":
        complex_spectra = generate_fluctuations_complex_spectra(fb, ntaps_corr, correlation_length, speed,
                                                                frequency, soundspeed, distance, mean_mu_squared,
                                                                fmin, include_saturation, state, window)

        modulated = modulate_with_complex_spectra(signal, fs, nhop, complex_spectra, ntaps=ntaps_spectra)

    elif method=="spectra_and_delay":

        spectra, delay = generate_fluctuations_spectra_and_delay(fb, ntaps_corr, correlation_length, speed,
                                                                 frequency, soundspeed, distance, mean_mu_squared,
                                                                 fmin, include_saturation, state, window)
        # Apply modulations
        if not include_amplitude:
            spectra = None
        if not include_phase:
            delay = None
        modulated = modulate_with_spectra_and_delay(signal, fs, nhop, spectra, delay, ntaps=ntaps_spectra)
    else:
        raise ValueError("Incorrect method.")

    return modulated




