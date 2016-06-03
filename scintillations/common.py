"""
Common
======

Expressions common to both :mod:`stream` and :mod:`sequence`.
"""

import numpy as np
from scipy.special import erf
from scipy.fftpack import dct


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


#def impulse_response_fluctuations(covariance, ntaps, window=None):
    #"""Impulse response describing fluctuations.

    #:param covariance: Covariance vector.
    #:param fs: Sample frequency
    #:param window: Window to apply to impulse response. If passed `None`, no window is applied.
    #:returns: Impulse response of fluctuations filter.

    #"""
    #if window is not None:
        #nsamples = covariance.shape[-1]
        #covariance = covariance * window(nsamples)[...,:] # Not inplace!
    ## The covariance is a symmetric, real function.
    #autospectrum = np.abs(np.fft.rfft(covariance, n=ntaps))#/df**2.0 # Autospectrum
    ##autospectrum[..., 0] = 0.0 # Remove DC component from spectrum.

    ## The autospectrum is real-valued. Taking the square root given an amplitude spectrum.
    ## Because we have a symmetric spectrum, taking the inverse DFT results in an even, real-valued
    ## impulse response. Furthermore, because we have zero phase the impulse response is even as well.
    #ir = np.fft.ifftshift(np.fft.irfft(np.sqrt(autospectrum), n=ntaps).real)

    #return ir


# Uses DCT
def impulse_response_fluctuations(covariance, ntaps, window=None):
    """Impulse response describing fluctuations.

    :param covariance: Covariance vector.
    :param fs: Sample frequency
    :param window: Window to apply to impulse response. If passed `None`, no window is applied.
    :returns: Impulse response of fluctuations filter.

    """
    if window is not None:
        nsamples = covariance.shape[-1]
        covariance = covariance * window(nsamples)[...,:] # Not inplace!
    # The covariance is a symmetric, real function.
    autospectrum = np.abs(dct(covariance, type=1))#/df**2.0 # Autospectrum
    #autospectrum[..., 0] = 0.0 # Remove DC component from spectrum.

    # The autospectrum is real-valued. Taking the square root given an amplitude spectrum.
    # Because we have a symmetric spectrum, taking the inverse DFT results in an even, real-valued
    # impulse response. Furthermore, because we have zero phase the impulse response is even as well.
    ir = np.fft.ifftshift(np.fft.irfft(np.sqrt(autospectrum), n=ntaps).real)

    return ir

def tau(ntaps, fs):
    """Time lag :math:`\\tau` for autocorrelation :math:`B(\\tau)`.

    :param ntaps: Amount of taps.
    :param fs: Sample frequency.

    """
    return np.fft.rfftfreq(ntaps, fs/ntaps)


def correlation_spherical_wave(spatial_separation, correlation_length):
    """Correlation of spherical waves.

    :param spatial_separation: Spatial separation :math:`\\rho`.
    :param correlation_length: Correlation length :math:`\\L_0`.
    :returns: Correlation

    .. math:: \\frac{\\sqrt{\\pi}}{2} \\frac{\\mathrm{erf}{(\\rho/L_0)}}{(\\rho/L_0)}

    .. note:: Instead of spatial separation and correlation length, you can also use time lag and correlation time.

    """
    x = np.atleast_1d(spatial_separation/correlation_length)
    cor = np.sqrt(np.pi) / 2.0 * erf(x)/x
    cor[x==0.0] = 1.0
    return cor


def _correlation_spherical_wave(x):
    cor = np.sqrt(np.pi) / 2.0 * erf(x)/x
    cor[x==0.0] = 1.0
    return cor


def _fluctuations_with_variance(variance_func, fluctuations, wavenumber, distance, correlation_length, mean_mu_squared, include_saturation):
    """Add correct variance to fluctuations.

    :returns: Fluctuations with correct variance.
    """
    variance = variance_func(distance, wavenumber, correlation_length, mean_mu_squared, include_saturation)
    return fluctuations * np.sqrt(variance)


#def logamp_fluctuations(variance_func, fluctuations, wavenumber, distance, correlation_length, mean_mu_squared, include_saturation=True):
    #"""Determine log-amplitude fluctuations.

    #:param fluctuations: Fluctuations with variance of one.
    #:param frequency: Frequency to compute fluctuations for.
    #:returns: Log-amplitude fluctuations with correct variance.
    #"""
    #variance_logamp = variance_gaussian(distance, wavenumber, correlation_length, mean_mu_squared, include_saturation=include_saturation)
    #logamp = fluctuations * np.sqrt(variance_logamp)
    #return logamp


#def phase_fluctuations(fluctuations, wavenumber, distance, correlation_length, mean_mu_squared):
    #"""Determine phase fluctuations for given frequency.

    #:param fluctuations: Fluctuations with variance of one.
    #:param frequency: Frequency to compute fluctuations for.
    #:returns: Phase fluctuations with correct variance.
    #"""
    #variance_phase  = variance_gaussian(distance, wavenumber, correlation_length,
                                        #mean_mu_squared, include_saturation=False)
    #phase  = fluctuations * np.sqrt(variance_phase)
    #return phase


def amplitude_fluctuations(logamp):
    """Return amplitude fluctuations given log-amplitude fluctuations.

    :param logamp: Log-amplitude fluctuations.
    :returns: Amplitude fluctuations.

    .. math:: A = \\exp{\\chi}

    """
    return np.exp(logamp)


def delay_fluctuations(phase, fs, frequency=1.0):
    """Return propagation delay fluctuations given phase fluctuations for the specified frequency.

    :param phase: Phase fluctuations for given `frequency`.
    :param frequency: Frequency.
    :returns: Propagation delay fluctuations.

    .. math:: \\mathrm{d}t = -S / (2 \\pi f)

    .. note:: Note the minus sign! This is according to the definition.

    """
    omega = (2.0*np.pi*frequency)
    return phase/omega * (-1)
    #return (phase/omega) / fs * (-1) # We explicitly do a multiplication because Stream does not yet support unary ops


def complex_fluctuations(logamp, phase):
    """Complex fluctuations.

    :param logamp: Log-amplitude fluctuations :math:`\\chi`.
    :param phase: Phase fluctuations :math:`S`.
    :returns: Complex fluctuations :math:`\\Psi`.

    .. math:: \\Psi = e^{\\chi} e^{j\\S}
    """
    return amplitude_fluctuations(logamp) * np.exp(1j*phase)


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


def transverse_speed(velocity_source, orientation):
    """Transverse speed computed from source velocity and source-receiver orientation.

    :param velocity_source: Source velocity.
    :param orientation: Unit vector pointing from source to receiver

    Each row is a sample and each column a spatial dimension.
    The transverse speed is the cross product of the velocity and orientation.

    .. note:: It does not matter whether source-receiver or receiver-source unit vector is given.

    """
    return np.linalg.norm(np.cross(orientation, velocity_source))
