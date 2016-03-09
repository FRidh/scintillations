from scintillations.invariant import *
from scipy.stats import linregress

def test_autocorrelation():
    """Test whether the autocorrelation of the generated fluctuations matches with
    the desired autocorrelation that is used in the filter design process.
    """

    # Correlation function parameters
    ntaps = 8192
    correlation_length = 1.0
    speed = 1.
    factor = 5.0
    correlation_time = correlation_length / speed
    state = np.random.RandomState(seed=100)
    window = None

    # Duration of sequence.
    fs = factor / correlation_time
    nsamples = ntaps

    fluctuations = generate_gaussian_fluctuations_standard(nsamples, ntaps, fs, correlation_time, state=state, window=window)

    # We expect to obtain fluctuations with variance 1.0
    assert np.allclose( fluctuations.var(), 1.0)

    # We now calculate the autocorrelation
    autocorrelation = fftconvolve(fluctuations, fluctuations[::-1], mode='same')
    # To compare with the desired autocorrelation we need to rescale and shift
    autocorrelation /= autocorrelation.max()
    autocorrelation = np.fft.fftshift(autocorrelation)

    # This is the correlation we used to design our filter.
    correlation = correlation_spherical_wave(tau(ntaps, fs), correlation_time)

    # We only compare one side of the spectrum.
    autocorrelation = autocorrelation[:ntaps//2]
    correlation = correlation[:ntaps//2]

    # We perform a linear regression. We only pick values close to tau=0, because
    # at higher values there is relatively more noise.
    regression = linregress(autocorrelation[:50], correlation[:50])
    print(regression)
    assert np.abs(regression.slope - 1) < 0.05
    assert np.abs(regression.intercept) < 0.05
    assert np.abs(regression.rvalue) > 0.95
