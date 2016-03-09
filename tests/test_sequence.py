import pytest
from scintillations.sequence import *
from scipy.stats import linregress
from acoustics import Signal

#def parameters(p, name):
    #d = {}
    #d['params'] = p
    #d['ids'] = ["{}:{}".format(name, i) for i in p]


@pytest.fixture(params=[20.0, 70.0], ids=["v = 20 m/s", "v = 70 m/s"])
def speed(request):
    return request.param

@pytest.fixture(params=[1.0, 15.0], ids=["L = 1 m", "L = 15 m"])
def correlation_length(request):
    return request.param

@pytest.fixture(params=[None, 100, 200])
def seed(request):
    return request.param

@pytest.fixture
def state(seed):
    return np.random.RandomState(seed)

@pytest.fixture
def ntaps():
    return 8192

def test_autocorrelation(speed, correlation_time, state, fb, ntaps, fluctuations_standard):
    """Test whether the autocorrelation of the generated fluctuations matches with
    the desired autocorrelation that is used in the filter design process.
    """
    fs = fb
    nsamples = ntaps
    fluctuations = fluctuations_standard

    # We expect to obtain fluctuations with variance 1.0
    assert np.allclose(fluctuations.var(), 1.0, rtol=0.1)

    # We now calculate the autocorrelation
    autocorrelation = fftconvolve(fluctuations, fluctuations[::-1], mode='same')
    # To compare with the desired autocorrelation we need to recorrelation_length and shift
    autocorrelation /= autocorrelation.max()
    autocorrelation = np.fft.fftshift(autocorrelation)

    # This is the correlation we used to design our filter.
    correlation = correlation_spherical_wave(tau(ntaps, fs), correlation_time)

    # We only compare one side of the spectrum.
    autocorrelation = autocorrelation[:ntaps//2]
    correlation = correlation[:ntaps//2]

    # We perform a linear regression. We only pick values close to tau=0, because
    # at higher values there is relatively more noise.
    regression = linregress(autocorrelation[:20], correlation[:20])

    assert np.abs(regression.slope - 1) < 0.15
    assert np.abs(regression.intercept) < 0.15
    assert np.abs(regression.rvalue) > 0.97


@pytest.fixture(params=[100, 2000.], ids=["d = 100 m","d = 2000 m"])
def distance(request):
    return request.param

@pytest.fixture(params=[True], ids=["With saturation"])
def include_saturation(request):
    return request.param

@pytest.fixture(params=[True, False], ids=["With amp", "Without amp"])
def include_amplitude(request):
    return request.param

@pytest.fixture(params=[True, False], ids=["With phase", "Without phase"])
def include_phase(request):
    return request.param

@pytest.fixture(params=[100.0, 450.], ids=["f = 100 Hz ","f = 450 Hz"])
def frequency(request):
    return request.param

@pytest.fixture
def duration():
    return 30

@pytest.fixture(params=[8000., 44100.], ids=["fs = 8000 Hz","fs = 44100 Hz"])
def fs(request):
    return request.param

@pytest.fixture
def nsamples(fs, duration):
    return int(fs*duration)

@pytest.fixture
def window():
    return None

@pytest.fixture
def correlation_time(correlation_length, speed):
    return correlation_length / speed

@pytest.fixture
def factor():
    return 5.0

@pytest.fixture
def fb(factor, correlation_time):
    return factor / correlation_time

@pytest.fixture
def soundspeed():
    return 343.

@pytest.fixture
def mean_mu_squared():
    return 3.0e-6

@pytest.fixture
def wavenumber(frequency, soundspeed):
    return  2.*np.pi*frequency/soundspeed

@pytest.fixture
def variance_expected_logamp(distance, wavenumber, correlation_length, mean_mu_squared, include_saturation):
    return variance_gaussian(distance, wavenumber, correlation_length, mean_mu_squared, include_saturation)

@pytest.fixture
def variance_expected_phase(distance, wavenumber, correlation_length, mean_mu_squared):
    return variance_gaussian(distance, wavenumber, correlation_length, mean_mu_squared, False)

@pytest.fixture
def fluctuations_standard(nsamples, ntaps, fb, correlation_time, state, window):
    fluctuations = generate_gaussian_fluctuations_standard(nsamples, ntaps, fb, correlation_time, state, window)
    return fluctuations

@pytest.fixture
def times(nsamples, fs):
    return Signal(np.arange(nsamples) / fs, fs)

@pytest.fixture
def tone(times, frequency):
    return np.sin(2.*np.pi*frequency*times)

@pytest.fixture
def logamp_and_phase(nsamples, fb, ntaps, correlation_length, speed, frequency,
                     soundspeed, distance, mean_mu_squared, include_saturation, state, window):

    return generate_fluctuations_logamp_and_phase(nsamples, fb, ntaps, correlation_length,
                                                  speed, frequency, soundspeed, distance, mean_mu_squared,
                                                  include_saturation, state, window)

@pytest.fixture
def logamp(logamp_and_phase):
    return logamp_and_phase[0]

@pytest.fixture
def phase(logamp_and_phase):
    return logamp_and_phase[1]

@pytest.fixture
def modulated_tone(tone, fs, frequency, fb, correlation_length, speed, distance, soundspeed,
                   mean_mu_squared, ntaps, window, include_saturation, state, include_amplitude, include_phase):
    modulated = modulate_tone(tone, fs, frequency, fb, correlation_length, speed, distance, soundspeed,
                              mean_mu_squared, ntaps, window, include_saturation, state, include_amplitude, include_phase)
    return Signal(modulated, fs)

@pytest.fixture
def modulated_logamp(modulated_tone):
    return np.log(modulated_tone.amplitude_envelope())

@pytest.fixture
def modulated_phase(modulated_tone, tone):
    return modulated_tone.instantaneous_phase().unwrap() - tone.instantaneous_phase().unwrap()

@pytest.fixture
def modulated_variance_logamp(modulated_logamp):
    return modulated_logamp.var()

@pytest.fixture
def modulated_variance_phase(modulated_phase):
    return modulated_phase.var()

RTOL_VARIANCE = 0.1
#ATOL_VARIANCE = 0.1

def test_variance_logamp(variance_expected_logamp, logamp):
    """Test whether sequence produced by :func:`generate_fluctuations_logamp_and_phase` produces correct variance."""
    assert np.allclose(variance_expected_logamp, logamp.var(), rtol=RTOL_VARIANCE)

def test_variance_phase(variance_expected_phase, phase):
    """Test whether sequence produced by :func:`generate_fluctuations_logamp_and_phase` produces correct variance."""
    assert np.allclose(variance_expected_phase, phase.var(), rtol=RTOL_VARIANCE)

# Modulated tone still fails...

def test_variance_modulated_tone_logamp(variance_expected_logamp, modulated_variance_logamp, include_amplitude, include_phase):
    """Test whether the scintillations produced by :func:`modulate_tone` have the correct variance."""
    if include_phase:
        pytest.skip("Phase fluctuations can affect amplitude fluctuations.")

    if include_amplitude:
        assert np.allclose(variance_expected_logamp, modulated_variance_logamp, rtol=RTOL_VARIANCE)
    else:
        pytest.skip("Neither amplitude nor phase fluctuations.")

def test_variance_modulated_tone_phase(variance_expected_phase, modulated_variance_phase, include_amplitude, include_phase):
    """Test whether the scintillations produced by :func:`modulate_tone` have the correct variance."""
    if include_amplitude:
        pytest.skip("Amplitude fluctuations can affect phase fluctuations.")

    if include_phase:
        assert np.allclose(variance_expected_phase, modulated_variance_phase, rtol=RTOL_VARIANCE)
    else:
        pytest.skip("Neither amplitude nor phase fluctuations.")

