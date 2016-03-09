import scintillations
from scintillations.stream import *

from acoustics import Signal

@pytest.fixture(params=[1.0, 10.0, 100.0])
def speed(request):
    return request.param

@pytest.fixture(params=[1.0, 10.0])
def correlation_length(request):
    return request.param

@pytest.fixture(params=[None, 100, 200])
def seed(request):
    return request.param

def test_autocorrelation(speed, correlation_length, seed):

    ntaps = 8192
    factor = 5.0
    correlation_time = correlation_length / speed
    state = np.random.RandomState(seed=seed)
    window = None

    # Duration of sequence.
    fs = factor / correlation_time
    nsamples = ntaps

    #fluctuations = generate_gaussian_fluctuations_standard(nsamples, ntaps, fs, correlation_time, state=state, window=window)


@pytest.fixture(params=[100, 1000.])
def distance(request):
    return request.param

@pytest.fixture(params=[True, False])
def include_saturation(request):
    return request.param

@pytest.fixture(params=[100.0, 1000.])
def frequency(request):
    return request.param

def test_variance(speed, correlation_length, seed, distance, frequency, include_saturation):
    """Test variance of fluctuations.

    In this test we generate fluctuations and apply these to a sine of the specified frequency.

    """

    #duration = 1200.
    #fs = 8000.
    #nsamples = int(fs*duration)
    #ntaps = 8192
    #window = None
    #state = np.random.RandomState(seed)
    #mean_mu_squared = 3.0e-6
    #soundspeed = 343.
    #wavenumber = 2.*np.pi*frequency/soundspeed

    #modulated = (signal, fs, correlation_length, speed, distance, soundspeed, mean_mu_squared, ntaps=8192,
                 #nfreqs=100, window=None, include_saturation=False, state=None, factor=5.0,
                 #include_amplitude=True, include_phase=True)

    #modulated = Signal(modulated.take(nsamples).toarray())

    #amplitude = modulated.amplitude_envelope()
    #phase = modulated.instantaneous_



    #expected_logamp_var = variance_gaussian(distance, wavenumber, correlation_length, mean_mu_squared,
                                                             #include_saturation=include_saturation)
    #expected_phase_var  = variance_gaussian(distance, wavenumber, correlation_length, mean_mu_squared)

    #assert np.abs( logamp.var() - expected_logamp_var ) < 0.06
    #assert np.abs( phase.var() - expected_phase_var ) < 0.06





def test_compare_with_invariant():
    """Test whether the code in :mod:`variant` gives the same result in the case of time-invariant as :mod:`invariant`.

    """

    duration = 10.0
    ntaps = 8192
    correlation_length = 1.0
    speed = 1.
    factor = 5.0
    correlation_time = correlation_length / speed
    state = np.random.RandomState(seed=100)
    window = None

    def generate_variant():

        fluctuations = scintillations.variant.generate_gaussian_fluctuations()
        return fluctuations

    def generate_invariant():

        fluctuations = scintillations.invariant.generate_gaussian_fluctuations()

        return fluctuations


    fluctuations_variant = generate_variant()
    fluctuations_invariant = generate_invariant()
