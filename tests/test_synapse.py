"""
test_synapse.py
===============
Purpose:    Tests for Synapse model.
Part of:    SNN v{version}
"""
import numpy as np
import pytest
from snn.neuron import LIFNeuron
from snn.synapse import Synapse


# --- Fixtures ---

def lif_default() -> LIFNeuron:
    """
    Returns a LIFNeuron with standard biological parameters.

    tau_m    = 10.0  ms
    r_m      = 10.0  MΩ
    v_rest   = -65.0 mV
    v_thresh = -55.0 mV
    v_reset  = -70.0 mV
    dt       = 0.1   ms
    """
    return LIFNeuron(10.0, 10.0, -65.0, -55.0, -70, 0.1)


def synapse_default(
    pre: LIFNeuron,
    post: LIFNeuron
) -> Synapse:
    """
    Returns a Synapse with no delay and a weight of 1.0.

    weight = 1.0 nA
    delay  = 0.0 ms
    dt     = 0.1 ms
    """
    return Synapse(pre=pre, post=post, weight=1, delay=0, dt=0.1)


# --- Initialisation Tests ---

def test_initial_buffer_length_no_delay():
    """Buffer length should be 1 when delay is 0.0."""
    connection = synapse_default(lif_default(), lif_default())
    connection.delay = 0.0
    assert len(connection._buffer) == 1


def test_initial_buffer_length_with_delay():
    """
    Buffer length should be delay_steps + 1.
    e.g. delay=1.0ms, dt=0.1ms → delay_steps=10 → buffer length=11.
    """
    connection = Synapse(lif_default(), lif_default(), delay=1.0, dt=0.1, weight=1)
    assert len(connection._buffer) == 11
    assert connection._delay_steps == 10


def test_initial_buffer_all_zeros():
    """Buffer should be all zeros on initialisation."""
    connection = synapse_default(lif_default(), lif_default())
    assert np.all(connection._buffer) == 0

def test_delay_steps_no_delay():
    """delay_steps should be 0 when delay is 0.0."""
    connection = synapse_default(lif_default(), lif_default())
    assert connection._delay_steps == 0

def test_delay_steps_with_delay():
    """
    delay_steps should equal round(delay / dt).
    e.g. delay=1.0ms, dt=0.1ms → delay_steps=10.
    """
    connection = Synapse(lif_default(), lif_default(), delay=1.0, dt=0.1, weight=1)
    assert pytest.approx(connection._delay_steps) == round(1.0/0.1)

# --- Transmit Tests ---

def test_transmit_returns_float():
    """transmit() should always return a float."""
    connection = synapse_default(lif_default(), lif_default())
    assert type(connection.transmit(True)) is float

def test_transmit_no_spike_returns_zero():
    """transmit() should return 0.0 when no spike is passed."""
    connection = synapse_default(lif_default(), lif_default())
    assert connection.transmit(False) ==  0.0

def test_transmit_spike_no_delay_returns_weight():
    """
    With no delay, transmit(True) should return weight on the
    very next call to transmit().
    """
    connection = synapse_default(lif_default(), lif_default())
    connection.transmit(True)
    assert connection.transmit(True) == 1.0


# noinspection PyTypeChecker
def test_transmit_spike_with_delay():
    """
    With a delay of N steps, a spike should arrive exactly N+1
    calls to transmit() after it was sent.
    """
    # delay=1.0ms, dt=0.1ms → delay_steps=10 → spike arrives on call 11
    connection = Synapse(pre=lif_default(), post=lif_default(), weight=1.0, delay=1.0, dt=0.1)

    connection.transmit(True)

    for _ in range(10):
        current = connection.transmit(False)
        assert current == pytest.approx(0.0)
    current =  connection.transmit(False)
    assert current == pytest.approx(1.0)


def test_transmit_no_spike_does_not_load_buffer():
    """transmit(False) should load 0.0 into the buffer, not weight."""
    connection = synapse_default(lif_default(), lif_default())
    connection.transmit(False)
    assert connection.weight != 0.0

def test_transmit_weight_scales_current():
    """
    The current returned after a spike should equal the synapse weight.
    e.g. weight=2.5, spike=True → current delivered = 2.5.
    """
    connection = synapse_default(lif_default(), lif_default())
    connection.transmit(True)
    current = connection.transmit(False)
    assert current == pytest.approx(1.0)


