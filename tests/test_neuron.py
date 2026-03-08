"""
test_neuron.py
==============
Purpose:    Tests for LIFNeuron model.
Part of:    SNN v{version}
"""
import numpy as np
import pytest

from snn.neuron import LIFNeuron


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
    # Return a LIFNeuron with the above parameters
    return LIFNeuron(10, 10, -65, -55, -70, 0.1)


# --- Initialisation Tests ---

def test_initial_membrane_potential():
    """Membrane potential should equal v_rest on initialisation."""
    neuron = lif_default()
    assert neuron.v == -65.0

def test_initial_spike_history_empty():
    """Spike history should be empty on initialisation."""
    neuron = lif_default()
    assert neuron.spike_history == []


def test_initial_spike_count_zero():
    """Spike count should be zero on initialisation."""
    neuron = lif_default()
    assert neuron.spike_count == 0


# --- Step Tests ---

def test_step_no_current_no_spike():
    """
    With zero input current, membrane potential should decay toward
    v_rest and no spike should be emitted.
    """
    neuron = lif_default()
    for _ in range(10):
        neuron.step(0)
    assert neuron.spike_count == 0
    assert neuron.v <= -65

def test_step_returns_bool():
    """step() should always return a bool."""
    neuron = lif_default()
    assert type(neuron.step(0)) == bool

def test_step_large_current_causes_spike():
    """
    With a sufficiently large input current, the neuron should
    eventually reach v_thresh and emit a spike.
    """
    neuron = lif_default()
    spike = False
    for _ in range(1000):
        spike = neuron.step(10.0)
        if spike:
            break
    assert spike is True

def test_step_spike_resets_membrane_potential():
    """
    After a spike, membrane potential should be reset to v_reset,
    not remain at or above v_thresh.
    """
    neuron = lif_default()
    neuron.step(1000)
    assert neuron.v == -70

def test_step_appends_to_spike_history():
    """Each call to step() should append exactly one entry to spike_history."""
    neuron = lif_default()
    neuron.step(10)
    neuron.step(10)
    assert len(neuron.spike_history) == 2

def test_step_spike_recorded_in_history():
    """A spike event should be recorded as True in spike_history."""
    neuron = lif_default()
    neuron.step(1000)
    assert neuron.spike_history[0] is True


# --- Reset Tests ---

def test_reset_restores_membrane_potential():
    """After reset(), membrane potential should return to v_rest."""
    neuron = lif_default()
    neuron.step(1000)
    neuron.reset()
    assert neuron.v == -65.0


def test_reset_clears_spike_history():
    """After reset(), spike_history should be empty."""
    neuron = lif_default()
    neuron.step(1000)
    neuron.reset()
    assert neuron.spike_history == []



# --- Property Tests ---

def test_spike_count_reflects_history():
    """spike_count should equal the number of True values in spike_history."""
    neuron = lif_default()
    neuron.step(1000)
    neuron.step(0)
    neuron.step(1000)
    assert neuron.spike_count == 2

def test_firing_rate_zero_before_steps():
    """firing_rate should return 0.0 if no steps have been taken."""
    neuron = lif_default()
    assert neuron.firing_rate == 0.0

def test_firing_rate_units():
    """
    firing_rate should return a value in Hz.
    For a neuron spiking every timestep over 1 second, rate should be 1000 Hz
    given dt=1.0ms.
    """
    neuron = lif_default()  # dt = 0.1ms
    # Manually set spike history: 100 spikes over 1000 steps
    # 1000 steps * 0.1ms = 100ms = 0.1s
    # Expected rate = 100 / 0.1 = 1000 Hz
    neuron.spike_history = [True] * 100 + [False] * 900
    assert neuron.firing_rate == pytest.approx(1000.0)
