"""
synapse.py
==========
Purpose:    Implements synaptic connections and current transmission.
Part of:    SNN v{version}
"""

from __future__ import annotations
from snn.neuron import LIFNeuron


class Synapse:
    """
    Synaptic connection between two LIF neurons.

    Models the transmission of a spike from a pre-synaptic neuron to a
    post-synaptic neuron as a weighted current pulse, with an optional
    fixed transmission delay.

    Synaptic current (instantaneous, no delay):
        I_syn[t] = weight * spike[t]

    With delay (delay_steps = round(delay_ms / dt)):
        I_syn[t + delay_steps] = weight * spike[t]

    The delay is implemented as a circular buffer of length
    delay_steps + 1. Each call to transmit() shifts the buffer,
    loads any new spike current at the back, and returns the current
    due at the front.

    Parameters
    ----------
    pre     : LIFNeuron — the source (pre-synaptic) neuron
    post    : LIFNeuron — the target (post-synaptic) neuron
    weight  : float     — synaptic weight, scales transmitted current (nA)
    delay   : float     — transmission delay in ms (default 0.0)
    dt      : float     — simulation timestep in ms, must match neurons
    """

    def __init__(
        self,
        pre: LIFNeuron,
        post: LIFNeuron,
        weight: float,
        delay: float = 0.0,
        dt: float = 0.1,
    ) -> None:
        self.pre: LIFNeuron = pre
        self.post: LIFNeuron = post
        self.weight: float = weight
        self.delay: float = delay
        self.dt: float = dt
        self._delay_steps: int = int(round(delay / dt))
        self._buffer: list[float] = [0.0] * (self._delay_steps + 1)

    def transmit(self, spike: bool) -> float:
        """
        Advance the synapse by one timestep.

        Reads the current due at the front of the buffer, shifts the
        buffer left by one position, then loads the new spike current
        at the back. Returns the current to be injected into post
        this timestep.

        Buffer shift (each timestep):
            current  = buffer[0]
            buffer   = buffer[1:] + [weight * spike]
            return current

        Parameters
        ----------
        spike : bool — whether the pre-synaptic neuron fired this step

        Returns
        -------
        float — synaptic current to inject into post this timestep
        """
        current: float = self._buffer[0]
        del self._buffer[0]
        self._buffer.append(self.weight if spike else 0.0)
        return current

    @property
    def delay_steps(self) -> int:
        """Return the delay expressed as an integer number of time steps."""
        return self._delay_steps
