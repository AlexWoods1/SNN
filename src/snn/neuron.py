"""
neuron.py
==========
Purpose:    Implements neuron models and spike generation.
Part of:    SNN v{version}
"""

from __future__ import annotations


class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) Neuron Model.

    Models a single neuron's membrane potential dynamics and spike
    generation. The LIF model approximates biological neuron behavior
    by integrating input current and 'leaking' charge over time.

    Membrane potential dynamics (continuous form):
        τ_m · dV/dt = -(V - V_rest) + R_m · I(t)

    Euler-discretised for simulation (timestep dt):
        V[t+1] = V[t] + (dt / τ_m) · (-(V[t] - V_rest) + R_m · I[t])

    Spike condition:
        if V[t] >= V_thresh:
            emit spike
            V[t] = V_reset

    Parameters
    ----------
    tau_m    : float  — membrane time constant (ms), controls leak rate
    r_m      : float  — membrane resistance (MΩ), scales input current
    v_rest   : float  — resting membrane potential (mV)
    v_thresh : float  — spike threshold potential (mV)
    v_reset  : float  — post-spike reset potential (mV)
    dt       : float  — simulation timestep (ms)
    """

    def __init__(
        self,
        tau_m: float,
        r_m: float,
        v_rest: float,
        v_thresh: float,
        v_reset: float,
        dt: float,
    ) -> None:
        self.tau_m = tau_m
        self.r_m = r_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.dt = dt
        self.v = v_rest
        self.spike_history = []

    def step(self, current: float) -> bool:
        """
        Advance the neuron by one timestep given an input current.

        Applies the discrete time LIF equation to update membrane
        potential, checks the spike threshold, resets if spiked,
        and records spike history.

        Parameters
        ----------
        current : float — injected current I(t) in nA

        Returns
        -------
        bool — True if a spike was emitted this timestep, else False
        """

        # 1. Apply Euler step:
        self.v += (self.dt / self.tau_m) * (
            -(self.v - self.v_rest) + self.r_m * current
        )
        # 2. Check spike threshold:

        # 3. Append spike to history
        if self.v >= self.v_thresh:
            spike = True
            self.v = self.v_reset

        else:
            spike = False
        self.spike_history.append(spike)
        return spike

    def reset(self) -> None:
        """
        Reset neuron state to initial conditions.

        Resets membrane potential to v_rest and clears spike history.
        Useful between simulation runs.
        """
        self.v = self.v_rest
        self.spike_history = []

    @property
    def spike_count(self) -> int:
        """Return the total number of spikes emitted so far."""
        return sum(self.spike_history)

    @property
    def firing_rate(self) -> float:
        """
        Return mean firing rate in Hz.

        Calculated as:
            rate = spike_count / (len(spike_history) * dt / 1000)

        Converts dt from ms to seconds for Hz units.
        """
        if not self.spike_history:
            return 0.0
        return self.spike_count / (len(self.spike_history) * self.dt / 1000)
