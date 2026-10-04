"""Qblox Ramsey — supplies only ``probe()``.

Same one-method pattern as resonator spectroscopy, on a completely different physics
experiment: X90 — idle(t) — X90 — Measure, looping the idle time. Parameters, the
decaying-cosine fit, T2*/detuning extraction, and the drive_freq_hz writeback are all
inherited from ``scqo.experiments.QubitRamsey``.

The artificial detuning rides as a phase ramp on the second pi/2; both its SPELLING and
its SIGN are load-bearing and were each silently wrong before 2026-08-01 (see probe()).
``tests/test_ramsey_detuning.py`` pins them.

ONE LATENT LIMIT worth knowing before raising ``num_points``: while the drive's
``drag_beta`` is 0 every point shares one normalized waveform and the phase rides in the
AWG gains, so the ramp is free. With a calibrated non-zero beta each DISTINCT phase mints
its own I/Q waveform pair, and the sequencer's 16384-sample budget becomes the ceiling —
at a 80 ns pi/2, 101 points fits (16160 samples) and 103 does not (16480).
"""

from __future__ import annotations

from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitRamsey
from scqo.requirements import Requirement

from scqo_qblox.experiments._reset import add_reset
from scqo_qblox.experiments._state import measure_kwargs


@register
class QbloxQubitRamsey(QubitRamsey):
    """Build a multiplexed Ramsey Schedule for a Qblox cluster."""

    #: what only this backend consumes, on top of the neutral requirements
    requires: ClassVar[tuple[Requirement, ...]] = (
        *QubitRamsey.requires,
        Requirement("pi_amp", "both pi/2 pulses are Rxy(theta=90), played at half the pi amplitude"),
    )
    #: true of THIS probe only; scqo shows them in `scqo run qubit_ramsey --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "plays X90 first where the declared sequence has y90, so the fringe starts a quarter "
        "turn later; the fitted frequency and T2* do not depend on it (SCQO BACKLOG I19)",
        "both pi/2 pulses are Rxy(theta=90) at half of pi_amp; pi_amp_x90 is not realized here",
        "the virtual detuning is the phase of the second Rxy; the same phase passed to X90 "
        "would be dropped by the compiler",
        "with a non-zero drag_beta every distinct phase needs its own waveform: at an 80 ns "
        "pi/2 pulse 101 points fit the sequencer's 16384 samples and 103 do not",
    )

    #: readout is held at the calibrated point for the whole run and the Reset is
    #: a genuine state reset, so reset_method='active' is valid here (_reset.py).
    #: Ramsey is also the SENSITIVE test of the settle idle: residual readout
    #: photons Stark-shift the first X90 and surface as a fitted-detuning error.
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, Rxy, X90
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        idle_ns = self.sweep_axes["idle_time_ns"]
        reps = self.params.num_averages
        detuning = self.params.frequency_detuning_hz

        schedule = Schedule("ramsey_multiplexed")
        for qubit_name in self.params.targets:
            acq = measure_kwargs(self, qubit_name)  # {} or the thresholded protocol
            sub = Schedule(f"ramsey_{qubit_name}")
            with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                with sub.loop(
                    linspace(idle_ns[0] * 1e-9, idle_ns[-1] * 1e-9, idle_ns.size, dtype=DType.TIME)
                ) as tau:
                    # First pi/2; the artificial detuning is a phase ramp on the SECOND
                    # pi/2 that grows with the idle time, producing the Ramsey fringe.
                    add_reset(sub, self, qubit_name)
                    sub.add(X90(qubit_name))
                    sub.add(IdlePulse(tau))
                    # It MUST be Rxy(phi=...), never X90(..., phase=...). `phase` is not an
                    # Rxy factory kwarg, so it lands in gate_info["device_overrides"] and
                    # circuit_to_device drops it (`if key in factory_kwargs`, line ~762) —
                    # silently, on a schedule that still compiles. This probe spent its
                    # whole life requesting a detuning the instrument never applied. `phi`
                    # IS a gate_info factory kwarg (transmon_element.py
                    # gate_info_factory_kwargs=["theta","phi"]), and Rxy(theta=90) is the
                    # same pulse as X90 (amplitude = amp180*theta/180) — only the phase
                    # differs.
                    #
                    # The sign is NEGATIVE, and that is not a typo. A programmed phase is a
                    # carrier-phase ADVANCE (the NCO emits I*cos(wt) - Q*sin(wt) =
                    # Re[(I+iQ)e^{iwt}], backends/qblox/operation_handling/pulses.py), so it
                    # runs the OPPOSITE way to the free precession of a qubit sitting above
                    # its drive. Only a backward ramp makes the observed fringe
                    # (applied + err) — the quantity scqo's shared estimate() subtracts
                    # `applied` from. A positive ramp gives |applied - err|, and then every
                    # accepted update walks the drive to the absorbing point err = 2*applied,
                    # where detuning_error_hz reads exactly 0.0 and the fit looks perfect.
                    sub.add(Rxy(theta=90.0, phi=-360.0 * detuning * tau, qubit=qubit_name))
                    sub.add(
                        Measure(
                            qubit_name,
                            coords={f"tau_{qubit_name}": tau},
                            acq_channel=f"S_21_{qubit_name}",
                            **acq,
                        )
                    )
                    sub.add(IdlePulse(4e-9))
            schedule.add(sub)
        return schedule
