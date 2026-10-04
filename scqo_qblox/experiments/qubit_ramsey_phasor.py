"""Qblox phasor Ramsey — supplies only ``probe()``.

X90 — idle(t) — ShiftClockPhase(phi) — X90 — Measure, with the closing pulse's
phase swept through a full turn at every idle time. Parameters, the lock-in, the
stretched-exponential envelope fit and the writeback are all inherited from
``scqo.experiments.QubitRamseyPhasor``.

TWO VENDOR CONSTRAINTS SHAPE THIS PROBE, and they pull in opposite directions:

THE FRAME AXIS is a realtime PHASE loop driving ``ShiftClockPhase`` (one of the
three types a realtime loop may vary — voltage_offset, phase_shift, frequency),
with ``ResetClockPhase`` opening every point. It CANNOT be spelled
``Rxy(theta=90, phi=<loop var>)`` the way plain ``qubit_ramsey`` spells its
detuning ramp: a variable ``phi`` does not compile inside a realtime loop. Nor
may it be ``X90(..., phase=...)`` — ``phase`` is not an Rxy factory kwarg, so
``circuit_to_device`` drops it SILENTLY on a schedule that still compiles.

THE IDLE AXIS is Python-unrolled, because it is LOG-spaced and a log axis is not
``linspace``-able. Unrolling is what costs instructions, so the ceiling below is
on the number of idle points — NOT, as in the Ramsey cryoscope, on the longest
duration. That probe unrolls every nanosecond up to its maximum and so caps at
512 ns; this one unrolls only its ~60 log points and can therefore reach
hundreds of microseconds of idle in the same budget.

NO ARTIFICIAL DETUNING, and so no phase ramp: plain ``qubit_ramsey`` must run
``Rxy(phi=...)`` BACKWARD in proportion to the idle time, and a positive ramp
walks the drive to an absorbing point where the fitted error reads exactly 0.0.
Here the frame is the swept tomography axis itself — there is no ramp, no
proportionality, and no sign to get wrong.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from scqo import register
from scqo.experiments import QubitRamseyPhasor
from scqo.experiments.qubit_ramsey_phasor import FRAME_AXIS, IDLE_AXIS
from scqo.requirements import Requirement

from scqo_qblox.experiments._requires import HALF_PI_AMPLITUDE
from scqo_qblox.experiments._reset import add_reset
from scqo_qblox.experiments._state import measure_kwargs

#: Ceiling on the PYTHON-UNROLLED idle points. Each one mints its own block of
#: sequencer instructions (the frame loop inside it is realtime and costs
#: nothing per iteration), so this — not the idle duration — is what the
#: instruction budget actually bounds. The Ramsey cryoscope demonstrates ~512
#: unrolled blocks compiling; half that leaves generous headroom for the reset
#: and readout operations each block also carries.
MAX_IDLE_POINTS = 256


def validate_inputs(idle_ns: np.ndarray) -> None:
    """Refuse a sweep whose unrolled length would overrun the sequencer."""
    realized = int(np.asarray(idle_ns).size)
    if realized > MAX_IDLE_POINTS:
        raise ValueError(
            f"qubit_ramsey_phasor unrolls each idle point into its own block, and "
            f"{realized} of them exceeds the Qblox backend's {MAX_IDLE_POINTS}-point "
            f"ceiling. Lower num_points (the realized axis is already shorter than "
            f"num_points — log points collide on the 4 ns grid and are de-duplicated). "
            f"The frame axis is free: it is a realtime loop, so num_frames costs no "
            f"instructions."
        )


@register
class QbloxQubitRamseyPhasor(QubitRamseyPhasor):
    """Build a multiplexed phasor-Ramsey Schedule for a Qblox cluster."""

    #: what only this backend consumes, on top of the neutral requirements
    requires: ClassVar[tuple[Requirement, ...]] = (
        *QubitRamseyPhasor.requires, HALF_PI_AMPLITUDE)
    #: true of THIS probe only; scqo shows them in `scqo run qubit_ramsey_phasor --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "both pi/2 pulses are played at half of pi_amp; pi_amp_x90 is not realized here",
        "the frame axis is a shift of the drive clock's phase before the second pulse; the "
        "phase is reset at the start of every point",
    )

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import (
            IdlePulse,
            Measure,
            ResetClockPhase,
            ShiftClockPhase,
            X90,
        )
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        idle_ns = np.asarray(self.sweep_axes[IDLE_AXIS], dtype=float)
        frames = np.asarray(self.sweep_axes[FRAME_AXIS], dtype=float)
        reps = self.params.num_averages
        validate_inputs(idle_ns)

        # frame turns -> ShiftClockPhase degrees. The axis is endpoint-exclusive,
        # and rebuilding it from its own endpoints reproduces it exactly (the
        # same trick the time axes rely on).
        frame_deg = 360.0 * frames

        schedule = Schedule("qubit_ramsey_phasor")
        for qubit_name in self.params.targets:
            acq = measure_kwargs(self, qubit_name)  # {} or the thresholded protocol
            drive_clock = f"{qubit_name}.01"
            sub = Schedule(f"ramsey_phasor_{qubit_name}")
            with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                # idle OUTER (Python-unrolled, ascending — a log axis is not
                # linspace-able), frame INNER (realtime PHASE loop): the flat bin
                # order matches the canonical sweep-axes order (idle_time_ns, frame).
                for t_ns in (float(v) for v in idle_ns):
                    with sub.loop(linspace(frame_deg[0], frame_deg[-1],
                                           frame_deg.size, dtype=DType.PHASE)) as phi:
                        add_reset(sub, self, qubit_name)
                        sub.add(ResetClockPhase(clock=drive_clock))
                        sub.add(IdlePulse(4e-9))
                        sub.add(X90(qubit_name))
                        # the free evolution IS the swept quantity here, so —
                        # unlike the cryoscope — it is deliberately NOT padded to
                        # a constant window: the decay across it is the signal.
                        sub.add(IdlePulse(t_ns * 1e-9))
                        # positive tomography frame — no ramp, see the docstring
                        sub.add(ShiftClockPhase(phase_shift=phi, clock=drive_clock))
                        sub.add(IdlePulse(4e-9))
                        sub.add(X90(qubit_name))
                        sub.add(Measure(
                            qubit_name,
                            coords={f"idle_time_{qubit_name}": t_ns,
                                    f"frame_{qubit_name}": phi},
                            acq_channel=f"S_21_{qubit_name}",
                            **acq,
                        ))
                        sub.add(IdlePulse(4e-9))
            schedule.add(sub)
        return schedule
