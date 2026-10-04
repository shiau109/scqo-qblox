"""Qblox Ramsey cryoscope — supplies only ``probe()``.

Phase tomography of the flux line's step response: ``x90`` — flux pulse of swept
DURATION (every nanosecond, riding RELATIVE on ``idle_flux``) — constant total
free evolution — frame-swept closing ``x90`` — measure. Parameters, the
sum-of-exponentials fit and the ``distortion_amp``/``distortion_tau_s`` fact
writeback are inherited from ``scqo.experiments.QubitRamseyCryoscope``.

THE 1-NS DURATION AXIS, without QM's baked waveforms. Three compiler facts
(measured 2026-08-24 on qblox-scheduler 1.0.0b6) force the shape:

* every same-sequencer gap compiles to a ``wait`` that must be 0 or >= 4 ns, and
* ``VoltageOffset`` / ``play`` each consume an intrinsic 4 ns (``upd_param`` /
  the minimum play window) before that wait starts, so
* an offset pair can express only durations n with ``n - 4 in {0} + [4, inf)``,
  and a per-duration waveform of n samples only ``n <= 4`` the same way —
  while a full per-duration waveform bank (240 pulses of 1..240 samples) blows
  the 16384-sample waveform memory.

So each duration is COMPOSED: ``n = 4k + r`` plays as an offset segment of
``4k`` ns (``VoltageOffset`` up, ``IdlePulse(4k)``, ``VoltageOffset`` back) with
the r-nanosecond REMAINDER as a tiny ``SquarePulse`` waveform appended at the
same timestamp the offset returns — the play's latched-parameter update takes
the offset down exactly as the waveform takes over, so the line sits at
``idle + amp`` for precisely n contiguous nanoseconds. Only three remainder
waveforms ever exist (r = 1, 2, 3 — six samples total), and every flux-sequencer
wait is a multiple of 4 by construction. The pulse is LEFT-aligned after the
opening x90 exactly like the QM sequence, so the down-step settling tail stays
inside the (constant) free-evolution window — the estimator's model assumes it.

THE FRAME AXIS is a realtime PHASE loop driving ``ShiftClockPhase`` (one of the
three expression-safe operands) before a fixed closing ``X90``, with
``ResetClockPhase`` opening every point — the mirror of QM's
``frame_rotation_2pi`` + ``reset_frame``. The sign is POSITIVE: this is phase
TOMOGRAPHY (the frame is the swept abscissa, not a virtual detuning), and a
programmed phase advance has the same handedness on both vendors, so the fringe
is ``cos(2*pi*frame - phase)`` exactly as ``simulate()`` writes it. Contrast
``qubit_ramsey``, whose detuning RAMP is negative for a physics reason that does
not apply here.

INSTRUCTION CEILING: one Q1ASM block per (duration x point) is Python-unrolled,
and the READOUT sequencer (QRM: 12288 instructions) fills first — measured:
max_duration_ns=512 compiles, 640 does not. The probe refuses a longer axis by
name; the microsecond tails belong to qubit_spectroscopy_cryoscope anyway.
"""

from __future__ import annotations

from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitRamseyCryoscope
from scqo.experiments._capabilities import flux_anchor_v
from scqo.experiments.qubit_ramsey_cryoscope import DURATION_AXIS, FRAME_AXIS
from scqo.requirements import Requirement

from ._flux_limits import check_flux_pulse_relative, to_dac_fraction
from ._requires import HALF_PI_AMPLITUDE
from ._reset import add_reset
from ._state import measure_kwargs
from ._vendor import vendor_element

#: Largest realizable duration axis: one unrolled block per nanosecond fills the
#: readout sequencer's 12288-instruction Q1ASM budget first (512 compiles at
#: ~11.8k instructions, 640 exceeds it — measured on the fixture, 16 frames).
MAX_DURATION_NS = 512


def duration_split(n: int) -> tuple[int, int]:
    """``n = 4k + r``: the offset-segment length ``4k`` and the remainder ``r``
    played as a tiny waveform. Pure — the composed-pulse invariant lives here."""
    r = n % 4
    return n - r, r


def validate_inputs(targets: list, durations, max_duration_ns: int) -> None:
    """Refuse, by name and before any Schedule, what this probe cannot honour."""
    if len(targets) != 1:
        raise ValueError(
            f"qubit_ramsey_cryoscope on the Qblox backend builds its sequence "
            f"per qubit; run targets one at a time (got {list(targets)})")
    if max_duration_ns > MAX_DURATION_NS:
        raise ValueError(
            f"qubit_ramsey_cryoscope: max_duration_ns={max_duration_ns} exceeds "
            f"the Qblox backend's {MAX_DURATION_NS} ns ceiling — the probe "
            f"unrolls one Q1ASM block per nanosecond of the duration axis and "
            f"the readout sequencer's 12288-instruction budget fills first. "
            f"Shorten it (--set max_duration_ns={MAX_DURATION_NS}); the "
            f"microsecond tails belong to qubit_spectroscopy_cryoscope.")
    if int(durations[0]) != 1 or int(durations[-1]) != max_duration_ns:
        raise ValueError(
            f"qubit_ramsey_cryoscope: the duration axis must be every "
            f"nanosecond from 1 to max_duration_ns (got "
            f"[{durations[0]}, {durations[-1]}])")


@register
class QbloxQubitRamseyCryoscope(QubitRamseyCryoscope):
    """Build the Ramsey-cryoscope Schedule for a Qblox cluster."""

    #: what only this backend consumes, on top of the neutral requirements
    requires: ClassVar[tuple[Requirement, ...]] = (
        *QubitRamseyCryoscope.requires, HALF_PI_AMPLITUDE)
    #: true of THIS probe only; scqo shows them in
    #: `scqo run qubit_ramsey_cryoscope --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "both pi/2 pulses are played at half of pi_amp; pi_amp_x90 is not realized here",
        "max_duration_ns above 512 is refused: the sequence is written out once per "
        "nanosecond of the duration axis and the readout module's instruction memory "
        "fills first",
        "each flux pulse is a voltage offset held in steps of 4 ns, followed by a 1 to 3 "
        "ns pulse for the remainder",
        "the flux pulse starts as the first x90 ends, and the second x90 starts "
        "max_duration_ns + 8 ns after it",
    )

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import (
            IdlePulse,
            Measure,
            ResetClockPhase,
            ShiftClockPhase,
            SquarePulse,
            VoltageOffset,
            X90,
        )
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        durations = self.sweep_axes[DURATION_AXIS]
        frames = self.sweep_axes[FRAME_AXIS]
        reps = self.params.num_averages
        amp_v = float(self.params.flux_pulse_amp_v)
        max_dur = int(self.params.max_duration_ns)
        validate_inputs(list(self.params.targets), durations, max_dur)
        qubit_name = str(self.params.targets[0])

        acq = measure_kwargs(self, qubit_name)
        flux_port = vendor_element(self, qubit_name, "flux").ports.flux
        drive_clock = f"{qubit_name}.01"
        # the RELATIVE frame's origin — the same anchor estimate() records as
        # old_idle_flux, so the emitted bias and the fit's reference agree.
        idle_flux = flux_anchor_v(self, qubit_name)
        rail = check_flux_pulse_relative(
            self, name=f"{qubit_name} ramsey cryoscope flux pulse",
            port=flux_port, idle_v=idle_flux, amps_v=[amp_v])
        # VoltageOffset REPLACES the line level, so its operand is the absolute
        # sum; the remainder SquarePulse ADDS to the standing offset, so its
        # operand is the excursion alone.
        idle_frac = to_dac_fraction(idle_flux, rail)
        on_frac = to_dac_fraction(idle_flux + amp_v, rail)
        exc_frac = to_dac_fraction(amp_v, rail)
        # frame turns -> ShiftClockPhase degrees (endpoint-exclusive axis)
        frame_deg = 360.0 * frames

        schedule = Schedule("qubit_ramsey_cryoscope")
        sub = Schedule(f"ramsey_cryoscope_{qubit_name}")
        # establish the standing bias the excursions ride on
        sub.add(VoltageOffset(idle_frac, 0, port=flux_port))
        sub.add(IdlePulse(4e-9))
        with sub.loop(arange(0, reps, 1, DType.NUMBER)):
            # duration outer (Python-unrolled, ascending), frame inner
            # (realtime PHASE loop): flat bin order matches the canonical
            # sweep-axes order (duration_ns, frame).
            for n in (int(v) for v in durations):
                with sub.loop(linspace(frame_deg[0], frame_deg[-1],
                                       frame_deg.size, dtype=DType.PHASE)) as phi:
                    add_reset(sub, self, qubit_name)
                    sub.add(ResetClockPhase(clock=drive_clock))
                    sub.add(IdlePulse(4e-9))
                    sub.add(X90(qubit_name))
                    # the composed flux pulse: 4k ns of offset + r ns of
                    # waveform = idle + amp for exactly n contiguous ns
                    k4, r = duration_split(n)
                    if k4:
                        sub.add(VoltageOffset(on_frac, 0, port=flux_port))
                        sub.add(IdlePulse(k4 * 1e-9))
                        sub.add(VoltageOffset(idle_frac, 0, port=flux_port))
                    if r:
                        sub.add(SquarePulse(exc_frac, r * 1e-9, port=flux_port))
                    # pad to a CONSTANT free-evolution window (max + 4 ns from
                    # x90 end to the frame shift) — constant total decay, the
                    # same reason QM derives its idle from the longest duration
                    sub.add(IdlePulse((max_dur + 4 - n) * 1e-9))
                    # positive tomography frame — see the module docstring
                    sub.add(ShiftClockPhase(phase_shift=phi, clock=drive_clock))
                    sub.add(IdlePulse(4e-9))
                    sub.add(X90(qubit_name))
                    sub.add(Measure(
                        qubit_name,
                        coords={f"duration_{qubit_name}": float(n),
                                f"frame_{qubit_name}": phi},
                        acq_channel=f"S_21_{qubit_name}",
                        **acq,
                    ))
                    sub.add(IdlePulse(4e-9))
        # SAFETY: flux line back to 0 V at the end of the subschedule
        sub.add(VoltageOffset(0.0, 0, port=flux_port))
        sub.add(IdlePulse(4e-9))
        schedule.add(sub)
        return schedule
