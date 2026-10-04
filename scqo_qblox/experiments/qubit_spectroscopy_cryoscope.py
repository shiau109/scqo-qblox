"""Qblox long-time (spectroscopy) cryoscope — supplies only ``probe()``.

Per (detuning, wait) point: reset at the standing bias -> park the flux at
``idle + flux_pulse_amp_v`` (a held ``VoltageOffset`` — the step) -> wait ``t``
INTO the step -> play a weak pi-AREA square spectroscopy tone at the
arch-centered, detuning-stepped drive frequency -> 100 ns tail -> flux back to
idle -> measure. That mirrors the QM sequence exactly (flux held for
``t + drive_len + 100 ns``, readout after the line is back at idle), so the
shared estimator sees the same physics. Parameters, the drive centering
(``resolved_center_offset_hz``), the per-wait peak fit and the tap-fact
writeback are inherited from ``scqo.experiments.QubitSpectroscopyCryoscope``.

SQUARE ONLY. ``drive_shape`` 'cosine'/'gaussian' are REFUSED by name, not
downgraded: the settled verdict (2026-08-19, 5Q4C) is that the center precision
tracks the linewidth and square is the narrowest line per nanosecond — the
smooth envelopes exist only against a line that looks WRONG, and each would
also cost ``drive_len_ns`` waveform-memory samples where square stays a single
stitched pulse. ``drive_sigma_frac`` is gaussian-only and simply unused here.

THE PI-AREA AMPLITUDE is computed from the calibrated x180 the same way the QM
probe does it: the rotation area of the pulse the instrument ACTUALLY PLAYS —
``pi_amp`` times the integral of the rxy DRAG envelope at ``pi_duration_s``
(sigma = duration/8, average-offset-subtracted, exactly the
``rxy_drag_pulse`` factory's rendering) — held constant as the tone stretches
to ``drive_len_ns``, so a longer pulse is proportionally weaker and stays a
true pi pulse. The guard is the DAC bound (|amp| <= 1 of full scale): QM's
"loudest stored operation" ceiling has no Qblox analogue, since amplitudes
here are absolute fractions rather than scales on a stored waveform.

WAIT AXIS: consumed verbatim from ``sweep_axes`` (the base class already
log-spaces, 16 ns-floors, 4 ns-snaps and DEDUPES it) and Python-unrolled inside
a realtime FREQUENCY loop — a log axis is not ``linspace``-able, and ~50
unrolled blocks inside one realtime loop is a few hundred instructions. Flat
bin order stays detuning-major (the realtime loop) x wait-minor, matching the
contract's ``(detuning_hz, wait_time_ns)``.

Active reset stays refused here for now (default DENY): the spectroscopy tone
sweeps the drive frequency but the READOUT is at the calibrated point, so an
opt-in may be worth revisiting once the QM twin's active-reset experience is
validated on this backend's hardware.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from scqo import register
from scqo.experiments import QubitSpectroscopyCryoscope
from scqo.experiments._capabilities import flux_anchor_v
from scqo.experiments._capabilities.detuning import DETUNING_AXIS
from scqo.experiments.qubit_spectroscopy_cryoscope import WAIT_AXIS

from ._flux_limits import check_flux_pulse_relative, to_dac_fraction
from ._reset import add_reset
from ._state import measure_kwargs
from ._vendor import vendor_element

#: the sequencer's amplitude operand is a fraction of full scale — 1.0 IS the rail.
MAX_DAC_FRACTION = 1.0

#: the settle tail the flux step holds after the drive tone — QM's 25 cycles.
_FLUX_TAIL_S = 100e-9


def x180_area_amp_ns(amp180: float, duration_s: float) -> float:
    """Rotation area of the calibrated x180 in (DAC fraction x ns).

    Samples the exact envelope ``rxy_drag_pulse`` renders — ``waveforms.drag``
    at ``nr_sigma=4`` (sigma = duration/8) with the default average-offset
    subtraction — on the 1 ns grid and integrates the I quadrature. The DRAG
    derivative lives in Q and integrates to zero, so beta plays no part.
    """
    from qblox_scheduler.waveforms import drag

    n = int(round(float(duration_s) * 1e9))
    t = np.arange(n, dtype=float) * 1e-9
    envelope = drag(t, amplitude=float(amp180), beta=0.0,
                    duration=float(duration_s), nr_sigma=4)
    return float(np.sum(envelope.real))


def square_pi_amp(x180_area_ns: float, drive_len_ns: float,
                  amp_factor: float) -> float:
    """The square tone's amplitude holding the x180 rotation area: a square
    envelope's mean IS its amplitude, so ``factor * area / length`` — the QM
    ``drive_amp_for_area`` with the square envelope ratio (1.0) inlined."""
    return float(amp_factor) * float(x180_area_ns) / float(drive_len_ns)


def validate_inputs(targets: list, drive_shape: str) -> None:
    """Refuse, by name and before any device read, what this probe cannot honour."""
    if len(targets) != 1:
        raise ValueError(
            f"qubit_spectroscopy_cryoscope on the Qblox backend builds its "
            f"sequence per qubit; run targets one at a time (got {list(targets)})")
    if drive_shape != "square":
        raise NotImplementedError(
            f"qubit_spectroscopy_cryoscope: drive_shape={drive_shape!r} is not "
            f"realized on the Qblox backend — only 'square' (the narrowest line "
            f"per nanosecond; the settled verdict is that shape is not the "
            f"lever, length is). Use drive_shape='square'.")


def check_drive_amp(target: str, pi_amp: float, amp_sq: float,
                    drive_len_ns: float) -> None:
    """Refuse an uncalibrated x180 or a tone past the DAC's full scale, by name."""
    if not np.isfinite(pi_amp) or pi_amp == 0.0:
        raise ValueError(
            f"{target}: the spectroscopy tone's pi-area amplitude needs a "
            f"calibrated x180 (pi_amp is {pi_amp!r}) — run qubit_power_rabi "
            f"first.")
    if abs(amp_sq) > MAX_DAC_FRACTION:
        longer = 4 * int(np.ceil(abs(amp_sq) * drive_len_ns / 4))
        raise ValueError(
            f"{target}: the pi-area square tone needs amplitude {amp_sq:.3g}, "
            f"past the DAC's full scale (1.0). Lengthen the pulse "
            f"(--set drive_len_ns={longer}) or lower drive_amp_factor.")


@register
class QbloxQubitSpectroscopyCryoscope(QubitSpectroscopyCryoscope):
    """Build the long-time spectroscopy-cryoscope Schedule for a Qblox cluster."""

    #: true of THIS probe only; scqo shows them in
    #: `scqo run qubit_spectroscopy_cryoscope --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "drive_shape cosine and gaussian are refused: only the square pulse is realized, "
        "and drive_sigma_frac is not used",
        "the area of the x180 is that of the DRAG gaussian this backend plays for "
        "pi_amp and pi_duration_s",
        "a pulse whose amplitude would pass the full scale of the output is refused; "
        "lengthen drive_len_ns or lower drive_amp_factor",
        "the flux pulse is a voltage offset held from the start of the wait to 100 ns "
        "after the drive pulse, not a shaped pulse",
    )

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import (
            IdlePulse,
            Measure,
            SetClockFrequency,
            SquarePulse,
            VoltageOffset,
        )
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        detuning = self.sweep_axes[DETUNING_AXIS]
        # consumed VERBATIM: the base class already log-spaced, floored,
        # 4 ns-snapped and deduped the wait axis — never re-derive it.
        waits = self.sweep_axes[WAIT_AXIS]
        reps = self.params.num_averages
        amp_v = float(self.params.flux_pulse_amp_v)
        drive_len_ns = float(self.params.drive_len_ns)
        validate_inputs(list(self.params.targets), self.params.drive_shape)
        qubit_name = str(self.params.targets[0])

        view = self.device.channel(qubit_name, "drive")
        pi_amp = float(view.pi_amp)
        amp_sq = square_pi_amp(
            x180_area_amp_ns(pi_amp, float(view.pi_duration_s)),
            drive_len_ns, float(self.params.drive_amp_factor))
        check_drive_amp(qubit_name, pi_amp, amp_sq, drive_len_ns)

        acq = measure_kwargs(self, qubit_name)
        flux_port = vendor_element(self, qubit_name, "flux").ports.flux
        microwave_port = vendor_element(self, qubit_name, "drive").ports.microwave
        drive_clock = f"{qubit_name}.01"
        # arch-predicted parked line: the drive centers on the current knob plus
        # the base class's quadratic prediction; probe and estimator share the
        # one value (attach_acquisition_coords snapshots it onto the dataset).
        center = float(view.drive_freq_hz) + self.resolved_center_offset_hz(qubit_name)
        idle_flux = flux_anchor_v(self, qubit_name)
        rail = check_flux_pulse_relative(
            self, name=f"{qubit_name} spectroscopy cryoscope flux step",
            port=flux_port, idle_v=idle_flux, amps_v=[amp_v])
        idle_frac = to_dac_fraction(idle_flux, rail)
        on_frac = to_dac_fraction(idle_flux + amp_v, rail)

        schedule = Schedule("qubit_spectroscopy_cryoscope")
        sub = Schedule(f"spectroscopy_cryoscope_{qubit_name}")
        # establish the standing bias the step rides on
        sub.add(VoltageOffset(idle_frac, 0, port=flux_port))
        sub.add(IdlePulse(4e-9))
        with sub.loop(arange(0, reps, 1, DType.NUMBER)):
            # detuning outer (realtime FREQUENCY loop, endpoint form), wait
            # inner (Python-unrolled log axis): flat bin order matches the
            # canonical sweep-axes order (detuning_hz, wait_time_ns).
            with sub.loop(
                linspace(center + float(detuning[0]), center + float(detuning[-1]),
                         detuning.size, dtype=DType.FREQUENCY)
            ) as freq:
                sub.add(SetClockFrequency(clock=drive_clock, frequency=freq))
                sub.add(IdlePulse(4e-9))
                for wait_ns in (int(v) for v in waits):
                    # reset at the STANDING bias, before the step lands
                    add_reset(sub, self, qubit_name)
                    # the flux step ON — held through wait + drive + tail,
                    # exactly the window QM's bracketed `const` plays
                    sub.add(VoltageOffset(on_frac, 0, port=flux_port))
                    sub.add(IdlePulse(wait_ns * 1e-9))
                    # the pi-area square tone, `wait_ns` into the step
                    sub.add(SquarePulse(amp_sq, drive_len_ns * 1e-9,
                                        port=microwave_port, clock=drive_clock))
                    sub.add(IdlePulse(_FLUX_TAIL_S))
                    # flux back to idle BEFORE the readout (QM mirror: measure
                    # at the calibrated operating point)
                    sub.add(VoltageOffset(idle_frac, 0, port=flux_port))
                    sub.add(IdlePulse(4e-9))
                    sub.add(Measure(
                        qubit_name,
                        coords={f"frequency_{qubit_name}": freq,
                                f"wait_{qubit_name}": float(wait_ns)},
                        acq_channel=f"S_21_{qubit_name}",
                        **acq,
                    ))
                    sub.add(IdlePulse(4e-9))
        # SAFETY: flux line back to 0 V at the end of the subschedule
        sub.add(VoltageOffset(0.0, 0, port=flux_port))
        sub.add(IdlePulse(4e-9))
        schedule.add(sub)
        return schedule
