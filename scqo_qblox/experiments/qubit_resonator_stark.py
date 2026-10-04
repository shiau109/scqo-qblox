"""Qblox qubit spectroscopy under a resonator Stark tone — supplies only ``probe()``.

At every Stark-tone amplitude, a finite saturation ``SquarePulse`` on the qubit's
microwave port while the drive-clock NCO steps through the detuning sweep; the
tone is a plain ``SquarePulse`` on the READOUT port-clock, and the standard
``Measure`` comes after its photons have left. ONE SEQUENCE, both backends (scqo
``experiments/_stark_tone.py`` owns it):

    readout port-clock : [==== Stark tone: amp_prefactor x readout_amp ====]              [## Measure ##]
    drive port-clock   :               [====== saturation drive ==========]
                         |<- ring_up ->|<----------- drive_len ---------->|<-depletion->|

The tone goes on the ASAP chain; the drive hangs off its START by the ring-up and
the ``Measure`` off its END by the depletion — both non-negative ``rel_time``s,
since subschedules never get ``_normalize_absolute_timing``. The tone is exactly
ring-up + drive long, so the two END together. The tone and the Measure share a
port-clock, and the compiler refuses overlapping long pulses there; a zero
depletion makes them touch, which it accepts.

THE SWEPT AMPLITUDE IS THE LOOP DOMAIN ITSELF. The ``DType.AMPLITUDE`` loop runs
over the ABSOLUTE amplitudes (``check_amp_window`` validates prefactor x
``readout_amp`` against the DAC and returns them), and the pulse takes the bare
loop variable: the long-pulse pass rewrites a >= 100 ns ``SquarePulse`` into a
``VoltageOffset`` pair plus a 4 ns tail and carries a bare ``Variable`` through,
while arithmetic on it (``factor * readout_amp``) dies inside the compiler.

Targets run one after another (one sub-schedule each), as the QM probe runs them
(``multiplexed=False``): N concurrent Stark tones on one feedline would be a
different experiment.

The timing is NOT computed here: ``define_sweep()`` resolved it through
``_stark_tone.stark_windows`` and this probe reads ``resolved_windows()``.

Drive power contract: the core ``run()`` already solved the drive chain for
``drive_power_dbm`` (recorded set -> acquire -> revert), parking the residual on
``element.spec.spec_amp`` — so the drive plays each qubit's OWN solved amplitude
(``view.drive_amp``), exactly as ``qubit_spectroscopy`` does.
"""

from __future__ import annotations

from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitResonatorStark

from ._amp_limits import check_amp_window
from ._reset import add_reset
from ._vendor import vendor_element


def _port(value: Any) -> str:
    """A port attribute from either scheduler API generation (QCoDeS callable or
    plain pydantic attribute)."""
    return value() if callable(value) else value


@register
class QbloxQubitResonatorStark(QubitResonatorStark):
    """Build the Stark-tone spectroscopy Schedule for a Qblox cluster, one target
    at a time. Subclasses the CORE class, not ``QbloxQubitSpectroscopy``: that
    probe opts into active reset, and a subclass would inherit the opt-in
    silently."""

    #: true of THIS probe only; scqo shows them in
    #: `scqo run qubit_resonator_stark --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "the Stark tone is a square pulse on the readout port and clock; amp_prefactor "
        "times readout_amp has to stay inside the output's full scale",
        "targets are measured one after another",
    )

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import (
            IdlePulse,
            Measure,
            SetClockFrequency,
            SquarePulse,
        )
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        prefactors = self.sweep_axes["amp_prefactor"]
        detuning = self.sweep_axes["detuning_hz"]
        reps = self.params.num_averages
        window = self.resolved_windows()  # the ONE timing authority
        # / 1e9, never * 1e-9: the probes' float-exactness rule (the scheduler
        # refuses anything off its 1 ns grid)
        tone_s = round(window.tone_len_ns) / 1e9
        ring_up_s = round(window.ring_up_ns) / 1e9
        drive_s = round(window.drive_len_ns) / 1e9
        depletion_s = round(window.depletion_ns) / 1e9

        schedule = Schedule("qubit_resonator_stark")
        for qubit_name in self.params.targets:
            drive_view = self.device.channel(qubit_name, "drive")
            readout_view = self.device.channel(qubit_name, "readout")
            center = float(drive_view.drive_freq_hz)
            drive_amp = float(drive_view.drive_amp)  # run() parked the solved residual here
            # the prefactors scale the CURRENT readout amplitude, validated against
            # the DAC here so an over-range window is refused by name
            amps = check_amp_window(prefactors, readout_view.readout_amp,
                                    target=qubit_name, field="readout_amp")
            drive_port = _port(vendor_element(self, qubit_name, "drive").ports.microwave)
            readout_port = _port(vendor_element(self, qubit_name, "readout").ports.readout)
            drive_clock = f"{qubit_name}.01"
            readout_clock = f"{qubit_name}.ro"

            stark_label = f"stark_{qubit_name}"
            drive_label = f"drive_{qubit_name}"
            meas_label = f"meas_{qubit_name}"
            sub = Schedule(f"qubit_resonator_stark_{qubit_name}")
            with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                with sub.loop(
                    linspace(float(amps[0]), float(amps[-1]), prefactors.size,
                             dtype=DType.AMPLITUDE)
                ) as amp:
                    # endpoint form, never center +/- span/2: the scqo window is an
                    # explicit [start, end] relative to drive_freq_hz and may be
                    # ASYMMETRIC (it is, by default)
                    with sub.loop(
                        linspace(
                            center + float(detuning[0]),
                            center + float(detuning[-1]),
                            detuning.size,
                            dtype=DType.FREQUENCY,
                        )
                    ) as freq:
                        # retune the drive NCO while nothing plays, then a real
                        # state reset: the previous point's drive has ended
                        sub.add(SetClockFrequency(clock=drive_clock, frequency=freq))
                        add_reset(sub, self, qubit_name)
                        # the Stark tone leads the ASAP chain...
                        sub.add(
                            SquarePulse(amp, tone_s, port=readout_port, clock=readout_clock),
                            label=stark_label,
                        )
                        # ...the drive starts one ring-up later and ends with it...
                        sub.add(
                            SquarePulse(drive_amp, drive_s, port=drive_port, clock=drive_clock),
                            label=drive_label, ref_op=stark_label, ref_pt="start",
                            rel_time=ring_up_s,
                        )
                        # ...and the STANDARD readout waits out the photons
                        sub.add(
                            Measure(
                                qubit_name,
                                coords={f"amp_{qubit_name}": amp,
                                        f"frequency_{qubit_name}": freq},
                                acq_channel=f"S_21_{qubit_name}",
                            ),
                            label=meas_label, ref_op=stark_label, ref_pt="end",
                            rel_time=depletion_s,
                        )
                        # re-anchor the ASAP chain past the readout: the Measure
                        # ends with a zero-duration clock restore, and the next
                        # point must not start inside this measurement
                        sub.add(IdlePulse(4e-9), ref_op=meas_label, ref_pt="end")
            schedule.add(sub)
        return schedule
