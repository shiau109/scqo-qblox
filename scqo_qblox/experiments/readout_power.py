"""Qblox readout-amplitude scan — supplies only ``probe()``.

Per amplitude prefactor the readout pulse amplitude is set via
``Measure(pulse_amp=...)``, prefactor x the CURRENT ``readout_amp`` (read the same
way resonator_spectroscopy_power_amp's punchout probe reads it). Each prefactor runs
two sequential prepared-state blocks (|0>: Reset -> Measure, |1>: Reset -> X ->
Measure).

BOTH readout modes come off the same schedule, and the ONE difference is whether
the repetition loop's variable is CAPTURED into the bin coords:

* ``readout_mode="shot"`` — ``coords={... "shot_<q>": shot}`` labels every
  repetition, so the cluster appends one I/Q point per shot instead of averaging
  (the single_shot_readout mechanism). Flat bin order is amp-major, then state,
  then shot: ``amp_prefactor``, ``prepared_state``, ``shot_idx``.
* ``readout_mode="average"`` — the same loop runs unlabeled, so every repetition
  lands in the SAME bin and the cluster averages it (the resonator_spectroscopy
  idiom). No ``shot_idx`` at all, which is the form scqo's contract accepts as
  its alt set.

There is no separate ``num_averages``: ``num_shots`` IS the repetition count in
both modes.

Method note: QBLOX_training's cal17 calibrates readout amplitude via the AC-Stark
shift of the qubit frequency — a different method deliberately NOT followed here;
this probe measures the states directly, matching the QM backend. Parameters and
the analysis are inherited from ``scqo.experiments.ReadoutPower``.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import ReadoutPower

from ._amp_limits import check_amp_window
from ._reset import add_reset


@register
class QbloxReadoutPower(ReadoutPower):
    """Build a multiplexed readout-amplitude Schedule for a Qblox cluster."""

    # No supports_active_reset: this probe SWEEPS the readout amplitude, so the
    # discriminator single_shot_readout solved at the nominal power is wrong at
    # almost every point. _reset.py refuses reset_method='active' by name.

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, X
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        prefactors = self.sweep_axes["amp_prefactor"]
        num_shots = int(self.params.num_shots)
        per_shot = self.params.readout_mode == "shot"

        def shot_coord(qubit: str, shot) -> dict:
            """The shot label, and ONLY in shot mode: an unlabeled repetition
            loop is what makes the cluster average into one bin."""
            return {f"shot_{qubit}": shot} if per_shot else {}

        schedule = Schedule("readout_power_multiplexed")
        for qubit_name in self.params.targets:
            view = self.device.channel(qubit_name, "readout")
            # the prefactor scales the CURRENT readout amplitude (punchout pattern),
            # validated against the DAC here so an over-range window is refused by
            # name instead of dying inside the compiler on `awg_gain`
            amp_abs = check_amp_window(prefactors, view.readout_amp,
                                       target=qubit_name, field="readout_amp")
            amp_start, amp_end = float(amp_abs[0]), float(amp_abs[-1])
            sub = Schedule(f"readout_power_{qubit_name}")
            with sub.loop(
                linspace(amp_start, amp_end, prefactors.size, dtype=DType.AMPLITUDE)
            ) as amp:
                # prepared_state 0: Reset -> Measure, one bin per shot (shot
                # mode) or one averaged bin (average mode)
                with sub.loop(arange(0, num_shots, 1, DType.NUMBER)) as shot:
                    add_reset(sub, self, qubit_name)
                    sub.add(
                        Measure(
                            qubit_name,
                            pulse_amp=amp,
                            coords={
                                f"amp_{qubit_name}": amp,
                                f"state_{qubit_name}": 0,
                                **shot_coord(qubit_name, shot),
                            },
                            acq_channel=f"S_21_{qubit_name}",
                        )
                    )
                    # close the loop body with a real duration: the
                    # clock restore Measure's factory appends is a
                    # ZERO-duration parameter op and may not land on
                    # the loop's ControlFlowReturn (the chipA
                    # readout_frequency failure, 2026-07-26). Every
                    # other in-loop Measure here already does this;
                    # these two probes were the outliers.
                    sub.add(IdlePulse(4e-9))
                # prepared_state 1: Reset -> X -> Measure, same binning rule
                with sub.loop(arange(0, num_shots, 1, DType.NUMBER)) as shot:
                    add_reset(sub, self, qubit_name)
                    sub.add(X(qubit=qubit_name))
                    sub.add(
                        Measure(
                            qubit_name,
                            pulse_amp=amp,
                            coords={
                                f"amp_{qubit_name}": amp,
                                f"state_{qubit_name}": 1,
                                **shot_coord(qubit_name, shot),
                            },
                            acq_channel=f"S_21_{qubit_name}",
                        )
                    )
                    # close the loop body with a real duration: the
                    # clock restore Measure's factory appends is a
                    # ZERO-duration parameter op and may not land on
                    # the loop's ControlFlowReturn (the chipA
                    # readout_frequency failure, 2026-07-26). Every
                    # other in-loop Measure here already does this;
                    # these two probes were the outliers.
                    sub.add(IdlePulse(4e-9))
            sub.add(IdlePulse(4e-9))
            schedule.add(sub)
        return schedule
