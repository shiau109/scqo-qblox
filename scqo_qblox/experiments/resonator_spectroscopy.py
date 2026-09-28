"""Qblox resonator spectroscopy — supplies only ``probe()``.

Parameters, fitting, device writeback and the simulator are all inherited from
``scqo.experiments.ResonatorSpectroscopy``. ``qblox_scheduler`` is imported inside
``probe()`` so the simulated path (and ``import scqo_qblox``) needs no Qblox install.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import ResonatorSpectroscopy


@register
class QbloxResonatorSpectroscopy(ResonatorSpectroscopy):
    """Build a multiplexed resonator-spectroscopy Schedule for a Qblox cluster."""

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        detuning = self.sweep_axes["detuning_hz"]
        reps = self.params.num_averages

        schedule = Schedule("resonator_spectroscopy_multiplexed")
        for qubit_name in self.params.targets:
            # the knob lives on the target's readout CHANNEL entity (feedline.q1)
            center = self.device.channel(qubit_name, "readout").readout_freq_hz
            sub = Schedule(f"res_spec_{qubit_name}")
            with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                # endpoint form, never center +/- span/2: the scqo window is an
                # explicit [start, end] relative to readout_freq_hz and may be
                # ASYMMETRIC — re-deriving a symmetric span would silently
                # re-center it (the flux probe set this pattern)
                with sub.loop(
                    linspace(
                        center + float(detuning[0]),
                        center + float(detuning[-1]),
                        detuning.size,
                        dtype=DType.FREQUENCY,
                    )
                ) as freq:
                    # coords labels the sweep point and acq_channel names the S21
                    # acquisition: without them the cluster averages the whole sweep
                    # into one anonymous bin (verified on hardware 2026-07-04).
                    sub.add(
                        Measure(
                            qubit_name,
                            freq=freq,
                            coords={f"frequency_{qubit_name}": freq},
                            acq_channel=f"S_21_{qubit_name}",
                        )
                    )
                    sub.add(IdlePulse(10e-6))
            schedule.add(sub)
        return schedule
