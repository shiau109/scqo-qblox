"""Qblox qubit power Rabi — supplies only ``probe()``.

Amplitude sweep: X(amp) — Measure, looping the drive amplitude. Parameters, the cosine
fit, pi_amp recovery and writeback are inherited from ``scqo.experiments.QubitPowerRabi``.

The coupler-as-a-target fields (``drive_line``, ``readout_member``; SCQO
docs/coupler-transmon-plan.md) are refused BY NAME: this backend adopts no borrowed
drive channel and plays no mapped readout, so v1 runs them on QM (MW-FEM) only.
"""

from __future__ import annotations

from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitPowerRabi

from scqo_qblox.experiments._amp_limits import check_amp_window
from scqo_qblox.experiments._reset import add_reset
from scqo_qblox.experiments._state import measure_kwargs


@register
class QbloxQubitPowerRabi(QubitPowerRabi):
    """Build a multiplexed power-Rabi Schedule for a Qblox cluster."""

    #: readout is held at the calibrated point for the whole run and the Reset is
    #: a genuine state reset, so reset_method='active' is valid here (_reset.py).
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        routed = [f"{field}={getattr(self.params, field)!r}"
                  for field in ("drive_line", "readout_member")
                  if getattr(self.params, field, None) is not None]
        if routed:
            raise NotImplementedError(
                f"{', '.join(routed)} is not realized on the Qblox backend yet: it "
                f"adopts no borrowed drive channel and plays no mapped readout, so "
                f"a target driven through another line or read through a pair "
                f"member runs on QM (MW-FEM) only - SCQO "
                f"docs/coupler-transmon-plan.md")
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, X
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        amp_factor = self.sweep_axes["amp_prefactor"]
        reps = self.params.num_averages

        schedule = Schedule("power_rabi_multiplexed")
        for qubit_name in self.params.targets:
            # neutral field on the target's drive CHANNEL (q<n>_xy). DIMENSIONLESS:
            # a fraction of the port's full scale, not volts (scqo/catalog.py
            # declares its unit as ""), so amp_factor scales a fraction and the
            # product stays one. Only the FLUX path deals in volts, and it
            # converts explicitly (see experiments/_flux_limits.py).
            pi_amp = self.device.channel(qubit_name, "drive").pi_amp
            amp_abs = check_amp_window(amp_factor, pi_amp, target=qubit_name,
                                       field="pi_amp")
            acq = measure_kwargs(self, qubit_name)  # {} or the thresholded protocol
            sub = Schedule(f"power_rabi_{qubit_name}")
            with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                with sub.loop(linspace(amp_abs[0], amp_abs[-1], amp_abs.size, dtype=DType.AMPLITUDE)) as amp:
                    add_reset(sub, self, qubit_name)
                    sub.add(X(qubit_name, amp180=amp))
                    sub.add(
                        Measure(
                            qubit_name,
                            coords={f"amp_{qubit_name}": amp},
                            acq_channel=f"S_21_{qubit_name}",
                            **acq,
                        )
                    )
                    sub.add(IdlePulse(4e-9))
            schedule.add(sub)
        return schedule
