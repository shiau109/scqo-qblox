"""Qblox Hahn echo — supplies only ``probe()``.

cal15 reference pattern: Reset -> X90 -> X at ``rel_time=tau/2`` -> X90 at
``rel_time=tau/2`` -> Measure, sweeping the total idle time tau (loop-variable
arithmetic: the scheduler evaluates ``tau / 2`` per point). Parameters, the
exponential-envelope fit and T2_echo reporting are inherited from
``scqo.experiments.QubitEcho``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitEcho

from scqo_qblox.experiments._reset import add_reset
from scqo_qblox.experiments._state import measure_kwargs


@register
class QbloxQubitEcho(QubitEcho):
    """Build a multiplexed Hahn-echo Schedule for a Qblox cluster."""

    #: true of THIS probe only; scqo shows them in `scqo run qubit_echo --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "both pi/2 pulses are played at half of pi_amp; pi_amp_x90 is not realized here",
    )

    #: readout is held at the calibrated point for the whole run and the Reset is
    #: a genuine state reset, so reset_method='active' is valid here (_reset.py).
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, X, X90
        from qblox_scheduler.operations.loop_domains import DType, arange, linspace

        wait_ns = self.sweep_axes["wait_time_ns"]
        reps = self.params.num_averages

        schedule = Schedule("qubit_echo_multiplexed")
        for qubit_name in self.params.targets:
            acq = measure_kwargs(self, qubit_name)  # {} or the thresholded protocol
            sub = Schedule(f"echo_{qubit_name}")
            with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                with sub.loop(
                    linspace(wait_ns[0] * 1e-9, wait_ns[-1] * 1e-9, wait_ns.size, dtype=DType.TIME)
                ) as tau:
                    add_reset(sub, self, qubit_name)
                    sub.add(X90(qubit_name))
                    # central pi pulse refocuses quasi-static dephasing; the two
                    # tau/2 gaps make tau the TOTAL idle time (cal15 pattern)
                    sub.add(X(qubit=qubit_name), rel_time=tau / 2)
                    sub.add(X90(qubit_name), rel_time=tau / 2)
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
