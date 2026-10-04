"""Qblox discrete charge-parity monitor — supplies only ``probe()``.

Per cycle: Measure M1 — depletion wait — X90 — fixed idle — Y90 — Measure M2 —
pad wait, repeated as one labeled cycle loop at the total period
``cycle_period_ns``. M1 PROJECTS the qubit (measurement-based initialization —
the reason there is still no ``add_reset`` and no ``supports_active_reset``),
and the parity of each cycle is the WITHIN-CYCLE difference m1 XOR m2 — so,
unlike the continuous sibling, the pad may stretch the cycle arbitrarily long
without T1 corrupting the parity record.

BOTH measurements ride the SAME ``S_21_<q>`` acquisition channel,
distinguished by a constant ``meas`` coordinate (0 = M1) beside the captured
cycle loop variable — the cal19/cal13 reference pattern. A second acq channel
would compile clean and then be silently DROPPED by ``_to_canonical``, which
reads only ``S_21_{name}``; the extra length-2 axis instead rides
``sweep_axes`` (the neutral contract's ``meas_idx`` sweep), so the flat bins
fold back to ``(shot_idx, meas_idx)`` in loop order.

TWO ACQUISITION BINS PER CYCLE: the bin guard counts ``2 * num_shots`` against
the sequencer's 3e6-bin ceiling — the bin-limited record is HALF the
continuous variant's (the neutral ``max_num_shots`` default already sits under
half the limit for the same reason).

The idle time and the depletion wait resolve exactly as in the continuous
sibling; the pad math (``cycle_period_ns`` minus the scheduled sequence,
refused by name when negative) lives here because only the driver knows the
vendor operation durations.
"""

from __future__ import annotations

from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitParitySwitchDiscrete
from scqo.experiments._depletion import depletion_wait_ns
from scqo.requirements import Requirement

from scqo_qblox.experiments._requires import HALF_PI_AMPLITUDE
from scqo_qblox.experiments._state import measure_kwargs
from scqo_qblox.experiments.qubit_parity_switch_continuous import (
    _MAX_ACQ_BINS,
    _op_durations,
)

_PERIOD_TOO_SHORT = (
    "cycle_period_ns = {want:.0f} ns is shorter than the scheduled sequence "
    "on {target}: M1 + depletion + X90 + idle + Y90 + M2 already takes "
    "{sequence:.0f} ns. Raise cycle_period_ns to at least that (or leave it "
    "None for the minimal, unpadded cycle)."
)


@register
class QbloxQubitParitySwitchDiscrete(QubitParitySwitchDiscrete):
    """Build a two-measurement-per-cycle parity-monitor Schedule for a Qblox
    cluster."""

    #: what only this backend consumes, on top of the neutral requirements
    requires: ClassVar[tuple[Requirement, ...]] = (
        *QubitParitySwitchDiscrete.requires, HALF_PI_AMPLITUDE)
    #: true of THIS probe only; scqo shows them in
    #: `scqo run qubit_parity_switch_discrete --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "both pi/2 pulses are played at half of pi_amp; pi_amp_x90 is not realized here",
        "more than 1,500,000 cycles is refused: every cycle takes two acquisition bins",
        "the pad that fills the cycle is on the 1 ns grid",
        "targets are measured one after another, each at its own cycle period",
    )

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, X90, Y90
        from qblox_scheduler.operations.loop_domains import DType, arange

        # resolved by the neutral layer from record_time_s (or an explicit
        # num_shots override) — never read params.num_shots here.
        num_shots = self.resolved_num_shots()
        if 2 * num_shots > _MAX_ACQ_BINS:
            raise ValueError(
                f"{num_shots} cycles needs {2 * num_shots} acquisition bins — "
                f"TWO per cycle (M1 and M2 are each their own labeled bin) — "
                f"over the sequencer's {_MAX_ACQ_BINS} bin limit. Shorten "
                f"record_time_s, raise cycle_period_ns (a slower cadence "
                f"reaches the same spectral edge with fewer cycles), or take "
                f"several runs — a campaign step repeats this experiment with "
                f"its own folder per repeat."
            )

        self.probe_shot_period_s: dict[str, float] = {}
        schedule = Schedule("parity_switch_discrete_multiplexed")
        for qubit_name in self.params.targets:
            acq = measure_kwargs(self, qubit_name)  # {} or the thresholded protocol
            idle_ns = self.resolved_idle_ns(qubit_name)
            # never None: define_sweep refused a target without a governed wait
            depletion_ns = round(float(depletion_wait_ns(self, qubit_name)))
            readout_s, pi_s = _op_durations(self, qubit_name)
            sequence_ns = (float(idle_ns) + float(depletion_ns)
                           + 2.0 * readout_s * 1e9 + 2.0 * pi_s * 1e9)
            tau_ns = self._tau_wait_ns(
                self.params.cycle_period_ns, sequence_ns, qubit_name)

            sub = Schedule(f"parity_switch_discrete_{qubit_name}")
            with sub.loop(arange(0, num_shots, 1, DType.NUMBER)) as shot:
                # M1: the projective initialization, meas coord 0
                sub.add(
                    Measure(
                        qubit_name,
                        coords={f"shot_{qubit_name}": shot,
                                f"meas_{qubit_name}": 0},
                        acq_channel=f"S_21_{qubit_name}",
                        **acq,
                    )
                )
                # wait out M1's photons before the coherent block
                # (/ 1e9, never * 1e-9: the probes' float-exactness rule)
                if depletion_ns:
                    sub.add(IdlePulse(depletion_ns / 1e9))
                # x90 - fixed idle - y90: the reference protocol's order —
                # equivalent to the continuous y90-first order up to a
                # telegraph sign flip the PSD cannot see.
                sub.add(X90(qubit_name))
                sub.add(IdlePulse(idle_ns / 1e9))
                sub.add(Y90(qubit_name))
                # M2: the parity readout, meas coord 1, SAME channel
                sub.add(
                    Measure(
                        qubit_name,
                        coords={f"shot_{qubit_name}": shot,
                                f"meas_{qubit_name}": 1},
                        acq_channel=f"S_21_{qubit_name}",
                        **acq,
                    )
                )
                # pad the cycle to the requested period (0 = minimal cycle)
                if tau_ns:
                    sub.add(IdlePulse(tau_ns / 1e9))
            schedule.add(sub)
            # the EXACT scheduled cycle period — the telegraph timebase
            self.probe_shot_period_s[qubit_name] = (
                (float(idle_ns) + float(depletion_ns) + float(tau_ns)) / 1e9
                + 2.0 * readout_s + 2.0 * pi_s)
        return schedule

    @staticmethod
    def _tau_wait_ns(cycle_period_ns: float | None, sequence_ns: float,
                     target: str) -> int:
        """The pad that stretches the cycle to ``cycle_period_ns``, in ns on
        the 1 ns compile grid. None or an exact fit pads nothing; a period
        SHORTER than the sequence is refused by name."""
        if cycle_period_ns is None:
            return 0
        remainder = float(cycle_period_ns) - float(sequence_ns)
        if remainder < 0:
            raise ValueError(_PERIOD_TOO_SHORT.format(
                want=float(cycle_period_ns), sequence=float(sequence_ns),
                target=target))
        return int(round(remainder))
