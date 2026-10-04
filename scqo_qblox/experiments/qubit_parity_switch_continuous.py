"""Qblox continuous charge-parity monitor — supplies only ``probe()``.
(The two-measurement-per-cycle sibling is ``qubit_parity_switch_discrete``.)

Y90 — fixed idle — X90 — Measure, repeated as one labeled shot loop: the same
per-shot mechanism as ``single_shot_readout`` (the loop variable is CAPTURED
into the acquisition ``coords``, so every shot is a distinct bin and the cluster
appends instead of averaging). No ``num_averages`` exists for this experiment.

Two things are deliberately absent, and both are the physics:

* **No ``add_reset``.** Between shots this experiment waits ONLY for resonator
  depletion, and the absence of a reset is REQUIRED, not an optimization: the
  sequence inverts with the pole the previous shot left behind, so the readout
  is the running XOR of the parity and the consecutive-pair difference is the
  parity telegraph the rate is fitted from. Resetting to |0> each shot would
  sever that chain. The shot cadence is also the telegraph timebase. The
  Parameters carry no ``reset_method`` at all, so ``QbloxBackend.acquire``'s
  pre-probe ``check_reset_method`` sees the thermal default and passes
  trivially without a ``Reset`` ever being built.
* **No ``supports_active_reset``.** Nothing to opt into.

The idle time is resolved by the neutral layer (``define_sweep`` refused
already if the drive channel had no accepted ``parity_delta_f_hz``), and the
depletion wait comes through scqo's one precedence helper — never a number this
driver picks.
"""

from __future__ import annotations

from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitParitySwitchContinuous
from scqo.experiments._depletion import depletion_wait_ns
from scqo.requirements import Requirement

from scqo_qblox.experiments._requires import HALF_PI_AMPLITUDE
from scqo_qblox.experiments._state import measure_kwargs

#: acquisition bins per sequencer (qblox_scheduler constants.MAX_NUMBER_OF_BINS).
#: The parity monitors are the experiments whose shot counts can realistically
#: reach it, so they refuse by name rather than dying inside the compiler.
#: (The discrete sibling imports this and counts TWO bins per cycle.)
_MAX_ACQ_BINS = 3_000_000


def _op_durations(experiment, target: str) -> tuple[float, float]:
    """(readout_s, pi_s) off the vendor elements: the Measure's pulse +
    acq-delay span and one pi/2 pulse duration. Shared by both parity
    monitors' scheduled-period sums, so the two timebases cannot drift."""
    from scqo_qblox.experiments._vendor import vendor_element

    element = vendor_element(experiment, target, "readout")
    measure = element.measure
    readout_s = float(measure.pulse_duration()
                      if callable(measure.pulse_duration) else measure.pulse_duration)
    acq_delay = getattr(measure, "acq_delay", 0.0)
    readout_s += float(acq_delay() if callable(acq_delay) else acq_delay)

    drive = vendor_element(experiment, target, "drive").rxy
    pi_s = float(drive.duration() if callable(drive.duration) else drive.duration)
    return readout_s, pi_s


@register
class QbloxQubitParitySwitchContinuous(QubitParitySwitchContinuous):
    """Build a per-shot parity-monitor Schedule for a Qblox cluster."""

    #: what only this backend consumes, on top of the neutral requirements
    requires: ClassVar[tuple[Requirement, ...]] = (
        *QubitParitySwitchContinuous.requires, HALF_PI_AMPLITUDE)
    #: true of THIS probe only; scqo shows them in
    #: `scqo run qubit_parity_switch_continuous --help`
    backend_notes: ClassVar[tuple[str, ...]] = (
        "both pi/2 pulses are played at half of pi_amp; pi_amp_x90 is not realized here",
        "more than 3,000,000 shots is refused: every shot is its own acquisition bin",
        "targets are measured one after another, each at its own shot period",
    )

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, Measure, X90, Y90
        from qblox_scheduler.operations.loop_domains import DType, arange

        # resolved by the neutral layer from record_time_s (or an explicit
        # num_shots override) — never read params.num_shots here, it is None in
        # the normal case.
        num_shots = self.resolved_num_shots()
        if num_shots > _MAX_ACQ_BINS:
            raise ValueError(
                f"{num_shots} shots exceeds the sequencer's acquisition-bin "
                f"limit ({_MAX_ACQ_BINS}): every shot is its own labeled bin "
                f"here (no averaging). Shorten record_time_s, or take several "
                f"runs — a campaign step repeats this experiment with its own "
                f"folder per repeat."
            )

        self.probe_shot_period_s: dict[str, float] = {}
        schedule = Schedule("parity_switch_multiplexed")
        for qubit_name in self.params.targets:
            acq = measure_kwargs(self, qubit_name)  # {} or the thresholded protocol
            idle_ns = self.resolved_idle_ns(qubit_name)
            # never None: define_sweep refused a target without a governed wait
            depletion_ns = round(float(depletion_wait_ns(self, qubit_name)))

            sub = Schedule(f"parity_switch_{qubit_name}")
            with sub.loop(arange(0, num_shots, 1, DType.NUMBER)) as shot:
                # y90 - idle - x90: the 90-degree-SHIFTED second pulse measures
                # sin of the parity phase (+/- pi/2 at this idle), which is ODD
                # in parity — a same-axis pair would measure the even cos and
                # see no parity at all.
                sub.add(Y90(qubit_name))
                # / 1e9, never * 1e-9: the probes' float-exactness rule
                sub.add(IdlePulse(idle_ns / 1e9))
                sub.add(X90(qubit_name))
                sub.add(
                    Measure(
                        qubit_name,
                        coords={f"shot_{qubit_name}": shot},
                        acq_channel=f"S_21_{qubit_name}",
                        **acq,
                    )
                )
                if depletion_ns:
                    sub.add(IdlePulse(depletion_ns / 1e9))
            schedule.add(sub)
            self.probe_shot_period_s[qubit_name] = self._shot_period_s(
                qubit_name, idle_ns, depletion_ns)
        return schedule

    def _shot_period_s(self, target: str, idle_ns: float, depletion_ns: float) -> float:
        """The scheduled shot-to-shot period: the sum of the loop body's own
        durations. Exact under the scheduler's deterministic timing — the loop
        has no branching and every operation's length is known here.

        This is the telegraph timebase (the neutral layer falls back to a
        knob-only estimate when a probe does not report it), so it is measured
        off the same element the schedule above plays on.
        """
        readout_s, pi_s = _op_durations(self, target)
        return (idle_ns + depletion_ns) * 1e-9 + readout_s + 2.0 * pi_s
