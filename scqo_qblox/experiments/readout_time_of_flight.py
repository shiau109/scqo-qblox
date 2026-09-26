"""Qblox readout time of flight — supplies only ``probe()``.

Parameters, the edge fit and the writeback hint are inherited from
``scqo.experiments.ReadoutTimeOfFlight``. What this file owns is the one
acquisition protocol no other probe here uses: ``Trace``, the raw digitizer
samples, rather than an integrated result.

    readout port-clock : [======== readout pulse ========]
                         |<- window ->|[==== Trace ====]

The pulse and the trace sit on the SAME port-clock, which is the whole point —
what is being measured is how long that loop takes. ``Measure`` is deliberately
not used: it bundles the pulse with an integrated acquisition placed at the
element's own ``acq_delay``, i.e. at the very number under test.

Two details carry the measurement, and they are the same two as on QM:

* ``ResetClockPhase`` before every repetition. ``BinMode.AVERAGE`` averages the
  raw trace across repetitions, so without a deterministic NCO phase the signal
  averages toward zero, the step vanishes, and the fit reports
  ``arrival_unresolved`` on a setup that is perfectly fine.
* the window opens at the frame the neutral experiment resolved, NOT at the
  element's ``acq_delay``. Opening it where the element already points would
  make the measurement work only while it was not needed (see the experiment's
  module docstring).

Targets run one after another (one sub-schedule each), as the QM probe runs
them: two resonators on one feedline share an input, so a concurrent raw
capture would record both pulses superposed on one trace.

OFFLINE STATUS: this schedule compiles (``tests/test_readout_time_of_flight.py``
drives the repo's ``compile_probe`` fixture), but the SHAPE the cluster returns
for a ``Trace`` acquisition has never been seen here — no Qblox run of this
experiment exists yet. The canonicalization assumes one complex array per
``acq_channel`` over the trace samples, which is what every other protocol in
this driver returns over its swept axis. That assumption is the first thing to
check on the first real run (SCQO BACKLOG F22).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scqo import register
from scqo.experiments import ReadoutTimeOfFlight
from scqo.experiments.readout_time_of_flight import TIME_AXIS

from ._vendor import vendor_element


def _port(value: Any) -> str:
    """A port attribute from either scheduler API generation."""
    return value() if callable(value) else value


@register
class QbloxReadoutTimeOfFlight(ReadoutTimeOfFlight):
    """Build the raw-trace time-of-flight Schedule for a Qblox cluster."""

    def probe(self) -> Any:
        from qblox_scheduler import Schedule
        from qblox_scheduler.operations import IdlePulse, SquarePulse
        from qblox_scheduler.operations.acquisition_library import Trace
        from qblox_scheduler.operations.loop_domains import DType, arange
        from qblox_scheduler.enums import BinMode
        from qblox_scheduler.operations.pulse_library import ResetClockPhase

        reps = int(self.params.num_averages)
        frame = self.resolved_frame()
        window_s = float(frame["window_start_ns"]) * 1e-9
        times = np.asarray(self.sweep_axes[TIME_AXIS], dtype=float)
        trace_s = float(times.size) * float(frame["sample_ns"]) * 1e-9
        # the pulse must still be on at the far end of the trace, or the plateau
        # the threshold sits on ends inside the window
        pulse_s = window_s + trace_s

        schedule = Schedule("readout_time_of_flight")
        for qubit_name in self.params.targets:
            readout_view = self.device.channel(qubit_name, "readout")
            amp = float(readout_view.readout_amp)
            readout_port = _port(
                vendor_element(self, qubit_name, "readout").ports.readout)
            readout_clock = f"{qubit_name}.ro"

            pulse_label = f"ro_{qubit_name}"
            sub = Schedule(f"readout_time_of_flight_{qubit_name}")
            with sub.loop(arange(0, reps, 1, DType.NUMBER)):
                sub.add(ResetClockPhase(clock=readout_clock))
                sub.add(
                    SquarePulse(amp, pulse_s, port=readout_port,
                                clock=readout_clock),
                    label=pulse_label,
                )
                sub.add(
                    Trace(duration=trace_s, port=readout_port,
                          clock=readout_clock,
                          acq_channel=f"S_21_{qubit_name}",
                          bin_mode=BinMode.AVERAGE),
                    ref_op=pulse_label, ref_pt="start", rel_time=window_s,
                )
                # let the resonator empty before the next repetition, or the
                # next trace starts on the tail of this one's photons
                sub.add(IdlePulse(1e-6))
            schedule.add(sub)
        return schedule
