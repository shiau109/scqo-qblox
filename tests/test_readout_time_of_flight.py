"""The Qblox raw-trace time-of-flight probe.

What can be proved offline is the SCHEDULE: that it compiles at all (``Trace``
is the one acquisition protocol this driver had never used), that the trace
opens at the frame the neutral experiment resolved rather than at the element's
own ``acq_delay`` — which is the number under test — and that the pulse is
still on at the far end of the window.

What CANNOT be proved offline is the shape the cluster returns for a ``Trace``.
No Qblox run of this experiment exists; SCQO BACKLOG F22 owns that.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qblox_scheduler")

from conftest import (  # noqa: E402
    compile_probe,
    make_backend,
    make_experiment,
)

from scqo_qblox.experiments.readout_time_of_flight import (  # noqa: E402
    QbloxReadoutTimeOfFlight,
)


def _experiment(tmp_path, roster, **params):
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(
        QbloxReadoutTimeOfFlight, backend, roster,
        QbloxReadoutTimeOfFlight.Parameters(
            targets=["q1"], num_averages=10, **params),
    )
    return backend, exp


def _walk(node, out):
    """Every leaf operation, descending through sub-schedules AND loop bodies.

    A loop is a ``LoopOperation`` whose body hangs off ``control_flow_info``
    rather than off ``schedulables`` - a traversal that only follows the latter
    sees the loop and nothing inside it, which is where the whole sequence is.
    """
    for schedulable in getattr(node, "schedulables", {}).values():
        op = node.operations[schedulable["operation_id"]]
        body = None
        if hasattr(op, "data"):
            body = (op.data.get("control_flow_info") or {}).get("body")
        if body is not None:
            _walk(body, out)
        elif getattr(op, "schedulables", None):
            _walk(op, out)
        else:
            out.append((op, schedulable))
    return out


def _operations(node):
    """Leaf operations only. Matched by CLASS: a compiled acquisition's
    ``protocol`` attribute reads None here, so the class name is the identity."""
    return [op for op, _ in _walk(node, [])]


def _rel_times(node, cls_name):
    """The rel_time of every leaf of class ``cls_name``."""
    return [s["timing_constraints"][0].rel_time
            for op, s in _walk(node, [])
            if type(op).__name__ == cls_name and s["timing_constraints"]]


def test_the_schedule_compiles_with_a_trace_acquisition(tmp_path, roster):
    """Trace is the protocol no other probe here uses, and a Schedule that only
    ever gets BUILT proves nothing - the time grid and the acquisition
    declaration are enforced inside the compiler."""
    backend, exp = _experiment(tmp_path, roster, readout_len_ns=500)
    compiled = compile_probe(backend, exp)

    assert "Trace" in [type(op).__name__ for op in _operations(compiled)]


def test_the_trace_opens_at_the_resolved_frame_not_the_elements_acq_delay(
        tmp_path, roster):
    """Opening the window where the element already points would make the
    measurement work only while it was not needed: with a correct acq_delay the
    pulse is present at sample 0 and there is no baseline to threshold against.
    The window origin comes from the neutral experiment."""
    backend, exp = _experiment(tmp_path, roster, readout_len_ns=500,
                               window_start_ns=40.0)
    exp.sweep_axes = exp.define_sweep()
    schedule = exp.probe()

    assert exp.resolved_frame()["window_start_ns"] == 40.0
    # the acquisition hangs off the pulse's START by exactly the window
    assert _rel_times(schedule, "Trace") == [pytest.approx(40e-9)]


def test_the_readout_pulse_outlasts_the_window_plus_the_trace(tmp_path, roster):
    """The threshold sits between a baseline and a PLATEAU. A pulse that ends
    inside the trace takes the plateau with it and the fit reads the falling
    edge as noise."""
    backend, exp = _experiment(tmp_path, roster, readout_len_ns=500,
                               window_start_ns=40.0)
    exp.sweep_axes = exp.define_sweep()
    schedule = exp.probe()

    pulses = [op for op in _operations(schedule)
              if type(op).__name__ == "SquarePulse"]
    assert pulses, "no readout pulse in the schedule"
    trace_s = 500e-9
    assert float(pulses[0].duration) >= 40e-9 + trace_s - 1e-12


def test_the_clock_phase_is_reset_every_repetition(tmp_path, roster):
    """BinMode.AVERAGE averages the RAW trace, so a free-running NCO phase
    averages the step away - the run then reports arrival_unresolved on a setup
    that is perfectly fine. This is the same reason the QM probe calls
    reset_if_phase."""
    backend, exp = _experiment(tmp_path, roster, readout_len_ns=500)
    exp.sweep_axes = exp.define_sweep()

    names = [type(op).__name__ for op in _operations(exp.probe())]
    assert "ResetClockPhase" in names


def test_the_time_axis_is_one_sample_per_nanosecond(tmp_path, roster):
    """The digitizer runs at 1 GSa/s and the estimator reads the axis as ns
    from the window origin."""
    backend, exp = _experiment(tmp_path, roster, readout_len_ns=500)
    axes = exp.define_sweep()

    times = np.asarray(axes["readout_time_ns"])
    assert times.size == 500
    assert times[0] == 0.0 and times[1] - times[0] == 1.0


def test_the_backend_names_its_own_vendor_field_and_floor(tmp_path, roster):
    """The neutral layer learns no vendor spelling: the backend says WHICH
    VendorOnly entry holds the answer, and the hint reads path/unit/edit from
    that inventory. Qblox has no acquisition floor - unlike QM's 28 ns - and
    declares no full scale rather than guessing one, because a saturation flag
    against the wrong range is worse than the NaN an unchecked one records."""
    backend = make_backend(tmp_path, roster)
    context = backend.readout_delay_context("q1")

    assert context["field"] == "readout_acq_delay"
    assert context["field"] in backend.vendor_only()
    assert context["floor_ns"] == 0.0
    assert "full_scale_v" not in context


def test_an_unserved_target_reports_nothing_rather_than_a_default(tmp_path, roster):
    """The two-empties rule power_context follows: no readout channel means no
    acquisition path, so there is nothing to say - not a floor of 0 that reads
    as a real answer."""
    backend = make_backend(tmp_path, roster)
    assert backend.readout_delay_context("not_a_target") == {}
