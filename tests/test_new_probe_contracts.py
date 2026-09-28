"""Regressions for the two silent-wrong-data failures the new probes shipped with.

Both are the same shape — the probe quietly measured something OTHER than what
the dataset then claimed — and both are re-runs of a failure this stack has
already paid for once (SCQO issue #24). Compiling the schedule cannot catch
either: the programs were legal, they just answered a different question.

1. ``qubit_deterministic_benchmarking`` swept around the x180 amplitude no
   matter which gate was being benchmarked, while estimate()/update() used the
   x90 knob. scqo decides in ONE place (``amp_reference_field`` ->
   ``amp_knob(target_gate)``); the probe must READ that decision, not restate it.
2. ``broadband_qubit_spectroscopy`` measured ``targets[0]`` and tiled the result
   across every other target, so a two-qubit run reported q1's spectrum as q2's.
   Only one drive LO is ever stepped, so the honest answer is to refuse.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import (  # noqa: E402
    make_backend,
    make_experiment,
)

import scqo_qblox.experiments  # noqa: E402,F401  (import side effect: @register)
from scqo.experiments import get  # noqa: E402


def _prepared(cls, tmp_path, roster, params):
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(cls, backend, roster, params)
    exp.device.channel("q1", "drive").drive_power_dbm = -33.0
    readout = exp.device.channel("q1", "readout")
    readout.pos_g_i, readout.pos_g_q = 0.0, 0.0
    readout.pos_e_i, readout.pos_e_q = 4.0, 0.0
    readout.readout_depletion_s = 1e-6
    exp.sweep_axes = exp.define_sweep()
    return backend, exp


# ------------------------------------------------------------------ issue #24
def test_benchmarking_sweeps_the_knob_the_gate_calibrates(tmp_path, roster):
    """The played amplitude is built from ``amp_reference_field()``'s knob."""
    cls = get("qubit_deterministic_benchmarking")
    _, exp = _prepared(cls, tmp_path, roster, cls.Parameters(
        targets=["q1"], target_gate="x180", num_averages=2,
        start_amp_factor=0.3, end_amp_factor=0.5, num_amp_points=5))

    assert exp.amp_reference_field() == "pi_amp"
    # the schedule builds, and it builds off pi_amp — change the knob and the
    # played window has to move with it
    pi_amp = float(exp.device.channel("q1", "drive").pi_amp)
    assert pi_amp > 0, "fixture must seed a pi amplitude for this to mean anything"
    exp.probe()  # must not raise: the referenced knob is realized here


@pytest.mark.parametrize("gate", ["x90", "y90", "-x90"])
def test_benchmarking_refuses_a_pi_half_gate_by_name(tmp_path, roster, gate):
    """pi_amp_x90 is Unrealized on Qblox (X90 is DERIVED from rxy.amp180), so a
    pi/2 benchmark has no knob of its own to sweep or write. Refuse BY NAME
    rather than silently benchmark the pi amplitude instead."""
    cls = get("qubit_deterministic_benchmarking")
    params = cls.Parameters(
        targets=["q1"], target_gate=gate, num_averages=2,
        start_amp_factor=0.3, end_amp_factor=0.5, num_amp_points=5)
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(cls, backend, roster, params)

    assert exp.amp_reference_field() == "pi_amp_x90"
    # refused at define_sweep(), before the neutral layer reaches for the
    # unrealized anchor and reports a bare "has no value yet"
    with pytest.raises(NotImplementedError, match="pi_amp_x90"):
        exp.define_sweep()


def test_the_x90_knobs_stay_unrealized(tmp_path, roster):
    """Binding them to the x180 storage (amp180 = 2*pi_amp_x90, the shared
    rxy.beta) would let an x90 calibration overwrite the calibrated pi gate."""
    backend = make_backend(tmp_path, roster)
    # the BACKEND view, not a Session's RecordingDevice: these knobs raise from
    # the vendor view itself, which is the layer a writeback would reach
    drive = backend.device.component("xy1.q1")

    for knob in ("pi_amp_x90", "drag_beta_x90"):
        with pytest.raises(NotImplementedError, match="pi/2"):
            getattr(drive, knob)
        with pytest.raises(NotImplementedError, match="pi/2"):
            setattr(drive, knob, 0.1)

    # ...while the pi-gate DRAG coefficient IS realized, and round-trips in ns
    pi_amp_before = drive.pi_amp
    drive.drag_beta = 0.25
    assert drive.drag_beta == pytest.approx(0.25)
    # writing it leaves the pi AMPLITUDE alone -- the aliasing this pins out
    # (amp180 = 2*pi_amp_x90, and one shared beta) would have moved it
    assert drive.pi_amp == pytest.approx(pi_amp_before)


# ------------------------------------------------- one LO, one measured target
def test_broadband_qubit_refuses_more_than_one_target(tmp_path, roster):
    """It steps exactly ONE drive port-clock's LO; a second target would never
    be swept, so its row could only ever be a copy of the first."""
    cls = get("broadband_qubit_spectroscopy")
    _, exp = _prepared(cls, tmp_path, roster, cls.Parameters(
        targets=["q1", "q2"], num_averages=2))

    with pytest.raises(NotImplementedError, match="single target"):
        exp.probe()


def test_broadband_resonator_keeps_its_shared_feedline_broadcast(tmp_path, roster):
    """The resonator sibling legitimately reports one trace for every target —
    they share a feedline, and the neutral simulate() broadcasts the same way.
    Pinned so the qubit-side refusal above is not 'fixed' onto this one too."""
    cls = get("broadband_resonator_spectroscopy")
    _, exp = _prepared(cls, tmp_path, roster, cls.Parameters(
        targets=["q1", "q2"], num_averages=2))

    sim = exp.simulate({"frequency_hz": exp.sweep_axes["frequency_hz"]})
    assert sim["I"].shape[0] == 2
    assert (sim["I"][0] == sim["I"][1]).all(), "shared feedline: identical rows"


# --------------------------------------------------------- deadline estimation
def test_chunk_timeout_counts_the_thermalization(tmp_path, roster):
    """The hand-rolled formulas these replaced used a magic 500 us shot period
    and ignored the reset entirely — the underestimate issue #24 reported."""
    from scqo_qblox.backend.qblox_backend import chunk_timeout_s

    cls = get("qubit_drag_equator")
    _, exp = _prepared(cls, tmp_path, roster, cls.Parameters(
        targets=["q1"], num_averages=2, min_beta=-0.5, max_beta=0.5,
        num_beta_points=5))

    exp.device.channel("q1", "drive").thermalization_time_s = 1e-3
    short = chunk_timeout_s(exp, shots=20000, points=20)
    exp.device.channel("q1", "drive").thermalization_time_s = 10e-3
    long = chunk_timeout_s(exp, shots=20000, points=20)

    assert long > short, "a 10x longer reset must widen the deadline"
    # the formula these replaced was `max(300, shots * 2 * 500e-6 * 3)`, which
    # ignores the reset entirely and would have returned the same number twice
    assert chunk_timeout_s(exp, shots=2, points=2) >= 300, "module floor holds"
