"""The thermal-reset surface on the Qblox backend.

``thermalization_time_s`` is a neutral DRIVE-channel knob (scqo_state.json,
pushed to the vendor) that maps 1:1 onto ``element.reset.duration`` — both
absolute seconds, the one field needing no conversion on either side. Because
the scheduler's ``Reset()`` gate compiles to ``IdlePulse(reset.duration)``,
every probe honours the pushed value with no probe-side argument.

Offline: a real dut fixture + a minimal hw config; no cluster is contacted.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import make_backend, make_experiment  # noqa: E402


def _ops(schedule):
    """Every leaf operation in a (possibly loop-nested) schedule."""
    out = []

    def walk(sched):
        for op in sched.operations.values():
            body = getattr(op, "body", None)  # LoopOperation wraps a sub-schedule
            if body is not None and hasattr(body, "operations"):
                walk(body)
            elif hasattr(op, "operations"):
                walk(op)
            else:
                out.append(op)

    walk(schedule)
    return out


def test_thermalization_time_round_trips_1_to_1(tmp_path, roster):
    """Absolute seconds on both sides — a pass-through, no conversion. This is
    the backend where the neutral field maps exactly onto the vendor knob."""
    backend = make_backend(tmp_path, roster)
    xy = backend.device.component("xy1.q1")
    element = backend.device.component("xy1.q1")._element

    xy.thermalization_time_s = 3.715e-4
    assert xy.thermalization_time_s == pytest.approx(3.715e-4)
    reset = element.reset.duration
    assert float(reset() if callable(reset) else reset) == pytest.approx(3.715e-4)

    with pytest.raises(ValueError, match="must be positive"):
        xy.thermalization_time_s = 0.0


def test_every_qubit_probe_resets_before_pulsing(tmp_path, roster):
    """Shot independence is the assumption the averaging rests on, so every
    probe that pulses the qubit must Reset first. qubit_power_rabi and
    qubit_ramsey used to rely on a hard-coded trailing IdlePulse(10 us) — 50x
    shorter than the configured reset and shorter than several measured T1s."""
    from scqo.experiments import get

    backend = make_backend(tmp_path, roster)
    # end_amp_factor is capped for the DAC, not for size: the fixture's pi_amp
    # (0.83) x the stock top factor overruns the sequencer's [-1, 1], which
    # _amp.check_amp_window now refuses at BUILD time. Same cap as
    # test_probe_surface.py and test_time_grid.py, for the same reason.
    caps = {"qubit_power_rabi": {"end_amp_factor": 0.5}}
    for name in ("qubit_power_rabi", "qubit_ramsey", "qubit_relaxation", "qubit_echo"):
        cls = get(name)
        exp = make_experiment(cls, backend, roster,
                              cls.Parameters(targets=["q1"], **caps.get(name, {})))
        exp.sweep_axes = exp.define_sweep()
        names = [type(op).__name__ for op in _ops(exp.probe())]
        assert "Reset" in names, f"{name}: no Reset in the schedule"


def test_per_run_override_sets_and_reverts_exactly(tmp_path, roster):
    """Unlike QM — where the wait is baked in at program BUILD time — the
    scheduler expands Reset during COMPILATION inside run(), so the override
    must bracket the whole acquire. Here we assert the set/revert contract the
    context manager provides."""
    from scqo.experiments import get

    backend = make_backend(tmp_path, roster)
    xy = backend.device.component("xy1.q1")
    xy.thermalization_time_s = 5.0e-4
    cls = get("qubit_relaxation")

    exp = make_experiment(cls, backend, roster,
                          cls.Parameters(targets=["q1"],
                                         thermalization_time_ns=8_000.0))
    with backend._thermalization_override(exp):
        assert xy.thermalization_time_s == pytest.approx(8.0e-6)
    assert xy.thermalization_time_s == pytest.approx(5.0e-4)  # exact revert

    plain = make_experiment(cls, backend, roster, cls.Parameters(targets=["q1"]))
    with backend._thermalization_override(plain):
        assert xy.thermalization_time_s == pytest.approx(5.0e-4)  # untouched
