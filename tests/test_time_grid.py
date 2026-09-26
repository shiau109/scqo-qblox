"""Every schedule this driver builds must COMPILE, not merely exist.

chipA, 2026-07-26: `qubit_relaxation` died on hardware with

    ValueError: Attempting to use a time value of 2004649.6799999997 ns.
    Please ensure that the durations of operations and wait times between
    operations are multiples of 1 ns (tolerance: 1e-03 ns).

The swept wait times were floats — `np.linspace(16, 100000, 51)` steps by
1999.68 ns, so 48 of its 51 points were fractional. QM never noticed because its
shells re-round every axis to clock cycles; qblox_scheduler enforces its 1 ns
`GRID_TIME` and refuses the schedule. The offline suite missed it because
`test_probe_surface.py` then stopped at "a Schedule was built" — the grid check
only runs inside the compiler, in `HardwareAgent.run()`.

So this file compiles for real. It is the regression net for the whole class:
any future off-grid duration, from any source, fails here instead of on the
instrument. `test_probe_surface.py` now compiles too (every probe, stock
parameters); the division of labour is that it proves each probe compiles at
all, while the CASES below pin the specific WINDOWS whose naive linspace step
was fractional — coverage no catalog walk can give, because it depends on the
numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qblox_scheduler")

from conftest import compile_probe, make_backend, make_experiment  # noqa: E402

#: Windows chosen so the PRE-FIX `np.linspace` step was fractional — otherwise a
#: case proves nothing. `test_cases_would_have_failed_before_the_fix` enforces
#: that, because it is an easy mistake: trimming num_points to 7 makes
#: linspace(16, 100000, 7) land on an integer 16664 ns step and the case passes
#: for the wrong reason.
CASES = [
    # the configuration that actually failed on chipA (1999.68 ns step)
    ("qubit_relaxation", {"min_wait_ns": 16.0, "max_wait_ns": 100000.0, "num_points": 51}),
    # the stock window, equally off-grid
    ("qubit_relaxation", {"num_points": 6}),
    # echo halves tau into two arms: an ODD total would ask for half a ns
    ("qubit_echo", {"min_wait_ns": 33.0, "max_wait_ns": 100001.0, "num_points": 6}),
    ("qubit_echo", {"num_points": 6}),
    ("qubit_ramsey", {"min_idle_time_ns": 16.0, "max_idle_time_ns": 4000.0, "num_points": 6}),
    # no time axis, but it gained a Reset in v0.14.0 — prove that still compiles.
    # end_amp_factor kept low: the fixture's pi_amp x 2.0 exceeds the DAC's
    # [-1, 1] range, which is an amplitude concern, not a timing one.
    ("qubit_power_rabi", {"num_amp_points": 7, "end_amp_factor": 0.5}),
]

#: the (min, max, n) of every CASE that sweeps time, for the self-check below
_TIME_WINDOWS = [
    (16.0, 100000.0, 51), (16.0, 200000.0, 6), (33.0, 100001.0, 6),
    (32.0, 400000.0, 6), (16.0, 4000.0, 6),
]


def test_cases_would_have_failed_before_the_fix():
    """Self-check: every time-swept CASE must have a FRACTIONAL naive linspace
    step, or it does not exercise the fix at all. Guards against someone
    'simplifying' num_points into a window that divides evenly."""
    for lo, hi, n in _TIME_WINDOWS:
        assert _off_grid_ns(np.linspace(lo, hi, n)) > _TOL_NS, (
            f"linspace({lo}, {hi}, {n}) is already integral — this case would "
            f"pass with or without the fix and proves nothing")


#: qblox_scheduler's own GRID_TIME_TOLERANCE_TIME. Compare ABSOLUTELY, never with
#: np.allclose: its default rtol=1e-5 calls 40012.8 ns "equal to" 40013 ns, which
#: silently turns every check in this file into a no-op.
_TOL_NS = 1.1e-3


def _off_grid_ns(values) -> float:
    """Worst distance from a whole nanosecond, in ns."""
    values = np.asarray(values, dtype=float)
    return float(np.max(np.abs(values - np.round(values)))) if values.size else 0.0


@pytest.mark.parametrize("name,params", CASES,
                         ids=[f"{n}-{i}" for i, (n, _) in enumerate(CASES)])
def test_time_swept_probes_compile(tmp_path, roster, name, params):
    """The bug, end to end: these schedules must survive the 1 ns grid check."""
    import scqo_qblox.experiments  # noqa: F401  (registers the qblox probes)
    from scqo.experiments import get

    backend = make_backend(tmp_path, roster)
    cls = get(name)
    exp = make_experiment(cls, backend, roster,
                          cls.Parameters(targets=["q1"], num_averages=2, **params))
    compile_probe(backend, exp)  # raises ValueError on any off-grid time


@pytest.mark.parametrize("name,params", CASES,
                         ids=[f"{n}-{i}" for i, (n, _) in enumerate(CASES)])
def test_swept_times_are_whole_nanoseconds(tmp_path, roster, name, params):
    """The upstream invariant the compile test depends on: scqo hands the driver
    integer nanoseconds, and the echo family hands it EVEN ones (each probe
    plays two tau/2 arms)."""
    from scqo.experiments import get

    backend = make_backend(tmp_path, roster)
    cls = get(name)
    exp = make_experiment(cls, backend, roster,
                          cls.Parameters(targets=["q1"], num_averages=2, **params))
    axes = {k: v for k, v in exp.define_sweep().items() if k.endswith("_ns")}
    for axis, values in axes.items():  # power_rabi has none — vacuously fine
        assert _off_grid_ns(values) <= _TOL_NS, (
            f"{name}.{axis} is fractional (worst {_off_grid_ns(values):.4g} ns)")
        step = np.diff(values)
        assert float(np.max(np.abs(step - step[0]))) <= _TOL_NS, (
            f"{name}.{axis} is not uniform — the Qblox probes rebuild the axis "
            f"with linspace(first, last, size), which would re-expand it off-grid")
        if "echo" in name:
            assert not np.any(np.round(values).astype(int) % 2), (
                f"{name}.{axis} must be even (each probe plays two tau/2 arms)")


def test_thermalization_write_is_snapped_to_the_grid(tmp_path, roster):
    """The second live inlet: qubit_relaxation proposes thermalization_factor x
    t1_s, which is never on the grid by luck. Rounded (QM's decided policy for
    this field), not refused — the calibration loop must be able to write its
    own result."""
    backend = make_backend(tmp_path, roster)
    xy = backend.device.component("q1_xy")

    xy.thermalization_time_s = 2.0046496799999997e-4  # 10 x a real chipA T1
    assert xy.thermalization_time_s == pytest.approx(200465e-9, rel=0, abs=1e-15)
    assert xy.thermalization_time_s * 1e9 == pytest.approx(round(xy.thermalization_time_s * 1e9))

    with pytest.raises(ValueError, match="must be positive"):
        xy.thermalization_time_s = 0.0


def test_pi_duration_refuses_a_sub_nanosecond_length(tmp_path, roster):
    """A calibrated pulse is REFUSED rather than rounded: the stored pi_amp was
    measured against a specific length. (The reset wait rounds; the difference
    is deliberate and mirrored on QM.)"""
    backend = make_backend(tmp_path, roster)
    xy = backend.device.component("q1_xy")

    xy.pi_duration_s = 3.2e-8  # 32 ns, whole ns -> fine
    assert xy.pi_duration_s == pytest.approx(3.2e-8)
    with pytest.raises(ValueError, match="whole number of nanoseconds"):
        xy.pi_duration_s = 3.25e-8  # 32.5 ns
