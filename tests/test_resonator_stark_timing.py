"""``qubit_resonator_stark``: the Stark tone, the drive and the readout, compiled.

The Qblox half of the backend-parity rule for this experiment (SCQO CLAUDE.md,
*Backend parity*; the sequence lives in scqo ``experiments/_stark_tone.py``): a
tone on the READOUT port-clock runs one depletion wait before the saturation drive
and ends WITH it, and the standard ``Measure`` starts one more depletion wait
later. ``test_probe_surface`` proves the schedule compiles; only the compiled tree
shows WHEN each piece plays, so this walks it with absolute times (the helpers of
``test_sequential_timing``, which explains why compiled rather than built).

Reading the extents: every long square pulse arrives as a ``VoltageOffset`` pair
plus a 4 ns tail, so a pulse's extent is ``min(start) .. max(end)`` of what it
emitted. The tone and the Measure share ``q1:res``; they are told apart by the
acquisition, which opens one time-of-flight after the Measure's own pulse.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import compile_probe, make_backend, make_experiment  # noqa: E402
from test_sequential_timing import _acquisition, _events, _pulses  # noqa: E402

import scqo_qblox.experiments  # noqa: E402,F401  (import side effect: @register)
from scqo.experiments import get  # noqa: E402

DRIVE_PORT = "q1:mw"
READOUT_PORT = "q1:res"


def _run(tmp_path, roster, readout_amp=None, **params):
    """Compile the probe; return ``(events, tof_s, readout_amp)``."""
    cls = get("qubit_resonator_stark")
    backend = make_backend(tmp_path, roster)
    kwargs = dict(num_drive_freq_points=5, num_amp_points=3, num_averages=2, **params)
    exp = make_experiment(cls, backend, roster, cls.Parameters(targets=["q1"], **kwargs))
    # the drive plays the chain's residual (spec_amp), which the fixture leaves
    # unseeded; the core run() solves it before probing
    exp.device.channel("q1", "drive").drive_power_dbm = -33.0
    if readout_amp is not None:
        exp.device.channel("q1", "readout").readout_amp = readout_amp
    readout_amp = float(exp.device.channel("q1", "readout").readout_amp)
    element = backend.device.component("q1_ro")._element
    tof = element.measure.acq_delay
    tof_s = float(tof() if callable(tof) else tof)
    return _events(compile_probe(backend, exp)), tof_s, readout_amp


def _split_readout_port(events, tof_s):
    """``(tone, measure)`` pulse lists on the readout port-clock, cut at the
    Measure's own pulse onset (the acquisition minus the time of flight)."""
    measure_start = _acquisition(events)[0] - tof_s
    pulses = _pulses(events, READOUT_PORT)
    tone = [p for p in pulses if p[0] < measure_start - 1e-12]
    measure = [p for p in pulses if p[0] >= measure_start - 1e-12]
    assert tone and measure, (tone, measure)
    return tone, measure


def _extent_ns(pulses):
    return (min(s for s, _e, _p in pulses) * 1e9, max(e for _s, e, _p in pulses) * 1e9)


def test_the_drive_rings_up_after_the_tone_and_ends_with_it(tmp_path, roster):
    """THE concurrency claim: the drive starts one ring-up after the tone and the
    two END together."""
    events, tof_s, _amp = _run(tmp_path, roster, drive_len_ns=2000.0,
                               readout_depletion_ns=400.0)
    tone, _measure = _split_readout_port(events, tof_s)
    tone_start, tone_end = _extent_ns(tone)
    drive_start, drive_end = _extent_ns(_pulses(events, DRIVE_PORT))
    assert tone_end - tone_start == pytest.approx(400.0 + 2000.0)
    assert drive_start - tone_start == pytest.approx(400.0)
    assert drive_end == pytest.approx(tone_end)


def test_the_standard_readout_waits_out_the_photons(tmp_path, roster):
    """The Measure starts one depletion wait after the tone ends, and plays the
    standing readout_amp — the same readout in every amplitude row."""
    events, tof_s, readout_amp = _run(tmp_path, roster, drive_len_ns=2000.0,
                                      readout_depletion_ns=400.0)
    tone, measure = _split_readout_port(events, tof_s)
    assert _extent_ns(measure)[0] - _extent_ns(tone)[1] == pytest.approx(400.0)
    onset = next(p for _s, _e, p in measure if "offset_path_I" in p)
    assert onset["offset_path_I"] == pytest.approx(readout_amp)


def test_the_tone_amplitude_is_the_swept_loop_variable(tmp_path, roster):
    """The tone plays the bare AMPLITUDE loop variable (the absolute amplitudes are
    the loop domain), never a float and never arithmetic on the variable — which
    the compiler cannot lower."""
    events, tof_s, _amp = _run(tmp_path, roster, drive_len_ns=2000.0,
                               readout_depletion_ns=400.0)
    tone, _measure = _split_readout_port(events, tof_s)
    onset = next(p for _s, _e, p in tone if "offset_path_I" in p)
    assert type(onset["offset_path_I"]).__name__ == "Variable"


def test_a_zero_depletion_lets_the_readout_touch_the_tone(tmp_path, roster):
    """0 is a governed 'no wait': the tone and the Measure then touch on their
    shared port-clock, which the compiler accepts (it refuses only an overlap),
    and the drive starts with the tone."""
    events, tof_s, _amp = _run(tmp_path, roster, drive_len_ns=400.0,
                               readout_depletion_ns=0.0)
    tone, measure = _split_readout_port(events, tof_s)
    assert _extent_ns(measure)[0] == pytest.approx(_extent_ns(tone)[1])
    assert _extent_ns(_pulses(events, DRIVE_PORT))[0] == pytest.approx(_extent_ns(tone)[0])


def test_an_over_range_tone_is_refused_by_name(tmp_path, roster):
    """prefactor x readout_amp above the sequencer's range (0.6 x 1.9 > 1) is
    refused naming the knob, before the compiler reports an internal variable
    nobody has heard of."""
    with pytest.raises(ValueError, match=r"max_amp_factor"):
        _run(tmp_path, roster, readout_amp=0.6, max_amp_factor=1.9,
             readout_depletion_ns=400.0)
