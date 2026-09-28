"""``qubit_spectroscopy``: where the saturation drive sits against the readout.

This is the Qblox half of the backend-parity rule (SCQO CLAUDE.md, *Backend
parity*). The drive ENDS at an anchor and starts ``drive_len_ns`` earlier;
``readout_overlap`` picks the anchor — the readout tone's START (the default) or
its END.

NOTHING ELSE IN THIS SUITE CAN SEE THAT. ``test_probe_surface`` proves the
schedule compiles, and a drive that never turns off compiles just as happily —
which is exactly what this backend used to emit: a ``VoltageOffset`` latched
across the whole sweep, so the drive was live through every ``Measure`` while the
QM backend played a finite pulse, and the same ``scqo run qubit_spectroscopy``
measured a bare line on one instrument and an AC-Stark-shifted one on the other.
So this file walks the COMPILED tree with absolute times and asserts the instants
against each other.

Compiled, not built, for two reasons: the readout pulse is emitted as a
long-square optimization, and so is the DRIVE — any ``SquarePulse`` of 100 ns or
more is rewritten by ``compile_long_pulses_to_awg_offsets`` into a
``VoltageOffset`` pair plus a 4 ns tail. Only the compiler knows that, so a
built-tree assertion would be checking a shape the instrument never sees. It also
means the drive's extent is ``min(start) .. max(end)`` over its port rather than
one pulse's duration, which is what ``_span`` is for.

Reading the numbers below: ``acq_delay`` is measured from the readout pulse
ONSET and already carries the element's time-of-flight, so an acquisition that
starts at ``TOF + acq_start_ns`` in emit time is integrating the tone from
``acq_start_ns`` in ARRIVAL time. TOF is not part of the lead — it is the cable,
and the probe adds to it rather than replacing it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import compile_probe, make_backend, make_experiment  # noqa: E402

import scqo_qblox.experiments  # noqa: E402,F401  (import side effect: @register)
from scqo.experiments import get  # noqa: E402

DRIVE_PORT = "q1:mw"
READOUT_PORT = "q1:res"


def _events(schedule, base=0.0, out=None):
    """Every operation in the compiled tree as ``(abs_time_s, name, op)``.

    Absolute time has to ACCUMULATE down the nesting: each schedulable's
    ``abs_time`` is relative to its own (sub)schedule, and this probe's ops sit
    four levels deep inside two loop bodies plus the Measure's subschedule.
    Resolve schedulables rather than iterating ``operations`` — that dict is
    hash-keyed, so the two ``VoltageOffset(0, 0)``-shaped ops would collapse.
    """
    out = [] if out is None else out
    ops = getattr(schedule, "operations", None) or {}
    for sch in getattr(schedule, "schedulables", {}).values():
        op = ops.get(sch.get("operation_id"))
        if op is None:
            continue
        t = base + float(sch.get("abs_time") or 0.0)
        out.append((t, type(op).__name__, op))
        if hasattr(op, "body"):
            _events(op.body, t, out)
        elif hasattr(op, "operations"):
            _events(op, t, out)
    return out


def _pulses(events, port):
    """``(start_s, end_s, info)`` for every emitted pulse on one port."""
    found = []
    for t, _name, op in events:
        info = (getattr(op, "data", {}) or {}).get("pulse_info") or []
        if isinstance(info, dict):
            info = [info]
        for p in info:
            if isinstance(p, dict) and p.get("port") == port:
                start = t + float(p.get("t0") or 0.0)
                found.append((start, start + float(p.get("duration") or 0.0), p))
    return sorted(found)


def _span(events, port):
    """``(start_ns, end_ns)`` of everything emitted on one port.

    The extent, not a duration field: a long drive arrives as an offset pair plus
    a 4 ns tail, a short one as a single waveform, and this is the one reading
    that means the same thing for both.
    """
    pulses = _pulses(events, port)
    assert pulses, f"expected emitted pulses on {port}"
    return (min(s for s, _e, _p in pulses) * 1e9,
            max(e for _s, e, _p in pulses) * 1e9)


def _acquisition(events):
    """``(start_s, end_s, info)`` of the single acquisition."""
    found = []
    for t, _name, op in events:
        info = (getattr(op, "data", {}) or {}).get("acquisition_info") or []
        if isinstance(info, dict):
            info = [info]
        for a in info:
            if isinstance(a, dict):
                start = t + float(a.get("t0") or 0.0)
                found.append((start, start + float(a.get("duration") or 0.0), a))
    assert len(found) == 1, f"expected exactly one acquisition, got {len(found)}"
    return found[0]


def _run(tmp_path, roster, **params):
    """Compile the probe and return ``(events, tof_s, drive_amp, readout_s)``."""
    cls = get("qubit_spectroscopy")
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(
        cls, backend, roster,
        cls.Parameters(targets=["q1"], num_drive_freq_points=5, num_averages=2, **params),
    )
    # the two-tone probes play the drive chain's residual (spec_amp), which the
    # fixture leaves unseeded; the core run() solves it before probing
    exp.device.channel("q1", "drive").drive_power_dbm = -33.0
    drive_amp = float(exp.device.channel("q1", "drive").drive_amp)
    readout_s = float(exp.device.channel("q1", "readout").readout_duration_s)
    element = backend.device.component("fl.q1")._element
    tof = element.measure.acq_delay
    tof_s = float(tof() if callable(tof) else tof)
    return _events(compile_probe(backend, exp)), tof_s, drive_amp, readout_s


# ------------------------------------------------------- sequential (the default)

@pytest.mark.parametrize("drive_len_ns", [64.0, 600.0, 20_000.0],
                         ids=["waveform", "stitched", "long"])
def test_the_drive_ends_exactly_where_the_readout_tone_begins(
        tmp_path, roster, drive_len_ns):
    """THE parity claim, at all three emission shapes: below the 100 ns stitching
    threshold the drive is one waveform, above it an offset pair plus a tail, and
    at 20 us it is the length the QM configs actually carry. In every case it has
    a real END, and the readout tone starts there — nothing is driving while the
    ADC integrates."""
    events, _tof, _amp, _ro = _run(tmp_path, roster, drive_len_ns=drive_len_ns)

    drive_start, drive_end = _span(events, DRIVE_PORT)
    tone_start, _tone_end = _span(events, READOUT_PORT)
    assert drive_end - drive_start == pytest.approx(drive_len_ns, abs=1e-3)
    assert drive_end == pytest.approx(tone_start, abs=1e-3)


def test_the_acquisition_opens_with_the_tone_and_the_drive_is_over(tmp_path, roster):
    """The consequence worth stating separately: with no lead the ADC follows the
    tone by the cable delay alone, and by then the drive has been off since the
    tone began."""
    events, tof_s, _amp, readout_s = _run(tmp_path, roster, drive_len_ns=600.0)

    _drive_start, drive_end = _span(events, DRIVE_PORT)
    tone_start, tone_end = _span(events, READOUT_PORT)
    acq_start, _acq_end, _info = _acquisition(events)

    assert (tone_end - tone_start) == pytest.approx(readout_s * 1e9, abs=1e-3)
    assert acq_start * 1e9 == pytest.approx(tone_start + tof_s * 1e9, abs=1e-3)
    assert drive_end <= tone_start + 1e-6


# ------------------------------------------------------------------- overlap

def test_a_short_drive_ends_with_the_tone(tmp_path, roster):
    """Tone longer than the drive: they END together, so the drive starts late by
    the readout lead. On this backend that lead is a non-negative ``rel_time``
    hung off the Measure — a subschedule never gets ``_normalize_absolute_timing``,
    so a negative offset is not an option."""
    events, _tof, _amp, readout_s = _run(
        tmp_path, roster, readout_overlap=True, drive_len_ns=100.0)
    tone_ns = readout_s * 1e9
    assert tone_ns > 100.0, "fixture too short to exercise this branch"

    drive_start, drive_end = _span(events, DRIVE_PORT)
    tone_start, tone_end = _span(events, READOUT_PORT)
    assert drive_end == pytest.approx(tone_end, abs=1e-3)
    assert drive_start - tone_start == pytest.approx(tone_ns - 100.0, abs=1e-3)


def test_a_long_drive_starts_before_the_tone_and_still_ends_with_it(tmp_path, roster):
    """The case the old co-start definition could not express, and the one a real
    20 us saturation lands in: the drive simply begins before the tone and runs
    through it, so the MEASURE is what carries the offset."""
    events, _tof, _amp, readout_s = _run(
        tmp_path, roster, readout_overlap=True, drive_len_ns=20_000.0)
    tone_ns = readout_s * 1e9
    assert tone_ns < 20_000.0, "fixture too long to exercise this branch"

    drive_start, drive_end = _span(events, DRIVE_PORT)
    tone_start, tone_end = _span(events, READOUT_PORT)
    assert drive_end == pytest.approx(tone_end, abs=1e-3)
    assert tone_start - drive_start == pytest.approx(20_000.0 - tone_ns, abs=1e-3)


def test_equal_lengths_start_and_end_together(tmp_path, roster):
    """The boundary between the two anchoring branches: both leads are zero, so
    neither op is offset from the other."""
    _e, _tof, _amp, readout_s = _run(tmp_path, roster, drive_len_ns=64.0)
    tone_ns = readout_s * 1e9

    events, _tof2, _amp2, _ro2 = _run(
        tmp_path, roster, readout_overlap=True, drive_len_ns=tone_ns)
    drive = _span(events, DRIVE_PORT)
    tone = _span(events, READOUT_PORT)
    assert drive[0] == pytest.approx(tone[0], abs=1e-3)
    assert drive[1] == pytest.approx(tone[1], abs=1e-3)


@pytest.mark.parametrize("acq_start_ns", [0.0, 400.0])
def test_the_tone_is_lengthened_and_the_adc_delayed_by_the_same_lead(
        tmp_path, roster, acq_start_ns):
    """``acq_start_ns`` moves the tone END and the ADC by the same amount, or the
    integration window would run off the end of the pulse."""
    events, tof_s, _amp, readout_s = _run(
        tmp_path, roster, readout_overlap=True, drive_len_ns=20_000.0,
        acq_start_ns=acq_start_ns)

    tone_start, tone_end = _span(events, READOUT_PORT)
    assert tone_end - tone_start == pytest.approx(
        acq_start_ns + readout_s * 1e9, abs=1e-3), "tone not lengthened by the lead"

    acq_start, acq_end, _info = _acquisition(events)
    assert (acq_start * 1e9 - tone_start) == pytest.approx(
        tof_s * 1e9 + acq_start_ns, abs=1e-3), "acq_delay is not TOF + the lead"
    # the window still lands inside the tone, in the ARRIVAL frame (TOF removed)
    assert (acq_end - tof_s) * 1e9 <= tone_end + 1e-6


def test_the_adc_opens_after_both_tones_are_already_on(tmp_path, roster):
    """With a lead, the acquisition starts strictly after BOTH have started — and
    by the lead, not by an accident of the cable delay (which is why TOF is
    subtracted before comparing)."""
    events, tof_s, _amp, _ro = _run(
        tmp_path, roster, readout_overlap=True, acq_start_ns=400.0, drive_len_ns=20_000.0)

    drive_on = _span(events, DRIVE_PORT)[0]
    tone_on = _span(events, READOUT_PORT)[0]
    acq_on = (_acquisition(events)[0] - tof_s) * 1e9  # arrival frame

    assert acq_on > drive_on and acq_on > tone_on
    assert acq_on - max(drive_on, tone_on) == pytest.approx(400.0, abs=1e-3)


def test_the_next_point_starts_after_the_acquisition_finishes(tmp_path, roster):
    """The trailing re-anchor. Without it the ASAP chain would continue from
    whichever op happens to be last — for a drive shorter than the tone that is
    the drive, well before the readout ends — and each sweep point would begin
    inside the previous measurement. Silent on hardware; visible only here."""
    events, _tof, _amp, _ro = _run(
        tmp_path, roster, readout_overlap=True, acq_start_ns=400.0, drive_len_ns=100.0)

    _acq_start, acq_end, _info = _acquisition(events)
    trailing = [t for t, name, _op in events if name == "IdlePulse" and t > 0]
    assert trailing, "expected the trailing re-anchor IdlePulse"
    assert max(trailing) >= acq_end - 1e-12
