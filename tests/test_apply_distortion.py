"""Unit tests for the Qblox apply_distortion CLI helpers with an INJECTED fake
session — no scqo config and no cluster. Mirrors the QM sibling's suite
(scqo-qm/tests/test_apply_distortion.py); the live ``build_session`` -> facts ->
apply path is scqo-owned.

Since SCQO 4.0.0 the taps are the flux LINE's facts (``z1.distortion_amp``: a
wire has one impulse response, however many targets ride it), so the fake
session carries the REAL fixture roster — the target -> line hop is the
roster's, not a string rule restated here.
"""

from types import SimpleNamespace

import pytest

from conftest import ROSTER_TOML

from scqo.roster import parse_components
from scqo_qblox.backend.apply_distortion import (
    BASEBAND_CLOCK,
    apply_distortion_from_state,
    clear_distortion,
)

KEY = "q1:fl-cl0.baseband"

#: the flux LINE of q1 in the fixture roster (its default flux channel z1.q1
#: rides it) — the owner of the taps and of the view the port is read from
LINE = "z1"


def _session(facts=None, runs=None, corrections=None):
    """A fake scqo Session exposing exactly what the helpers read."""
    roster = parse_components(ROSTER_TOML)
    element = SimpleNamespace(ports=SimpleNamespace(flux="q1:fl"))
    saves: list = []
    # only the flux LINE is served: the port must come from the line's view, and
    # asking for anything else (the channel z1.q1, the mode q1) is a KeyError
    views = {LINE: SimpleNamespace(_element=element)}
    device = SimpleNamespace(
        component=lambda name: views[name],
        save=lambda: saves.append(True),
        _hw_config_file="hw_config.json", _config_file="dut_config.json")
    opts = SimpleNamespace(distortion_corrections=corrections)
    hw_agent = SimpleNamespace(
        hardware_configuration=SimpleNamespace(hardware_options=opts))
    backend = SimpleNamespace(roster=roster, device=device, _hw_agent=hw_agent)
    physical = SimpleNamespace(
        get=lambda entity, field: (facts or {}).get((entity, field)))

    def load_run(run_id):
        if runs is None or run_id not in runs:
            raise KeyError(f"unknown run_id {run_id!r}")
        return runs[run_id]

    sess = SimpleNamespace(backend=backend, physical=physical, load_run=load_run)
    sess._saves = saves
    sess._opts = opts
    return sess


def _run(amps, taus, *, experiment="qubit_ramsey_cryoscope",
         outcome="successful", target="q1"):
    return {
        "record": {"experiment": experiment, "outcomes": {target: outcome}},
        "result": {"fit": {target: {"distortion_amp": amps,
                                    "distortion_tau_s": taus}}},
    }


def _facts(amps, taus):
    return {(LINE, "distortion_amp"): amps, (LINE, "distortion_tau_s"): taus}


def _stages(entry):
    return [(s.amplitude, s.time_constant) for s in
            (entry.exp0_coeffs, entry.exp1_coeffs, entry.exp2_coeffs,
             entry.exp3_coeffs) if s is not None]


def test_reads_flux_line_applies_and_saves():
    sess = _session(_facts([0.05, -0.03], [100e-9, 3000e-9]))
    out = apply_distortion_from_state("q1", session=sess)
    assert out["line"] == LINE                # fact-vs-mode bridge: q1 -> z1.q1 -> z1
    assert "channel" not in out               # the 3.x summary key is gone
    assert out["portclock"] == KEY            # flux plays on the identity clock
    entry = sess._opts.distortion_corrections[KEY]
    assert not isinstance(entry, list)        # ONE correction on a real output
    assert _stages(entry) == [(0.05, 100e-9), (-0.03, 3000e-9)]  # SECONDS
    assert out["saved"] is True and len(sess._saves) == 1
    assert out["overflow"] == [] and out["existing_taps"] == 0


def test_dry_run_and_save_false_write_no_files():
    for kwargs in ({"dry_run": True}, {"save": False}):
        sess = _session(_facts([0.05], [100e-9]))
        out = apply_distortion_from_state("q1", session=sess, **kwargs)
        assert out["saved"] is False and sess._saves == []


def test_missing_facts_refuse_by_name():
    # the refusal names the LINE the facts were looked up on
    with pytest.raises(SystemExit, match=f"no accepted distortion facts on {LINE}"):
        apply_distortion_from_state("q1", session=_session())
    # one of the pair missing is the same refusal (paired arrays)
    partial = {(LINE, "distortion_amp"): [0.05]}
    with pytest.raises(SystemExit, match="no accepted distortion facts"):
        apply_distortion_from_state("q1", session=_session(partial))


def test_run_addressed_taps_and_their_refusals():
    runs = {
        "R-OK": _run([0.02], [50e-9]),
        "R-SPEC": _run([0.04], [5e-6], experiment="qubit_spectroscopy_cryoscope"),
        "R-FAIL": _run([0.02], [50e-9], outcome="failed"),
        "R-OTHER": _run([0.02], [50e-9], experiment="qubit_ramsey"),
    }
    sess = _session(runs=runs)
    out = apply_distortion_from_state("q1", run_id="R-SPEC", session=sess)
    assert out["run_id"] == "R-SPEC" and out["kept"] == [(0.04, 5e-6)]
    with pytest.raises(SystemExit, match="not 'successful'"):
        apply_distortion_from_state("q1", run_id="R-FAIL", session=sess)
    with pytest.raises(SystemExit, match="qubit_ramsey"):
        apply_distortion_from_state("q1", run_id="R-OTHER", session=sess)
    with pytest.raises(SystemExit, match="unknown run_id"):
        apply_distortion_from_state("q1", run_id="R-NOPE", session=sess)


def test_replace_over_nonempty_warns():
    sess = _session(_facts([0.05], [100e-9]))
    apply_distortion_from_state("q1", session=sess, save=False)
    with pytest.warns(UserWarning, match="replacing 1 existing"):
        out = apply_distortion_from_state("q1", session=sess, save=False)
    assert out["existing_taps"] == 1 and len(out["kept"]) == 1


def test_extend_merges_and_repartitions_with_loud_overflow():
    """--extend cannot append (the 4-stage bank is hardware): it merges the
    existing stages with the new taps and re-ranks by |A| — the least
    significant tap overflows LOUDLY, never silently."""
    sess = _session(_facts([0.5, 0.4, 0.3, 0.2], [1e-7, 2e-7, 3e-7, 4e-7]))
    apply_distortion_from_state("q1", session=sess, save=False)
    sess.physical = SimpleNamespace(
        get=lambda entity, field: _facts([0.05], [5e-6])[(entity, field)])
    with pytest.warns(UserWarning, match="exceed the QCM's 4-stage"):
        out = apply_distortion_from_state("q1", session=sess, replace=False,
                                          save=False)
    assert out["existing_taps"] == 4
    kept_amps = [a for a, _ in out["kept"]]
    assert len(kept_amps) == 4 and 0.05 not in kept_amps  # least |A| dropped
    assert out["overflow"] == [(0.05, 5e-6)]


def test_all_overflow_refuses():
    # every tau under the 6 ns hardware floor -> nothing representable
    sess = _session(_facts([0.5, 0.2], [1e-9, 2e-9]))
    with pytest.warns(UserWarning, match="exceed the QCM's 4-stage"):
        with pytest.raises(SystemExit, match="none of the 2 tap"):
            apply_distortion_from_state("q1", session=sess)


def test_list_valued_entry_refuses():
    sess = _session(_facts([0.05], [100e-9]),
                    corrections={KEY: [{"exp0_coeffs": None}]})
    with pytest.raises(SystemExit, match="LIST of corrections"):
        apply_distortion_from_state("q1", session=sess)


def test_clear_removes_the_entry():
    sess = _session(_facts([0.05, -0.03], [100e-9, 3000e-9]))
    apply_distortion_from_state("q1", session=sess, save=False)
    out = clear_distortion("q1", session=sess, dry_run=True)
    assert out["removed"] == [(0.05, 100e-9), (-0.03, 3000e-9)]
    assert out["line"] == LINE and "channel" not in out  # same summary key
    assert out["portclock"] == KEY
    assert KEY in sess._opts.distortion_corrections  # dry-run kept it
    out = clear_distortion("q1", session=sess)
    assert KEY not in sess._opts.distortion_corrections
    assert out["saved"] is True and len(sess._saves) == 1


def test_hook_is_the_hint_name_scqo_asks_for():
    """The cross-repo contract: the hint module's HOOK constant resolves to the
    backend method, and the command it returns is this CLI, run-addressed when
    the run is known."""
    from scqo.experiments._distortion_hint import HOOK

    from scqo_qblox.backend.qblox_backend import QbloxBackend

    hook = getattr(QbloxBackend, HOOK, None)
    assert callable(hook)
    assert hook(None, "q1", "RUN-1") == (
        "python -m scqo_qblox.backend.apply_distortion --target q1 --run RUN-1")
    assert hook(None, "q2") == (
        "python -m scqo_qblox.backend.apply_distortion --target q2")


def test_baseband_clock_is_the_vendor_identity():
    from qblox_scheduler.resources import BasebandClockResource

    assert BASEBAND_CLOCK == BasebandClockResource.IDENTITY
