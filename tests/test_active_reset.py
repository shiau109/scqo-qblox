"""`reset_method="active"` on Qblox: the allow-list, the refusals, the schedule.

Active reset replaces a 1.86 ms passive wait (chipA) with ~18.8 us of
measure-and-conditionally-correct. Almost everything that can go wrong with it
is SILENT, which is why this file is mostly refusals:

* a probe that should not use it (readout swept / it IS the discriminator's
  calibration / its Reset is a driven dwell) would still return a
  plausible-looking dataset;
* an uncalibrated discriminator COMPILES CLEAN and would threshold every shot
  against zero — which is also why the guard cannot ride on
  `use_state_discrimination`: active reset does not require it;
* the wrong `acq_channel` also COMPILES CLEAN and MERGES the conditional
  measurement's bins into the probe's own, so a 5-point sweep returns 10 bins and
  the run dies later in `_to_canonical`, far from the cause;
* `thermalization_time_ns` moves a wait `ConditionalReset` never plays.

What offline CANNOT prove is that the feedback path works on the instrument: a
schedule that compiles says nothing about whether the trigger reaches the drive
sequencer. That needs the chipA walkthrough (QBLOX_training's cal19 first).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("qblox_scheduler")

from conftest import compile_probe, make_backend, make_experiment  # noqa: E402

#: the four probes that opt in — readout held at the calibrated point for the
#: whole run, and a Reset that is a real state reset. Exactly the set
#: `_state.py` serves, which is not a coincidence: both need the discriminator.
CARRIERS = ["qubit_relaxation", "qubit_ramsey", "qubit_echo", "qubit_power_rabi",
            # its saturation drive is a finite SquarePulse now, so the Reset it
            # already had became a real state reset; the readout condition never
            # moves (only the DRIVE frequency sweeps). NOT hardware-validated.
            "qubit_spectroscopy"]

#: probes that carry `reset_method` and must REFUSE 'active', with why.
DENIED = {
    "readout_frequency": "sweeps the readout frequency",
    "readout_power": "sweeps the readout amplitude",
    "single_shot_readout": "is the discriminator's own calibration",
    # both cryoscopes' Resets are real state resets at the standing bias, so
    # active reset is physically plausible — denied until the QM twins'
    # active-reset experience is validated on THIS backend's hardware.
    "qubit_ramsey_cryoscope": "not yet validated on hardware",
    # sequence-identical to plain qubit_ramsey (a CARRIER) apart from the frame
    # sweep, which touches neither the Reset nor the readout — so active reset is
    # physically valid here and this is a HARDWARE-VALIDATION gate, not a physics
    # one. The QM twin already opts in; that repo has no pending-hardware bar.
    "qubit_ramsey_phasor": "not yet validated on hardware",
    "qubit_spectroscopy_cryoscope": "not yet validated on hardware",
    # a Ramsey whose Reset and readout both stay at the idle point (the z pulse
    # only runs inside the idle), so active reset is physically valid; the QM twin
    # opts in. A hardware-validation gate, as for qubit_ramsey_phasor.
    "qubit_ramsey_flux_pulse": "not yet validated on hardware",
}

#: small but legal for every probe's own Parameters minimums. Two point-count
#: spellings because the dict is filtered per experiment: `num_amp_points` is the
#: amplitude capability's, `num_points` the time-sweep carriers'.
SMALL = {"num_points": 5, "num_amp_points": 5, "num_averages": 2,
         "end_amp_factor": 0.5, "num_shots": 100}

#: a plausible calibrated discriminator, radians (the neutral unit). Negative
#: because that is what the solve returns; see test_state_discrimination.py.
ROTATION_RAD = -0.38962897776554817
THRESHOLD = -3.25e-4

#: what resonator_spectroscopy would propose for a 1 MHz linewidth at the default
#: depletion_factor of 5: 5 / (2 pi x 1e6) = 796 ns.
DEPLETION_S = 795.77e-9

HW_2Q = Path(__file__).resolve().parent / "fixtures" / "hw_config_2q.json"


def _calibrate(exp, *targets, depletion_s=DEPLETION_S):
    """Bring the target up the way `scqo accept` would — through the RECORDING
    device, which is what the guards read. A direct element write is invisible to
    them (the recording surface seeds its cache at construction).

    Active reset needs BOTH calibrations, from two different experiments: the
    discriminator (single_shot_readout) to branch on, and the depletion wait
    (resonator_spectroscopy) to settle for. Pass ``depletion_s=None`` to leave
    the second one unseeded."""
    for target in targets:
        view = exp.device.channel(target, "readout")
        view.readout_rotation_rad = ROTATION_RAD
        view.readout_threshold = THRESHOLD
        if depletion_s is not None:
            view.readout_depletion_s = depletion_s


def _experiment(tmp_path, roster, name, *, targets=("q1",), hw_config=None, **params):
    from scqo.experiments import get

    import scqo_qblox.experiments  # noqa: F401  (registers the qblox probes)

    backend = make_backend(tmp_path, roster, hw_config=hw_config)
    cls = get(name)
    kwargs = {k: v for k, v in SMALL.items() if k in cls.Parameters.model_fields}
    kwargs.update(params)
    exp = make_experiment(cls, backend, roster,
                          cls.Parameters(targets=list(targets), **kwargs))
    if "drive_power_dbm" in cls.Parameters.model_fields:
        # the two-tone probes read the drive chain's residual (spec_amp) BEFORE
        # they reach the reset, and the fixture leaves it unseeded; the core
        # run() solves it inside drive_power_boundary, which probe()-only tests
        # never enter (test_probe_surface seeds it the same way)
        for target in targets:
            exp.device.channel(target, "drive").drive_power_dbm = -33.0
    return backend, exp


def _walk(schedule, out=None, seen=None):
    """Every operation in the tree, in SCHEDULABLE order, as (class name, op).

    Two details the naive walker gets wrong. It must resolve schedulables rather
    than iterate ``operations`` — that dict is hash-keyed, so three identical
    ConditionalResets would collapse to one entry and a rounds test would pass
    vacuously. And it must record a node BEFORE descending, because
    ``ConditionalReset`` is itself a schedule: a walker that only descends walks
    straight through it and never names it.
    """
    out = [] if out is None else out
    seen = set() if seen is None else seen
    if id(schedule) in seen:
        return out
    seen.add(id(schedule))
    ops = getattr(schedule, "operations", None) or {}
    for sch in getattr(schedule, "schedulables", {}).values():
        op = ops.get(sch.get("operation_id"))
        if op is None:
            continue
        out.append((type(op).__name__, op))
        if hasattr(op, "body"):
            _walk(op.body, out, seen)
        elif hasattr(op, "operations"):
            _walk(op, out, seen)
    return out


def _names(schedule) -> list[str]:
    return [name for name, _ in _walk(schedule)]


def _gate(op) -> dict:
    return getattr(op, "data", {}).get("gate_info") or {}


def _measures(schedule) -> list[dict]:
    """Every Measure's gate_info, in schedule order."""
    return [_gate(op) for _, op in _walk(schedule)
            if _gate(op).get("operation_type") == "measure"]


# ------------------------------------------------------------ the allow-list

def test_only_the_named_probes_opt_in():
    """THE census. Active reset is opt-in BY NAME (`supports_active_reset`), read
    with a defaulting getattr so a probe added later fails CLOSED. That only means
    something if the opted-in set stays deliberate — this test is the deliberation:
    adding a probe to it must be an edit here, with a reason."""
    import scqo_qblox.experiments  # noqa: F401
    from scqo.experiments import catalog, get

    opted = {entry["name"] for entry in catalog()
             if getattr(get(entry["name"]), "supports_active_reset", False)}
    assert opted == set(CARRIERS)


def test_denied_probes_carry_the_field_but_not_the_optin():
    """The refusal only fires for probes that actually have the parameter, so the
    DENIED list must be real. (qubit_spectroscopy_flux_pulse is deliberately NOT
    here: it is flux-tagged, carries no reset_method at all, and routes through
    add_reset only to keep the no-direct-gate invariant checkable.)"""
    import scqo_qblox.experiments  # noqa: F401
    from scqo.experiments import get

    for name in DENIED:
        cls = get(name)
        assert "reset_method" in cls.Parameters.model_fields, name
        assert not getattr(cls, "supports_active_reset", False), name


def test_the_requirement_listing_follows_the_opt_in():
    """`scqo run <name> --help` lists what a reset needs from scqo's declared
    requirements, and the shared Parameters accept `reset_method="active"` on
    every backend. `experiments/__init__.py` tells scqo that active reset is
    opt-in here, so a probe that refuses it does not list the discriminator and
    the settle it would never read, while a carrier still does."""
    import scqo_qblox.experiments  # noqa: F401
    from scqo.experiments import get

    def conditions(name):
        return {req.condition() for req in get(name).requirements()}

    listing = [name for name in DENIED if "reset_method=active" in conditions(name)]
    assert not listing, f"thermal-only probes still list active-reset needs: {listing}"
    assert "reset_method=thermal" in conditions("readout_power")
    for name in ("qubit_relaxation", "qubit_ramsey"):
        assert "reset_method=active" in conditions(name), name


# --------------------------------------------------------------- it compiles

@pytest.mark.parametrize("name", CARRIERS)
def test_active_reset_compiles(tmp_path, roster, name):
    """The real check. Building a Schedule proves nothing here: the 364 ns
    trigger delay, the one-outstanding-label rule and the trigger-address budget
    are all enforced by `compile_conditional_playback`, which the fixture picks up
    from the default compilation_passes (it declares none of its own)."""
    backend, exp = _experiment(tmp_path, roster, name, reset_method="active")
    _calibrate(exp, "q1")
    compile_probe(backend, exp)


@pytest.mark.parametrize("name", CARRIERS)
def test_active_reset_plays_a_conditional_reset(tmp_path, roster, name):
    """The conditional acquisition must be thresholded, labelled for feedback, and
    on its OWN acquisition channel."""
    backend, exp = _experiment(tmp_path, roster, name, reset_method="active")
    _calibrate(exp, "q1")
    exp.sweep_axes = exp.define_sweep()
    schedule = exp.probe()

    assert "ConditionalReset" in _names(schedule)
    assert "Reset" not in _names(schedule)
    cond = [m for m in _measures(schedule)
            if m.get("acq_protocol") == "ThresholdedAcquisition"
            and m.get("feedback_trigger_label")]
    assert cond, "no conditional (feedback-labelled) measurement in the schedule"
    for m in cond:
        assert m["feedback_trigger_label"] == "q1"


@pytest.mark.parametrize("name", CARRIERS)
def test_thermal_is_unchanged(tmp_path, roster, name):
    """Default parameters must produce byte-for-byte the old sequence — the
    passive path is what every calibration in the lab currently runs."""
    backend, exp = _experiment(tmp_path, roster, name)
    exp.sweep_axes = exp.define_sweep()
    names = _names(exp.probe())

    assert "Reset" in names
    assert "ConditionalReset" not in names


def test_the_conditional_measurement_gets_its_own_acq_channel(tmp_path, roster):
    """Load-bearing, and a SILENT failure if wrong. On the probe's own
    `S_21_<q>` channel the schedule still compiles and the two acquisitions'
    bins MERGE — a 5-point sweep comes back with 10 and the run dies later in
    `_to_canonical` as "expected shape (5,) ... got (10,)". The helper owns the
    string precisely so a probe cannot get it wrong."""
    backend, exp = _experiment(tmp_path, roster, "qubit_relaxation", reset_method="active")
    _calibrate(exp, "q1")
    exp.sweep_axes = exp.define_sweep()
    measures = _measures(exp.probe())

    channels = {m.get("acq_channel_override") for m in measures}
    assert channels == {"cond_q1", "S_21_q1"}
    for m in measures:
        if m.get("feedback_trigger_label"):
            assert m["acq_channel_override"] == "cond_q1"
        else:
            assert m["acq_channel_override"] == "S_21_q1"


# ------------------------------------------------------------- rounds + settle

@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_each_round_is_another_conditional_reset(tmp_path, roster, rounds):
    """Qblox has no repeat-until-success loop: N attempts are N sequential
    conditional resets, each paying a full readout. Counted through SCHEDULABLES,
    because `operations` is hash-keyed and would collapse identical ones."""
    backend, exp = _experiment(tmp_path, roster, "qubit_relaxation",
                               reset_method="active", active_reset_rounds=rounds)
    _calibrate(exp, "q1")
    exp.sweep_axes = exp.define_sweep()
    schedule = exp.probe()

    assert _names(schedule).count("ConditionalReset") == rounds
    compile_probe(backend, exp)


def test_the_settle_comes_from_the_knob_and_zero_switches_it_off(tmp_path, roster):
    """The settle is DEVICE STATE (`fl.q1.readout_depletion_s`, proposed by
    resonator_spectroscopy from the measured linewidth), not a per-run number
    this backend picks. 0 must stay legal — it is how you turn it off, and it is
    distinct from 'never calibrated', which refuses."""
    def idle_count(tag, depletion_s):
        room = tmp_path / tag  # a fresh dir per build: make_backend copies the dut into it
        room.mkdir()
        backend, exp = _experiment(room, roster, "qubit_relaxation",
                                   reset_method="active")
        _calibrate(exp, "q1", depletion_s=depletion_s)
        exp.sweep_axes = exp.define_sweep()
        return _names(exp.probe()).count("IdlePulse")

    # the probe already ends each iteration with one IdlePulse of its own
    assert idle_count("on", DEPLETION_S) == idle_count("off", 0.0) + 1


def test_the_settle_is_snapped_to_the_grid(tmp_path, roster):
    """A fractional nanosecond is refused by the compiler, and this value is a
    factor over 1/(2 pi kappa) — never on the grid by luck, exactly like the
    T1-derived thermal wait."""
    backend, exp = _experiment(tmp_path, roster, "qubit_relaxation",
                               reset_method="active")
    _calibrate(exp, "q1", depletion_s=1000.4e-9)
    compile_probe(backend, exp)


def test_active_refuses_an_uncalibrated_depletion(tmp_path, roster):
    """A discriminator alone is not enough. Refused rather than defaulted,
    because a silently-missing settle Stark-shifts the first pulse and shows up
    as a fitted-frequency error nobody attributes to the reset."""
    backend, exp = _experiment(tmp_path, roster, "qubit_relaxation",
                               reset_method="active")
    _calibrate(exp, "q1", depletion_s=None)
    exp.sweep_axes = exp.define_sweep()

    with pytest.raises(ValueError, match="resonator_spectroscopy"):
        exp.probe()


def test_zero_depletion_is_not_the_same_as_uncalibrated(tmp_path, roster):
    """The distinction the NaN default exists for: 'I measured this resonator and
    it needs no settle' must run, while 'nobody has measured it' must refuse."""
    backend, exp = _experiment(tmp_path, roster, "qubit_relaxation",
                               reset_method="active")
    _calibrate(exp, "q1", depletion_s=0.0)
    compile_probe(backend, exp)


# ------------------------------------------------------------------ refusals

@pytest.mark.parametrize("name", sorted(DENIED))
def test_a_denied_probe_refuses_active_by_name(tmp_path, roster, name):
    """Refused, never silently thermalized. A downgraded run completes, looks
    plausible, and only the wall clock disagrees — which nobody checks.

    Driven through ``acquire`` rather than ``probe`` because probe-side ordering
    is not guaranteed: ``qubit_spectroscopy`` reads ``drive_amp`` before it ever
    reaches the reset, so an unseeded chip would raise the knob's own KeyError
    first. Loud either way, but only the backend check is guaranteed to be the
    FIRST thing that happens — which is the reason it exists."""
    backend, exp = _experiment(tmp_path, roster, name, reset_method="active")
    _calibrate(exp, "q1")

    with pytest.raises(ValueError, match=name):
        backend.acquire(exp)


def test_the_probe_refuses_on_its_own_too(tmp_path, roster):
    """The other half: add_reset checks as well, so a probe exercised directly —
    which is how every test in this repo drives one — refuses without the backend
    in the loop."""
    backend, exp = _experiment(tmp_path, roster, "single_shot_readout",
                               reset_method="active")
    _calibrate(exp, "q1")
    exp.sweep_axes = exp.define_sweep()

    with pytest.raises(ValueError, match="single_shot_readout"):
        exp.probe()


def test_active_refuses_an_uncalibrated_discriminator(tmp_path, roster):
    """The guard must NOT ride on use_state_discrimination: active reset
    thresholds the RESET, not the data, so this combination (active + averaged
    I/Q readout) is legal and likely — and a ConditionalReset against
    acq_rotation=acq_threshold=0.0 compiles perfectly happily."""
    backend, exp = _experiment(tmp_path, roster, "qubit_relaxation",
                               reset_method="active", use_state_discrimination=False)
    exp.sweep_axes = exp.define_sweep()

    with pytest.raises(ValueError, match="single_shot_readout"):
        exp.probe()


def test_active_refuses_the_thermalization_override(tmp_path, roster):
    """thermalization_time_ns moves element.reset.duration, which
    ConditionalReset never reads — so honouring both would silently drop the one
    the caller typed. Same discipline as _thermalization_override's own refusal."""
    backend, exp = _experiment(tmp_path, roster, "qubit_relaxation",
                               reset_method="active", thermalization_time_ns=5000.0)
    _calibrate(exp, "q1")
    exp.sweep_axes = exp.define_sweep()

    with pytest.raises(ValueError, match="thermalization_time_ns"):
        exp.probe()


def test_the_backend_refuses_before_it_probes(tmp_path, roster):
    """The backstop. add_reset refuses inside every probe, but only if the probe
    remembers to call it; QbloxBackend.acquire checks independently, before any
    schedule is built and long before the instrument is touched."""
    backend, exp = _experiment(tmp_path, roster, "single_shot_readout",
                               reset_method="active")
    _calibrate(exp, "q1")

    def explode():
        raise AssertionError("probe() ran — the backstop fired too late")

    exp.probe = explode
    with pytest.raises(ValueError, match="single_shot_readout"):
        backend.acquire(exp)


# ------------------------------------------------- one trigger label at a time

def _hw_2q() -> dict:
    """A second readout+drive path, so two targets are addressable.

    Its own file rather than an edit to hw_config_min.json, which is
    make_backend's default and whose exact output_att keys test_qblox_power.py
    asserts on. (And no explanatory key inside the JSON: the vendor's
    QbloxHardwareCompilationConfig forbids extras, so a "_comment" fails
    validation.)"""
    return json.loads(HW_2Q.read_text(encoding="utf-8"))


def test_multiple_targets_stay_sequential(tmp_path, roster):
    """The compiler allows only ONE outstanding feedback label: a second
    conditional acquisition before the first is consumed is a hard error. Today's
    per-target sub-schedule pattern (`schedule.add(sub)` with no ref_op) satisfies
    that by construction — this pins it, because a future 'multiplex the targets
    for speed' refactor would break it silently at build time and loudly on the
    cluster."""
    backend, exp = _experiment(tmp_path, roster, "qubit_relaxation",
                               targets=("q1", "q2"), hw_config=_hw_2q(),
                               reset_method="active")
    _calibrate(exp, "q1", "q2")
    compile_probe(backend, exp)


def test_interleaving_two_targets_feedback_is_a_hard_error(tmp_path, roster):
    """The negative half, so the positive one cannot pass for the wrong reason —
    and a warning about how far the compiler's help extends.

    It catches the INTERLEAVED shape (acq q1, acq q2, cond q1, cond q2), which is
    exactly what "multiplex the reset across targets for speed" produces. It does
    NOT catch two whole ConditionalResets merely overlapping in time: measured,
    that compiles clean, while the vendor's own docs warn that triggers sent
    inside the same 364 ns window may not work. So sequencing is a correctness
    requirement the toolchain will only partly enforce for you."""
    from qblox_scheduler import Schedule
    from qblox_scheduler.backends.graph_compilation import SerialCompiler
    from qblox_scheduler.operations import ConditionalOperation, Measure, X

    backend = make_backend(tmp_path, roster, hw_config=_hw_2q())
    outer = Schedule("interleaved_feedback")
    for q in ("q1", "q2"):
        outer.add(Measure(q, acq_protocol="ThresholdedAcquisition",
                          feedback_trigger_label=q, acq_channel=f"cond_{q}"))
    for q in ("q1", "q2"):
        outer.add(ConditionalOperation(body=X(q), qubit_name=q), rel_time=364e-9)

    qd = backend._hw_agent.quantum_device
    qd.hardware_config = backend._hw_agent.hardware_configuration
    with pytest.raises(RuntimeError, match="not used yet"):
        SerialCompiler().compile(schedule=outer,
                                 config=qd.generate_compilation_config())


# ---------------------------------------------------------------- the decode

def test_the_conditional_channel_never_reaches_the_dataset(tmp_path, roster):
    """The conditional measurement's own result — the pre-reset excited-state
    population — is DISCARDED: `_to_canonical` reads only `S_21_*`, and the
    neutral contract has no slot for it. Known gap, pinned so it stays a
    deliberate one; the failure mode this rules out is an extra data_var
    reaching dataset.nc and breaking contract validation."""
    from scqo_qblox.backend.qblox_backend import QbloxBackend

    backend, exp = _experiment(tmp_path, roster, "qubit_relaxation",
                               reset_method="active")
    _calibrate(exp, "q1")
    exp.sweep_axes = exp.define_sweep()
    n = exp.sweep_axes["wait_time_ns"].size

    signal = xr.DataArray(np.linspace(1.0, 0.1, n) + 0j, dims=("acq_index",))
    signal.attrs["acq_protocol"] = "SSBIntegrationComplex"
    cond = xr.DataArray(np.zeros(n) + 0j, dims=("acq_index_cond",))
    cond.attrs["acq_protocol"] = "ThresholdedAcquisition"
    raw = xr.Dataset({"S_21_q1": signal, "cond_q1": cond})

    ds = QbloxBackend._to_canonical(raw, exp)

    assert set(ds.data_vars) == {"I", "Q"}
    assert "cond_q1" not in ds.data_vars
    exp.Contract.validate(ds)


# ------------------------------------------------------------- the one door

def test_no_probe_constructs_the_reset_gate_directly():
    """`_reset.py` is the only place either reset gate is built. This is what
    makes the opt-in fail CLOSED in practice: a probe copied from a neighbour
    cannot quietly bypass the guard, the acq-channel rule and the settle."""
    probes = Path(__file__).resolve().parents[1] / "scqo_qblox" / "experiments"
    offenders = []
    for path in sorted(probes.glob("*.py")):
        if path.name == "_reset.py":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if "Reset(" in code:
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "build the reset through experiments/_reset.py::add_reset:\n  "
        + "\n  ".join(offenders))
