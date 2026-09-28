"""Every Qblox probe COMPILES its Schedule against the greenfield device surface.

The cutover re-homed every probe's device READ from one per-qubit view onto the
CHANNEL ENTITY that owns the knob — ``self.device.channel(q, "readout")``
(``fl.q1``), ``...channel(q, "drive")`` (``xy1.q1``) — or, for the standing flux
bias, onto the LINE that owns it (``self.device.flux_line(q)``: ``z1``), plus
the roster-resolved raw element for the vendor-only bits (ports). A stale field
spelling or a missing entity would otherwise surface only on hardware, so this
walks the WHOLE registered Qblox catalog and exercises each probe once.

It COMPILES rather than merely builds, and the difference is the whole point.
A built Schedule is just a tree of operations; every rule that can reject one
lives in the compiler — the 1 ns time grid, the DAC range, and the latched
parameter alignment that ``readout_frequency`` violated on chipA 2026-07-26 by
ending a loop body with ``Measure(freq=...)`` (the clock restore that factory
appends is a zero-duration parameter op, which may not land on the loop's
``ControlFlowReturn``). Building it caught nothing. Compiling it catches the
whole class, for every probe, including the ones nobody thought to list.

Offline: the real dut fixture + the minimal hw config, no cluster.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import (  # noqa: E402
    ROSTER_TOML,
    compile_probe,
    make_backend,
    make_experiment,
)

import scqo_qblox.experiments  # noqa: E402,F401  (import side effect: @register)
from scqo_qblox.backend.qblox_backend import SELF_ACQUIRING_ATTR  # noqa: E402
from scqo.experiments import catalog, get  # noqa: E402

#: every experiment whose registered class comes from THIS driver
QBLOX_PROBES = sorted(
    entry["name"] for entry in catalog()
    if get(entry["name"]).__module__.startswith("scqo_qblox."))

#: keep the schedules small — this test is about the device surface, not physics
#: (the values still clear each Parameters' own minimums: >4 sweep points, >=100
#: shots; the per-shot loops are hardware loops, so the schedule stays tiny).
#: ``end_amp_factor`` is the one value chosen for the COMPILER rather than for
#: size: the fixture's pi_amp x the stock top factor exceeds the DAC's [-1, 1]
#: range, which is an amplitude concern and not what this file is about.
#: Every point-count spelling appears because ``_params`` filters per
#: experiment: ``num_amp_points`` is the amplitude capability's,
#: ``num_drive_freq_points`` / ``num_readout_freq_points`` the two detuning
#: FRAMES' (the frame is in the field name — see scqo's _capabilities/detuning),
#: and ``num_points`` the time sweeps'.
SMALL = {"num_points": 5, "num_amp_points": 5, "num_drive_freq_points": 5,
         "num_readout_freq_points": 5, "num_flux_points": 5,
         "num_power_points": 5, "num_averages": 2,
         "num_shots": 100, "end_amp_factor": 0.5,
         # the cryoscopes (dict is field-filtered per experiment): a short
         # 1-ns duration axis / few frames for the ramsey one, a small
         # log-wait x detuning grid + a short square tone for the other
         "max_duration_ns": 32, "num_frames": 4,
         "num_wait_points": 5, "max_wait_ns": 200, "drive_len_ns": 64}

#: applied ON TOP of SMALL, for the carriers SMALL alone cannot satisfy. Keyed
#: by registered name so the reason lives next to the value -- a boolean over
#: field names would have to re-derive each one.
PER_EXPERIMENT: dict[str, dict] = {
    # its amplitude window defaults to 0.9 -> 1.1: SMALL's DAC-safe 0.5 end
    # alone would leave the 0.9 start over the DAC range, so it comes down too
    "qubit_deterministic_benchmarking": {"start_amp_factor": 0.3,
                                         "end_amp_factor": 0.5},
    # RB inlines every Clifford of every sequence at every depth into ONE
    # program: the stock 30 sequences x depth 100 is far past what this test
    # needs (and past Q1ASM's program memory)
    "qubit_sqrb": {"max_circuit_depth": 4, "num_random_sequences": 2},
    # the training block is num_training_shots long and precedes the tomography
    # schedule; the stock 2000 would dominate the compile
    "qubit_tomography": {"gate_counts": [0, 1], "num_training_shots": 10},
    # beta is a DRAG coefficient, not a point count -- SMALL has no spelling for it
    "qubit_drag_equator": {"min_beta": -0.5, "max_beta": 0.5,
                           "num_beta_points": 5},
}


def _params(cls):
    fields = set(cls.Parameters.model_fields)
    overrides = {k: v for k, v in SMALL.items() if k in fields}
    overrides.update(PER_EXPERIMENT.get(cls.name, {}))
    return cls.Parameters(targets=["q1"], **overrides)


def _band_edges(exp, kind: str, fallback: float):
    """A small sub-band around the fixture's parked frequency."""
    field = "readout_freq_hz" if kind == "readout" else "drive_freq_hz"
    freq = float(getattr(exp.device.channel("q1", kind), field) or fallback)
    return freq - 10e6, freq + 10e6


#: name -> {label: zero-arg builder}, for probes whose probe() acquires. Each
#: builder must return a Schedule the cluster would really be asked to run.
SELF_ACQUIRING_BUILDS = {
    "broadband_qubit_spectroscopy": lambda exp: {
        "sub-band": lambda: exp.build_sub_schedule(
            "q1", *_band_edges(exp, "drive", 5.0e9), 5, 2)},
    "broadband_resonator_spectroscopy": lambda exp: {
        "sub-band": lambda: exp.build_sub_schedule(
            "q1", *_band_edges(exp, "readout", 7.0e9), 5, 2)},
    "qubit_drag_equator": lambda exp: {
        "per-beta": exp.build_schedule},
    "qubit_tomography": lambda exp: {
        "training": exp.build_training_schedule,
        "tomography": exp.build_tomography_schedule},
}


def test_the_whole_driver_catalog_is_covered():
    """The parametrization below is only worth as much as its list."""
    assert len(QBLOX_PROBES) == len(scqo_qblox.experiments.__all__)


@pytest.mark.parametrize("name", QBLOX_PROBES)
def test_probe_compiles(tmp_path, roster, name):
    cls = get(name)
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(cls, backend, roster, _params(cls))
    # the two-tone probes play the drive chain's residual (spec_amp), which the
    # fixture leaves unseeded (NaN); the core run() solves it before probing
    exp.device.channel("q1", "drive").drive_power_dbm = -33.0
    # qubit_thermal_population prepares only |g>, so it cannot locate |e> in its
    # own data and refuses to sweep without the stored blob centers an accepted
    # single_shot_readout leaves behind. Monitor writes, harmless to every other
    # probe here.
    readout = exp.device.channel("q1", "readout")
    readout.pos_g_i, readout.pos_g_q = 0.0, 0.0
    readout.pos_e_i, readout.pos_e_q = 4.0, 0.0
    # the parity monitors derive their fixed idle from the drive channel's stored
    # beat splitting and refuse without a GOVERNED depletion wait (their shot
    # cadence is the telegraph timebase). Both harmless to every other probe.
    exp.device.channel("q1", "drive").parity_delta_f_hz = 250e3
    readout.readout_depletion_s = 1e-6

    # compile, don't just build — the compiler is where the instrument's own
    # rules live (see this module's docstring). compile_probe sets sweep_axes
    # and calls probe() the way the Session does.
    if getattr(cls, SELF_ACQUIRING_ATTR, None):
        # Its probe() drives the cluster itself, so there is no single schedule
        # to hand the compiler — but the PIECES it runs are still schedules, and
        # they are what the instrument will actually see. SELF_ACQUIRING_BUILDS
        # names them; a probe missing from that map fails the census below
        # rather than quietly skipping, which is how ~400 lines of new probe
        # went uncompiled when this branch was first added.
        from qblox_scheduler.backends.graph_compilation import SerialCompiler

        qd = backend._hw_agent.quantum_device
        qd.hardware_config = backend._hw_agent.hardware_configuration
        for label, build in SELF_ACQUIRING_BUILDS[name](exp).items():
            compiled = SerialCompiler().compile(
                schedule=build(), config=qd.generate_compilation_config())
            assert compiled.operations, f"{name}: empty {label} schedule"
        return

    compiled = compile_probe(backend, exp)
    assert compiled.operations, f"{name}: empty schedule"


def test_every_self_acquiring_probe_declares_its_schedules():
    """The map above is the census's escape hatch; keep it honest.

    A probe that declares SELF_ACQUIRING_ATTR skips the ordinary compile path,
    so if it is also absent from SELF_ACQUIRING_BUILDS it is compiled NOWHERE.
    """
    declared = {n for n in QBLOX_PROBES if getattr(get(n), SELF_ACQUIRING_ATTR, None)}
    assert declared == set(SELF_ACQUIRING_BUILDS), (
        "self-acquiring probes without a schedule map (or a stale entry): "
        f"{declared ^ set(SELF_ACQUIRING_BUILDS)}")


def _acquisition_bins(compiled) -> int:
    """How many acquisition bins the CLUSTER will fill — the compiled truth
    behind 'shot mode appends, average mode averages'. Walks the binned
    acquisition mapping's control-flow tree, whose leaves carry the indices."""
    indices = set()

    def walk(node):
        acq_index = getattr(node, "acq_index", None)
        if acq_index is not None and hasattr(acq_index, "acq_index"):
            indices.update(acq_index.acq_index)
        for child in getattr(node, "children", None) or []:
            walk(child)

    for cluster in compiled.compiled_instructions.values():
        if not isinstance(cluster, dict):
            continue
        for module in cluster.values():
            mapping = module.get("acq_hardware_mapping") if isinstance(module, dict) else None
            for sequencer in (mapping or {}).values():
                for binned in sequencer.binned:
                    walk(binned.tree)
    return len(indices)


@pytest.mark.parametrize("name,points_field", [("readout_power", "num_amp_points"),
                                               ("readout_frequency", "num_readout_freq_points")])
def test_readout_sweeps_bin_per_shot_or_average(tmp_path, roster, name, points_field):
    """The ONE thing that makes average mode average is that the repetition
    loop's variable is not captured into the bin coords. Asserted where it is
    decidable — the COMPILED acquisition mapping: shot mode reserves a bin per
    (sweep point, prepared state, shot), average mode reserves one per (sweep
    point, prepared state) and the cluster averages num_shots repetitions into
    each. A missed label would silently merge every shot into one bin."""
    cls = get(name)
    backend = make_backend(tmp_path, roster)
    params = _params(cls)
    points = getattr(params, points_field)
    swept_bins = points * 2  # x prepared_state

    def bins(mode):
        exp = make_experiment(cls, backend, roster,
                              params.model_copy(update={"readout_mode": mode}))
        return _acquisition_bins(compile_probe(backend, exp))

    assert bins("shot") == swept_bins * params.num_shots
    assert bins("average") == swept_bins


def test_flux_probe_refuses_a_target_with_no_flux_channel(tmp_path):
    """Reaching the flux port through the target's FLUX channel makes wiring the
    guard: a qubit with no flux line refuses in the roster, by name, instead of
    failing on a missing vendor port deep inside the schedule."""
    from scqo.roster import RosterError, parse_components

    from scqo_qblox.experiments.resonator_spectroscopy_flux import (
        QbloxResonatorSpectroscopyFlux,
    )

    # same chip, but q2's flux wire was never installed
    roster = parse_components(ROSTER_TOML.replace('[lines.z2]\nflux = ["q2"]\n', ""))
    assert "z2" not in roster.entities  # the replace really removed the wire
    assert not [c.name for c in roster.channels_of("q2") if "flux" in c.kinds]
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(
        QbloxResonatorSpectroscopyFlux, backend, roster,
        _params(QbloxResonatorSpectroscopyFlux).model_copy(
            update={"targets": ["q2"]}),
    )
    exp.sweep_axes = exp.define_sweep()
    with pytest.raises(RosterError, match="no unique flux channel for 'q2'"):
        exp.probe()


#: the readout_detuning carriers — every Qblox probe that builds its OWN
#: readout-frequency NCO sweep. (readout_power sweeps amplitude at a fixed
#: frequency and has no such loop.)
READOUT_DETUNING_PROBES = [
    "readout_frequency",
    "resonator_spectroscopy",
    "resonator_spectroscopy_flux",
    "resonator_spectroscopy_power_amp",
    "resonator_spectroscopy_power_chain",
]

#: one-sided on purpose: 30 MHz of window, all but 5 MHz of it BELOW the
#: current readout frequency. A probe that re-derives `center +/- span/2`
#: reproduces the width and the point count exactly and lands 10 MHz off.
ASYMMETRIC_WINDOW = {"start_readout_detuning_hz": -25e6,
                     "end_readout_detuning_hz": 5e6,
                     "num_readout_freq_points": 5}


def _frequency_domains(operation) -> list:
    """Every FREQUENCY loop domain in a BUILT Schedule, outermost first."""
    found = []

    def walk(op):
        for domain in (getattr(op, "domain", None) or {}).values():
            if getattr(domain.dtype, "value", domain.dtype) == "frequency":
                found.append(domain)
        children = getattr(op, "operations", None)
        if isinstance(children, dict):
            for child in children.values():
                walk(child)
        body = getattr(op, "body", None)
        if body is not None:
            walk(body)

    walk(operation)
    return found


@pytest.mark.parametrize("name", READOUT_DETUNING_PROBES)
def test_asymmetric_readout_window_reaches_the_nco_unshifted(tmp_path, roster, name):
    """A one-sided readout-detuning window must arrive at the NCO sweep as
    given — the endpoint form, never `center +/- span/2`.

    This is the failure mode that produces plausible WRONG numbers instead of
    an error: re-deriving a symmetric span from the detuning axis silently
    re-centers the sweep on readout_freq_hz, keeping the width and the point
    count intact while every acquired frequency — and so every fitted one — is
    attributed to the wrong place. It matters most on exactly the sweeps that
    need the asymmetry: a punchout walks the dip DOWN from the dressed
    resonator toward the bare one, and a flux map walks it down as the qubit
    detunes."""
    cls = get(name)
    backend = make_backend(tmp_path, roster)
    exp = make_experiment(cls, backend, roster,
                          _params(cls).model_copy(update=ASYMMETRIC_WINDOW))
    exp.sweep_axes = exp.define_sweep()
    center = float(exp.device.channel("q1", "readout").readout_freq_hz)

    domains = _frequency_domains(exp.probe())
    assert domains, f"{name}: no frequency sweep in the built schedule"
    for domain in domains:  # the _amp punchout unrolls one per power block
        assert domain.start == pytest.approx(
            center + ASYMMETRIC_WINDOW["start_readout_detuning_hz"]), name
        assert domain.stop == pytest.approx(
            center + ASYMMETRIC_WINDOW["end_readout_detuning_hz"]), name
        assert domain.num == ASYMMETRIC_WINDOW["num_readout_freq_points"], name
