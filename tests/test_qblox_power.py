"""The absolute-power surfaces on the Qblox backend: the output-att solves, the
authoritative hardware-config write surface, the dual-file save, and power_context.

Greenfield: the two chains live on DIFFERENT entities — ``fl.q1`` (readout) and
``xy1.q1`` (drive), addressed ``<line>.<target>`` since SCQO 4.0.0 — separate
channel views over the SAME vendor element, and ``power_context`` still takes
MODE names (``params.targets``) and resolves both default channels through the
roster.

Offline: a real dut fixture (SCQO/tests/demo_instr_config) + a minimal hw config —
HardwareAgent validates the config at construction, no cluster is contacted.
"""

from __future__ import annotations

import json
import math

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import make_backend, make_experiment, recording_device  # noqa: E402

from scqo_qblox.backend.qblox_backend import (  # noqa: E402
    QBLOX_NOMINAL_FULL_SCALE_DBM,
    QbloxBackend,
)


@pytest.fixture()
def backend(tmp_path, roster):
    return make_backend(tmp_path, roster)


def test_getter_math(backend):
    view = backend.device.component("fl.q1")
    view.readout_amp = 0.25
    # P = +5 (nominal full scale) - 10 (fixture att) + 20*log10(0.25)
    assert view.readout_power_dbm == pytest.approx(5.0 - 10.0 + 20 * math.log10(0.25))


def test_setter_solves_even_att_and_exact_residual(backend):
    view = backend.device.component("fl.q1")
    opts = backend._hw_agent.hardware_configuration.hardware_options

    view.readout_power_dbm = -20.0
    att = opts.output_att["q1:res-q1.ro"]
    assert att == 18 and att % 2 == 0 and 0 <= att <= 60
    amp = view.readout_amp
    assert 0.397 < amp <= 0.5  # the canonical operating window when att is unclamped
    assert view.readout_power_dbm == pytest.approx(-20.0, abs=1e-9)  # exact round-trip

    view.readout_power_dbm = -80.0  # far below reach: att clamps at 60, amp absorbs
    assert opts.output_att["q1:res-q1.ro"] == 60
    assert view.readout_power_dbm == pytest.approx(-80.0, abs=1e-9)
    assert view.readout_amp < 0.397  # below the canonical window, by necessity

    with pytest.warns(UserWarning, match="canonical operating point"):
        view.readout_power_dbm = 0.0  # att already 0: amp must exceed 0.5
    assert opts.output_att["q1:res-q1.ro"] == 0
    assert view.readout_amp == pytest.approx(10 ** (-5.0 / 20.0))

    with pytest.raises(ValueError, match="exceeds the chain maximum"):
        view.readout_power_dbm = 6.0


def test_zero_amp_power_undefined(backend):
    view = backend.device.component("fl.q1")
    view.readout_amp = 0.0
    with pytest.raises(ValueError, match="absolute power undefined"):
        _ = view.readout_power_dbm
    assert backend.device.snapshot()["fl.q1"]["readout_power_dbm"] is None


def test_drive_getter_math(backend):
    view = backend.device.component("xy1.q1")
    view.drive_amp = 0.25
    # P = +5 (nominal full scale) - 18 (fixture drive att) + 20*log10(0.25)
    assert view.drive_power_dbm == pytest.approx(5.0 - 18.0 + 20 * math.log10(0.25))


def test_drive_setter_solves_even_att_and_exact_residual(backend):
    view = backend.device.component("xy1.q1")
    opts = backend._hw_agent.hardware_configuration.hardware_options

    view.drive_power_dbm = -33.0
    att = opts.output_att["q1:mw-q1.01"]
    assert att == 30 and att % 2 == 0 and 0 <= att <= 60
    amp = view.drive_amp
    assert 0.397 < amp <= 0.5  # the canonical operating window when att is unclamped
    assert view.drive_power_dbm == pytest.approx(-33.0, abs=1e-9)  # exact round-trip
    # the readout chain is untouched by a drive solve (its own port-clock key)
    assert opts.output_att["q1:res-q1.ro"] == 10

    view.drive_power_dbm = -80.0  # far below reach: att clamps at 60, amp absorbs
    assert opts.output_att["q1:mw-q1.01"] == 60
    assert view.drive_power_dbm == pytest.approx(-80.0, abs=1e-9)
    assert view.drive_amp < 0.397  # below the canonical window, by necessity

    with pytest.warns(UserWarning, match="canonical operating point"):
        view.drive_power_dbm = 0.0  # att already 0: amp must exceed 0.5
    assert opts.output_att["q1:mw-q1.01"] == 0
    assert view.drive_amp == pytest.approx(10 ** (-5.0 / 20.0))

    with pytest.raises(ValueError, match="exceeds the chain maximum"):
        view.drive_power_dbm = 6.0


def test_unset_spec_amp_drive_power_undefined(backend):
    """The demo dut carries no spec block: spec_amp deserializes as NaN, so BOTH
    drive fields read unknown (never a NaN leaking through log10 into the config)
    until drive_power_dbm (or drive_amp) is written."""
    view = backend.device.component("xy1.q1")
    assert math.isnan(view._element.spec.spec_amp)
    with pytest.raises(ValueError, match="absolute drive power undefined"):
        _ = view.drive_power_dbm
    with pytest.raises(ValueError, match="spec_amp is unset"):
        _ = view.drive_amp
    snap = backend.device.snapshot()["xy1.q1"]
    assert snap["drive_amp"] is None and snap["drive_power_dbm"] is None


def test_save_writes_both_files_consistently(backend, roster, tmp_path):
    """save() writes hw_config.json (the runtime truth) AND keeps the copy embedded
    in dut_config.json in step — the divergence-trap regression."""
    from qblox_scheduler.backends.qblox_backend import QbloxHardwareCompilationConfig
    from qcodes import Instrument

    backend.device.component("fl.q1").readout_power_dbm = -20.0
    backend.device.component("xy1.q1").drive_power_dbm = -33.0
    backend.device.save()

    # the separate hw config re-validates and carries the new atts (from_file returns
    # a plain dict in this scheduler version — validate explicitly, as the agent does)
    hw = QbloxHardwareCompilationConfig.model_validate(
        QbloxHardwareCompilationConfig.from_file(str(tmp_path / "hw_config.json"))
    )
    assert hw.hardware_options.output_att["q1:res-q1.ro"] == 18
    assert hw.hardware_options.output_att["q1:mw-q1.01"] == 30
    # the embedded dut copy matches (no drift through a save)
    dut = json.loads((tmp_path / "dut_config.json").read_text(encoding="utf-8"))
    assert dut["hardware_config"]["hardware_options"]["output_att"]["q1:res-q1.ro"] == 18
    assert dut["hardware_config"]["hardware_options"]["output_att"]["q1:mw-q1.01"] == 30
    # ...and the solved spec_amp persisted on the element (dut side)
    assert dut["elements"]["q1"]["spec"]["spec_amp"] == pytest.approx(10 ** (-8.0 / 20.0))

    # a fresh backend from the saved pair reproduces both powers
    Instrument.close_all()
    reloaded = QbloxBackend(
        hardware_config=str(tmp_path / "hw_config.json"),
        device_config=str(tmp_path / "dut_config.json"),
        output_dir=str(tmp_path / "out2"),
        roster=roster,
    )
    assert reloaded.device.component("fl.q1").readout_power_dbm == pytest.approx(
        -20.0, abs=1e-9)
    assert reloaded.device.component("xy1.q1").drive_power_dbm == pytest.approx(
        -33.0, abs=1e-9)


def test_power_context_matches_the_views(backend):
    """power_context is addressed by MODE name (params.targets) and resolves the
    target's default readout AND drive channels through the roster."""
    readout = backend.device.component("fl.q1")
    drive = backend.device.component("xy1.q1")
    readout.readout_power_dbm = -20.0
    drive.drive_power_dbm = -33.0
    ctx = backend.power_context(["q1", "nonexistent"])
    assert ctx["q1"]["output_att_db"] == 18
    assert ctx["q1"]["pulse_amp"] == pytest.approx(readout.readout_amp)
    assert ctx["q1"]["nominal_full_scale_dbm"] == QBLOX_NOMINAL_FULL_SCALE_DBM
    assert ctx["q1"]["readout_power_dbm"] == pytest.approx(-20.0, abs=1e-9)
    # the LO the data was taken at (a hand-edited lo_freq is otherwise invisible
    # in provenance) — the fixture configures 5.8e9 for q1:res-q1.ro
    assert ctx["q1"]["readout_lo_freq_hz"] == pytest.approx(5.8e9)
    # the drive chain rides along (fixture: att written by the solve, LO 4.5e9)
    assert ctx["q1"]["drive_output_att_db"] == 30
    assert ctx["q1"]["spec_amp"] == pytest.approx(drive.drive_amp)
    assert ctx["q1"]["drive_power_dbm"] == pytest.approx(-33.0, abs=1e-9)
    assert ctx["q1"]["drive_lo_freq_hz"] == pytest.approx(4.5e9)
    assert ctx["nonexistent"] == {}  # unknown target degrades, never raises


def test_power_context_readout_survives_unset_spec(backend):
    """An element with no seeded spec_amp still reports its full readout chain —
    the drive block must never take the readout provenance down with it."""
    backend.device.component("fl.q1").readout_power_dbm = -20.0
    ctx = backend.power_context(["q1"])
    assert ctx["q1"]["readout_power_dbm"] == pytest.approx(-20.0, abs=1e-9)
    assert "drive_power_dbm" not in ctx["q1"]  # unknown, not NaN


def test_catalog_registers_absolute_punchout():
    import scqo_qblox.experiments  # noqa: F401  (side effect: @register)
    from scqo import catalog

    names = {e["name"] for e in catalog()}
    assert "resonator_spectroscopy_power_chain" in names


def test_absolute_punchout_axis_uniform_and_probe_is_per_point_1d(backend, roster):
    """The chain-stepped absolute punchout: a uniform-dB power_dbm axis from the
    core (no driver override), and a probe that builds a 1D detuning schedule
    measuring at the element's CURRENT pulse_amp (the core run() solved the chain
    for the point being acquired — no pulse_amp override in the schedule)."""
    import numpy as np

    from scqo_qblox.experiments.resonator_spectroscopy_power_chain import (
        QbloxResonatorSpectroscopyPowerChain,
    )

    exp = make_experiment(
        QbloxResonatorSpectroscopyPowerChain, backend, roster,
        QbloxResonatorSpectroscopyPowerChain.Parameters(
            targets=["q1"], max_power_dbm=-15.0, min_power_dbm=-45.0,
            num_power_points=11, num_readout_freq_points=5, num_averages=10,
        ),
    )
    axes = exp.define_sweep()

    # uniform grid straight from the core (no driver override)
    power = np.asarray(axes["power_dbm"])
    assert power[0] == -45.0 and power[-1] == -15.0
    steps = np.diff(power)
    assert np.allclose(steps, steps[0])  # equally spaced in dB

    # mimic one per-point call: the run loop swaps in the 1D detuning axis and has
    # already solved the chain (device state carries the point's amplitude)
    exp.device.channel("q1", "readout").readout_power_dbm = -30.0
    exp.sweep_axes = {"detuning_hz": axes["detuning_hz"]}
    schedule = exp.probe()

    def collect_measures(sched):
        found = []
        for op in sched.operations.values():
            body = getattr(op, "body", None)  # LoopOperation wraps a sub-schedule
            if body is not None and hasattr(body, "operations"):
                found.extend(collect_measures(body))
            elif hasattr(op, "operations"):
                found.extend(collect_measures(op))
            elif type(op).__name__ == "Measure":
                found.append(op)
        return found

    measures = collect_measures(schedule)
    assert measures, "the per-point schedule must contain a Measure"
    for op in measures:  # no per-op amplitude override: the device state rules
        overrides = op.data["gate_info"].get("device_overrides", {})
        assert "pulse_amp" not in overrides


def test_power_amp_loop_order_and_relaxation_param(backend, roster):
    """The fast absolute punchout: the amplitude axis is Python-UNROLLED (one
    NUMBER -> FREQUENCY block per power point, geometric amplitudes anchored at the
    readout channel's CURRENT readout_amp — the core run() solved the chain for the
    window top before probing) so the dBm axis is UNIFORM, and
    the per-run readout_depletion_ns sets the between-readout IdlePulse (absent
    -> the calibrated readout_depletion_s knob, and absent THAT -> the probe's
    built-in 4 ns, because punchout is a bring-up sweep that must run on a chip
    nothing has measured yet)."""
    import numpy as np

    from scqo_qblox.experiments.resonator_spectroscopy_power_amp import (
        QbloxResonatorSpectroscopyPowerAmp,
    )

    # the Session's device surface, built once; mimic run()'s boundary set-top
    # (a recorded write through the channel entity, which pushes to the vendor)
    device = recording_device(backend, roster)
    device.channel("q1", "readout").readout_power_dbm = -20.0

    def build(relax_ns):
        exp = QbloxResonatorSpectroscopyPowerAmp(
            backend,
            QbloxResonatorSpectroscopyPowerAmp.Parameters(
                targets=["q1"], num_power_points=5, num_readout_freq_points=3, num_averages=7,
                readout_depletion_ns=relax_ns,
            ),
        )
        exp.device = device
        exp.sweep_axes = exp.define_sweep()
        # the axis is the core's uniform absolute grid — no driver override
        power = np.asarray(exp.sweep_axes["power_dbm"])
        assert power[0] == -50.0 and power[-1] == -20.0
        assert np.allclose(np.diff(power), np.diff(power)[0])  # equally spaced in dBm
        return exp.probe()

    def loop_dtype(op):
        """A LoopOperation's domain is {loop Var: LinearDomain}; return its dtype name."""
        domain = getattr(op, "domain", None)
        if isinstance(domain, dict) and domain:
            return next(iter(domain.values())).dtype.name
        return getattr(getattr(domain, "dtype", None), "name", None)

    def loop_domain_chains(sched, chain, out):
        """Walk nested LoopOperations, recording the dtype chain of every branch."""
        for op in sched.operations.values():
            body = getattr(op, "body", None)
            if body is not None and hasattr(body, "operations"):
                dtype = loop_dtype(op)
                new_chain = chain + ([dtype] if dtype else [])
                out.append(new_chain)
                loop_domain_chains(body, new_chain, out)
            elif hasattr(op, "operations"):
                loop_domain_chains(op, chain, out)
        return out

    schedule = build(relax_ns=None)
    chains = loop_domain_chains(schedule, [], [])
    # the amplitude axis is UNROLLED: no AMPLITUDE loop; one NUMBER -> FREQUENCY
    # branch per power point
    deepest = max(chains, key=len)
    assert deepest == ["NUMBER", "FREQUENCY"], deepest
    assert not any("AMPLITUDE" in chain for chain in chains)
    assert sum(1 for chain in chains if chain == ["NUMBER"]) == 5  # num_power_points

    # each block's Measure carries its exact GEOMETRIC amplitude (uniform dBm):
    # top = the solved amp (prefactor exactly 1.0), bottom = top * 10**((min-max)/20)
    def collect_pulse_amps(sched, out):
        for op in sched.operations.values():
            body = getattr(op, "body", None)
            if body is not None and hasattr(body, "operations"):
                collect_pulse_amps(body, out)
            elif hasattr(op, "operations"):
                collect_pulse_amps(op, out)
            elif type(op).__name__ == "Measure":
                overrides = op.data["gate_info"].get("device_overrides", {})
                if "pulse_amp" in overrides:
                    out.append(float(overrides["pulse_amp"]))
        return out

    amps = collect_pulse_amps(schedule, [])
    top_amp = device.channel("q1", "readout").readout_amp  # the solved top amplitude
    assert len(amps) == 5 and len(set(amps)) == 5
    assert amps == sorted(amps)  # ascending: only upward power jumps between blocks
    assert amps[-1] == pytest.approx(top_amp)
    assert amps[0] == pytest.approx(top_amp * 10.0 ** ((-50.0 + 20.0) / 20.0))
    ratios = np.diff(np.log10(np.asarray(amps)))
    assert np.allclose(ratios, ratios[0])  # geometric series = uniform dB steps

    # relaxation wait: default 4 ns, parameter overrides
    def idle_durations(sched):
        out = []
        for op in sched.operations.values():
            body = getattr(op, "body", None)
            if body is not None and hasattr(body, "operations"):
                out.extend(idle_durations(body))
            elif hasattr(op, "operations"):
                out.extend(idle_durations(op))
            elif type(op).__name__ == "IdlePulse":
                out.append(float(op.duration))
        return out

    assert set(idle_durations(schedule)) == {4e-9}
    # device state is unchanged by the rebuild; parameter set -> 20 us idle
    schedule2 = build(relax_ns=20000.0)
    assert set(idle_durations(schedule2)) == {2e-5}


def test_flux_probe_uses_cw_saturation_not_x(backend, roster):
    """The flux probe drives with a weak CW saturation VoltageOffset carrying the
    drive channel's drive_amp (the drive_power_dbm residual on spec_amp) instead of
    a calibrated X pulse; readout still follows the flux return to idle (QM parity).
    Structure only — a full compile needs flux-port wiring the minimal fixture lacks."""
    from scqo_qblox.experiments.qubit_spectroscopy_flux_pulse import QbloxQubitSpectroscopyFluxPulse

    exp = make_experiment(
        QbloxQubitSpectroscopyFluxPulse, backend, roster,
        QbloxQubitSpectroscopyFluxPulse.Parameters(
            targets=["q1"], num_drive_freq_points=3, num_flux_points=5, num_averages=7,
        ),
    )
    # parks spec_amp on the drive channel; the probe reads the same knob back
    view = exp.device.channel("q1", "drive")
    view.drive_power_dbm = -33.0
    drive_amp = view.drive_amp

    exp.sweep_axes = exp.define_sweep()
    schedule = exp.probe()

    def walk(sched, out):
        for op in sched.operations.values():
            body = getattr(op, "body", None)  # LoopOperation wraps a sub-schedule
            if body is not None and hasattr(body, "operations"):
                walk(body, out)
            elif hasattr(op, "operations"):
                walk(op, out)
            else:
                out.append(op)
        return out

    ops = walk(schedule, [])
    names = {type(op).__name__ for op in ops}
    assert "X" not in names  # the calibrated pi pulse is gone
    assert {"VoltageOffset", "SetClockFrequency", "Measure"} <= names

    offsets = [op.data["pulse_info"] for op in ops if type(op).__name__ == "VoltageOffset"]
    # a CW saturation VoltageOffset on the microwave port carries the solved drive_amp
    assert any(
        o["port"] == "q1:mw" and o["clock"] == "q1.01"
        and o["offset_path_I"] == pytest.approx(drive_amp)
        for o in offsets
    ), "expected a saturation VoltageOffset(drive_amp) on the microwave port"
    # ...and it is turned back OFF on the same port (safety), and the flux line is driven
    assert any(o["port"] == "q1:mw" and o["offset_path_I"] == 0 for o in offsets)
    assert any(o["port"] == "q1:fl" for o in offsets)
