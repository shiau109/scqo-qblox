"""Driver-side scqo glue: the `scqo` CLI works in THIS venv + the qblox factory.

The real CLI coverage lives in SCQO/tests (test_cli_*.py) against the built-in
simulated backend; this smoke test only proves the driver-side glue: the `scqo`
command runs end-to-end in the qblox venv, the per-repo demo scripts import
cleanly from scqo.cli, and the `scqo.backends` entry point resolves to a working
factory (``build_backend(cfg, setup, roster)`` — the setup is a NAMED record,
backend + note plus the DERIVED "instrument_config" vendor folder injected by
scqo, and the roster is the device's authority on which entities exist).

Greenfield: the temp lab writes a schema-3 components.toml (modes + lines; the
readout rider mints the mode q0_res and the channel fl.q0, the drive rider the
channel xy0.q0 — channels are named by ADDRESS since SCQO 4.0.0) plus the
design.toml the simulated vendor seeds its knobs from — without a datasheet no
knob has a standing value and every run fails pre-probe.

It also pins the backend's ENTITY surface against the fixture roster: which
names ``component()`` serves (channels and flux LINES), which it refuses by name
(borrowed channels, operations, 3.x names), and the derived views over them —
the ``components()`` witness, ``snapshot()`` and ``line_ports()``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO.parents[0] / "SCQO" / "tests" / "demo_instr_config"

#: the whole served surface: one binding table per CHANNEL KIND (covering the
#: kind's channel AND line fields — names are unique per kind, the catalog says
#: which level each is) and one channel view class per kind, plus the flux LINE
#: view that owns the standing bias since SCQO 4.0.0
SERVED_KINDS = {"drive", "readout", "flux"}

#: the fixture roster's (conftest.ROSTER_TOML) entities, by the role they play
#: in the served surface — spelled out rather than re-derived from the roster,
#: since deriving them would test the roster against itself
DESIGNED_CHANNELS = {"fl.q1", "fl.q2", "xy1.q1", "xy2.q2", "z1.q1", "z2.q2", "zc.c12"}
FLUX_LINES = {"z1": ("q1",), "z2": ("q2",), "zc": ("c12",)}
BORROWED_CHANNELS = {"xy1.q2", "xy1.c12", "xy2.q1", "xy2.c12"}
OPERATION = "q1_q2.cz"


def _env(tmp_path: Path) -> dict:
    data_root = tmp_path / "data"
    (data_root / "simdev").mkdir(parents=True)
    (data_root / "simdev" / "cooldowns.toml").write_text(
        '[cd1]\nstart = 2026-07-01\n[cd1.setup.practice]\nbackend = "simulated"\n',
        encoding="utf-8",
    )
    # post-cutover a CONFIGURED device REQUIRES a component roster
    (data_root / "simdev" / "components.toml").write_text(
        "schema = 3\n"
        '[modes.q0]\n'
        'kind = "transmon"\n'
        '[lines.fl]\n'
        'readout = ["q0"]\n'      # mints q0_res (mode) + fl.q0 (channel)
        '[lines.xy0]\n'
        'drive = ["q0"]\n',       # mints xy0.q0
        encoding="utf-8",
    )
    # ...and a datasheet: the simulated vendor seeds readout_freq_hz from the
    # resonator's f_dress0_hz and drive_freq_hz from the qubit's f_01_hz
    (data_root / "simdev" / "design.toml").write_text(
        "schema = 1\n[q0]\nf_01_hz = 3.8e9\n[q0_res]\nf_dress0_hz = 5.95e9\n",
        encoding="utf-8",
    )
    # Always pin parameters_file (empty): without it the CLI falls back to the
    # runner's real ~/.scqo/parameters.toml, whose standing defaults can flip
    # the sim fit to failed (same guard as SCQO's test_cli_run).
    params = tmp_path / "parameters.toml"
    params.write_text("", encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text(
        f"[lab]\ndevice = \"simdev\"\ndata_root = '{data_root.as_posix()}'\n"
        f"parameters_file = '{params.as_posix()}'\n",
        encoding="utf-8",
    )
    return {**os.environ, "SCQO_CONFIG": str(config), "SCQO_USER_CONFIG": "none"}


def test_scqo_run_end_to_end(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-m", "scqo.cli", "run", "resonator_spectroscopy", "--targets", "q0"],
        capture_output=True, text=True, env=_env(tmp_path), cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.split("\nsaved:")[0])
    assert result["outcomes"] == {"q0": "successful"}


def test_ai_loop_demo_runs(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "ai_loop_demo.py")],
        capture_output=True, text=True, env=_env(tmp_path), cwd=REPO,
    )
    assert proc.returncode == 0, proc.stderr


def test_field_catalog_matches_implementation():
    """The declared field catalog cannot drift: per CHANNEL KIND, bindings plus the
    declared Unrealized entries cover EXACTLY scqo's KNOB fields of that kind at
    both levels, the channel fields and the line fields the kind puts on its wire
    (since SCQO 4.0.0 a flux line's idle_flux / flux_delay_s are LINE fields, one
    per wire). A new core knob fails here until this driver binds or declines it
    — the combo-release alarm; monitors and facts are never pushed and must
    appear in neither. Also: coupled names are real sibling knobs, the operation
    knobs bind nothing (no gate-macro surface), the vendor-only inventory
    collides with no neutral field name, and the module is pure data (importable
    without qblox_scheduler — enforced on its import statements)."""
    import ast

    from scqo.catalog import ALL_FIELD_NAMES, CHANNELS

    from scqo_qblox.backend import fieldmap

    assert set(fieldmap.FIELD_BINDINGS) == SERVED_KINDS
    assert set(fieldmap.UNREALIZED) <= SERVED_KINDS
    for kind in SERVED_KINDS:
        spec = CHANNELS[kind]
        knobs = {f for f, fs in (*spec.fields.items(), *spec.line_fields.items())
                 if fs.role == "knob"}
        bindings = fieldmap.FIELD_BINDINGS[kind]
        unrealized = fieldmap.UNREALIZED.get(kind, {})
        assert set(bindings) | set(unrealized) == knobs, kind
        assert not set(bindings) & set(unrealized)  # realized XOR unrealized
        for name, binding in bindings.items():
            assert binding.path, f"{kind}.{name}: empty vendor path"
            assert set(binding.coupled) <= knobs - {name}, name
        for name, entry in unrealized.items():
            # the scqo dataclass attribute is still spelled 'category'; since
            # the greenfield model it carries the channel KIND
            assert entry.category == kind and entry.field == name, name
            assert entry.reason, name
    # the level split since SCQO 4.0.0: everything flux binds or declines is a
    # LINE field (one bias, one delay per wire) - the flux channel is knob-free
    flux = set(fieldmap.FIELD_BINDINGS["flux"]) | set(fieldmap.UNREALIZED["flux"])
    assert flux and flux <= set(CHANNELS["flux"].line_fields)

    assert not set(fieldmap.VENDOR_ONLY) & ALL_FIELD_NAMES
    assert all(v.path and v.doc for v in fieldmap.VENDOR_ONLY.values())

    # every entry carries a valid placement-rule kind; unique entries must state
    # the lock-in fact (no counterpart on the other backend)
    from scqo.fieldmap import VENDOR_ONLY_KINDS

    for name, v in fieldmap.VENDOR_ONLY.items():
        assert v.kind in VENDOR_ONLY_KINDS, name
        if v.kind == "unique":
            assert "no qm counterpart" in v.doc.lower(), name
        # the operational half. `coupled` is checked the way binding.coupled is:
        # a typo or a half-deleted pairing must not survive as a dangling name.
        assert set(v.coupled) <= (set(fieldmap.VENDOR_ONLY) | ALL_FIELD_NAMES) - {name}, name
        # a tuple typo here would render as a Python repr on a lab console
        assert isinstance(v.edit, str) and isinstance(v.counterpart, str), name
        assert all(s.isascii() for s in (v.doc, v.edit, v.counterpart)), name
    # DELIBERATELY NOT asserted, and it must stay that way: "every non-unique
    # entry declares a counterpart" and "every realizer declares an edit". Some
    # entries have no counterpart prose to extract and inventing one would be a
    # new claim; QM's twin file has realizers whose governed write is not
    # reachable at all. Both rules would fail on day one.

    tree = ast.parse(Path(fieldmap.__file__).read_text(encoding="utf-8"))
    imported = {
        name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for name in ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module])
    }
    assert imported <= {"__future__", "scqo.fieldmap"}, imported

    # the backend class serves exactly the declared catalog (methods are pure),
    # and it serves a view class for exactly the kinds the catalog declares
    from scqo_qblox.backend.qblox_backend import _CHANNEL_VIEWS, QbloxBackend

    assert QbloxBackend.field_bindings(None) == fieldmap.FIELD_BINDINGS
    assert QbloxBackend.unrealized(None) == fieldmap.UNREALIZED
    assert QbloxBackend.vendor_only(None) == fieldmap.VENDOR_ONLY
    assert QbloxBackend.operator_commands(None) == fieldmap.OPERATOR_COMMANDS
    assert set(_CHANNEL_VIEWS) == SERVED_KINDS
    # no gate-macro surface: an operation is refused WHOLE by component(), so no
    # operation knob is bound, nor declared Unrealized one field at a time
    assert QbloxBackend.operation_bindings(None) == {}
    assert QbloxBackend.operation_unrealized(None) == {}


def test_operator_command_inventory():
    """The vendor CLIs this driver ships. They are not scqo subcommands, so
    `scqo -h` cannot show them and this inventory (rendered by
    `scqo state --fields`) is where an operator finds them instead of
    memorizing them."""
    import importlib.util

    from scqo_qblox.backend import fieldmap

    commands = fieldmap.OPERATOR_COMMANDS
    assert commands, "the driver ships operator CLIs; declaring none hides them"
    names = [c.name for c in commands]
    assert len(set(names)) == len(names), names
    for c in commands:
        assert c.name and c.command and c.doc, c.name
        assert all(s.isascii() for s in (c.name, c.command, c.doc, c.options,
                                         c.caution)), c.name
        # anti-rot: a renamed or moved operator module fails HERE, in CI, and
        # not six weeks later in the lab with a command that no longer exists
        for word in c.command.split():
            if word.startswith("scqo_qblox."):
                assert importlib.util.find_spec(word), f"{c.name}: {word}"
    # calibrate_mixers is PATH-invoked, not `python -m`, so find_spec cannot
    # reach it - check the script itself is where the inventory says it is
    script = next(c for c in commands if c.name == "calibrate_mixers")
    assert "scripts/calibrate_mixers.py" in script.command
    assert (REPO / "scripts" / "calibrate_mixers.py").is_file()


def test_distortion_hint_and_inventory_agree():
    """The cryoscope writeback hint builds a RESOLVED command (this target, this
    run) while the inventory carries a <placeholder> template, so they are two
    artifacts on purpose - but they must name the same module. Deriving one from
    the other would buy a placeholder-substitution contract nothing else needs;
    this assert buys the anti-drift property instead."""
    from scqo_qblox.backend import fieldmap
    from scqo_qblox.backend.qblox_backend import QbloxBackend

    entry = next(c for c in fieldmap.OPERATOR_COMMANDS
                 if c.name == "apply_distortion")
    prefix = entry.command.split(" --")[0]
    # distortion_apply_command uses no self, so the unbound call needs no cluster
    assert QbloxBackend.distortion_apply_command(None, "q1").startswith(prefix)


def test_backend_entry_point_resolves(tmp_path, roster):
    """The scqo.backends entry point loads and the factory fails loudly (no
    hardware needed) when the setup's folder lacks the canonical vendor files."""
    from importlib.metadata import entry_points

    import pytest

    eps = {ep.name: ep for ep in entry_points(group="scqo.backends")}
    assert "qblox" in eps, "reinstall the editable (uv pip install -e .) to register entry points"
    factory = eps["qblox"].load()

    empty = tmp_path / "empty"
    empty.mkdir()
    setup = {"backend": "qblox", "instrument_config": str(empty)}
    with pytest.raises(SystemExit, match="dut_config.json"):
        factory(None, setup, roster)
    with pytest.raises(SystemExit, match="qblox"):
        factory(None, {"backend": "qm"}, roster)  # wrong family refused


def test_components_inventory_is_a_truthful_witness(tmp_path, roster):
    """``components()`` is the doctor's WITNESS: exactly the entities this backend
    serves a view for — every designed channel, each reported with the ROSTER's
    kind (a kind disagreement is a FAIL in scqo.checks.vendor_checks), its line
    and its derived operation, plus every flux LINE (kind ``line``, over the
    target of the one flux channel it carries). The pair's declared operation is
    absent — Qblox exposes no gate-macro surface — which the doctor reports as an
    ordinary WARN, never a failure; the borrowed channels are absent too (no
    element adopts them), and are never expected."""
    import pytest

    pytest.importorskip("qblox_scheduler")
    from conftest import make_backend
    from scqo.checks import FAIL, capability_checks, vendor_checks

    backend = make_backend(tmp_path, roster)
    inventory = backend.device.components()

    # every designed channel targets an element of the fixture dut, so all of
    # them are realized, as is every line carrying flux; lines with no flux
    # (fl, xy1, xy2), the borrowed channels and the operation are not
    assert set(roster.channels()) == DESIGNED_CHANNELS  # the fixture as spelled
    assert set(inventory) == DESIGNED_CHANNELS | set(FLUX_LINES)
    for name in DESIGNED_CHANNELS:
        info, entity = inventory[name], roster.entities[name]
        assert info.kind == entity.kind
        assert info.target == entity.target and info.line == entity.line
    for name, targets in FLUX_LINES.items():
        info = inventory[name]
        assert info.kind == "line" == roster.entities[name].kind
        assert info.line == name and info.target == targets
        assert info.operations == ()
    assert inventory["fl.q1"].operations == ("readout",)
    assert inventory["xy1.q1"].operations == ("rx",)
    assert inventory["zc.c12"].operations == ("flux_bias",)

    checks = vendor_checks(roster, inventory)
    assert not [c for c in checks if c.status == FAIL], checks
    assert not [c for c in checks if "absent from the roster" in c.message], checks
    missing = [c for c in checks if "does not realize" in c.message]
    # the operation only: the knob-free flux channels are never expected, and
    # the borrowed channels only once adopted
    assert len(missing) == 1 and f"['{OPERATION}']" in missing[0].message, missing
    # the flux channels reported (kind flux, one target each) keep the chipA
    # capability witness quiet: every one targets a mode the roster gives flux
    assert capability_checks(roster, inventory) == []


def test_the_flux_line_owns_the_bias_and_its_channel_is_knob_free(tmp_path, roster):
    """SCQO 4.0.0 moved the standing bias from the flux CHANNEL to the LINE (a wire
    has one DC offset, however many targets ride it). ``component(<flux line>)``
    is the QbloxFluxLine over the element of the one flux channel the line
    carries: ``idle_flux`` IS ``element.flux_params.sweet_spot`` (realized; NaN
    refuses by name, never 0.0) and ``flux_delay_s`` is Unrealized both ways.
    The flux channel keeps only the element door; a line with no flux serves no
    view. Through the Session's surface a probe reads the bias off
    ``device.flux_line(q)`` — the channel no longer answers for it — and a
    recorded write lands on the same vendor slot."""
    import math

    import pytest

    pytest.importorskip("qblox_scheduler")
    from conftest import make_backend, recording_device

    from scqo_qblox.backend.qblox_backend import QbloxFluxChannel, QbloxFluxLine

    backend = make_backend(tmp_path, roster)
    line = backend.device.component("z1")
    assert isinstance(line, QbloxFluxLine)
    assert line.name == "z1" and line.kind == "line"
    element = line._element
    assert element.name == "q1"  # the target of z1's one flux channel, z1.q1
    assert backend.device.component("zc")._element.name == "c12"

    # the bias IS the vendor sweet spot, both directions
    assert line.idle_flux == pytest.approx(float(element.flux_params.sweet_spot))
    line.idle_flux = 0.2
    assert float(element.flux_params.sweet_spot) == pytest.approx(0.2)
    assert line.idle_flux == pytest.approx(0.2)
    for access in (lambda: line.flux_delay_s,
                   lambda: setattr(line, "flux_delay_s", 1e-8)):
        with pytest.raises(NotImplementedError, match="flux_delay_s is Unrealized"):
            access()

    # the channel is the element door and nothing else: no knob of either level
    channel = backend.device.component("z1.q1")
    assert isinstance(channel, QbloxFluxChannel)
    assert channel.name == "z1.q1" and channel.kind == "flux"
    assert channel._element is element
    for knob in ("idle_flux", "flux_delay_s"):
        assert not hasattr(channel, knob), knob
        # a write aimed at the old home fails loudly - never a silent
        # instance attribute that leaves the element untouched
        with pytest.raises(AttributeError, match=f"serves no field '{knob}'"):
            setattr(channel, knob, 0.3)
    assert float(element.flux_params.sweet_spot) == pytest.approx(0.2)

    # a line carrying no flux owns no field, so there is nothing to serve
    for name in ("fl", "xy1", "xy2"):
        with pytest.raises(KeyError, match="carries no flux channel"):
            backend.device.component(name)

    # the Session's surface: seeded from the line, written back to the element
    device = recording_device(backend, roster)
    assert device.flux_line("q1").idle_flux == pytest.approx(0.2)
    assert device.flux_line("c12").idle_flux == pytest.approx(
        float(backend.device.component("zc")._element.flux_params.sweet_spot))
    with pytest.raises(AttributeError, match="idle_flux"):
        _ = device.channel("q1", "flux").idle_flux
    device.flux_line("q1").idle_flux = 0.25
    assert float(element.flux_params.sweet_spot) == pytest.approx(0.25)

    # uncalibrated refuses by name: 0.0 is exactly where the two frames coincide
    line.idle_flux = math.nan
    with pytest.raises(ValueError, match="z1: idle_flux .* not calibrated"):
        _ = line.idle_flux


def test_borrowed_channels_and_operations_are_refused_by_name(tmp_path, roster):
    """What the roster knows and this backend does not realize is a KeyError that
    says WHY — the contract scqo degrades against (a pull seed skips it, the
    doctor witnesses the gap). A BORROWED channel (q1 driven through q2's line)
    exists in the roster, but no element adopts it; an OPERATION has no
    gate-macro surface to land on; a mode's or a composite's values are facts;
    and the 3.x rider names are simply unknown — no alias survives the cutover.
    Through the Session's surface an unadopted borrowed knob has no value to
    read, and a write surfaces this refusal before anything is recorded."""
    import pytest

    pytest.importorskip("qblox_scheduler")
    from conftest import make_backend, recording_device

    backend = make_backend(tmp_path, roster)
    assert set(roster.borrowed_channels()) == BORROWED_CHANNELS  # as spelled
    for name in sorted(BORROWED_CHANNELS):
        with pytest.raises(KeyError, match="BORROWED channel .* must adopt it"):
            backend.device.component(name)
    assert OPERATION in roster.operation_entities()
    with pytest.raises(KeyError, match="is an operation.*no gate-macro surface"):
        backend.device.component(OPERATION)
    with pytest.raises(KeyError, match="'q1_q2' is a composite"):
        backend.device.component("q1_q2")
    with pytest.raises(KeyError, match="'q1' is a mode"):
        backend.device.component("q1")
    for old in ("q1_ro", "q1_xy", "q1_z", "c12_z"):
        with pytest.raises(KeyError, match="not in this device's roster"):
            backend.device.component(old)

    device = recording_device(backend, roster)
    with pytest.raises(KeyError, match="has no value yet"):
        _ = device.channel_on("xy2", "q1").pi_amp
    with pytest.raises(KeyError, match="BORROWED channel"):
        device.channel_on("xy2", "q1").pi_amp = 0.1
    assert "xy2.q1" not in device.snapshot()  # no false state, no history


def test_snapshot_reports_the_bound_knobs_of_realized_owners(tmp_path, roster):
    """``snapshot()`` is what a pull-mode Session seeds from: ``{entity: {bound
    knob: value}}`` over the channels and flux LINES this backend realizes. The
    knob-free flux channels, the borrowed channels and the operation are absent;
    an Unrealized knob (flux_delay_s) has no vendor value to seed from, so it is
    absent from its line's entry; and an uncalibrated line seeds None — never
    NaN, never 0.0, never a crashed session."""
    import math

    import pytest

    pytest.importorskip("qblox_scheduler")
    from conftest import make_backend

    from scqo_qblox.backend.fieldmap import FIELD_BINDINGS

    backend = make_backend(tmp_path, roster)
    snap = backend.device.snapshot()

    knobbed = {"fl.q1", "fl.q2", "xy1.q1", "xy2.q2"}  # the designed, knob-carrying
    assert set(snap) == knobbed | set(FLUX_LINES)
    for name in knobbed:
        assert set(snap[name]) == set(FIELD_BINDINGS[roster.entities[name].kind]), name
    for name in FLUX_LINES:
        sweet = float(backend.device.component(name)._element.flux_params.sweet_spot)
        assert snap[name] == pytest.approx({"idle_flux": sweet}), name

    backend.device.component("z2").idle_flux = math.nan
    assert backend.device.snapshot()["z2"] == {"idle_flux": None}


def test_line_ports_come_from_the_connectivity_graph(tmp_path, roster):
    """``line_ports()`` is display only (``scqo state`` prints ``line (port)``):
    each roster LINE -> the module output its designed channels' element port is
    wired to in the hardware connectivity graph. A multiplexed feedline is ONE
    output, listed once; a line whose port the graph does not wire is omitted,
    never guessed; and with no HardwareAgent there is no graph, so nothing."""
    import json

    import pytest

    pytest.importorskip("qblox_scheduler")
    from conftest import make_backend

    from scqo_qblox.backend.qblox_backend import QbloxDeviceModel

    # the 2q fixture wires every line of the roster: both qubits' readout on one
    # QRM-RF output, a QCM-RF output per drive line, a QCM output per flux line
    room = tmp_path / "2q"
    room.mkdir()
    hw_2q = json.loads((REPO / "tests" / "fixtures" / "hw_config_2q.json")
                       .read_text(encoding="utf-8"))
    wired = make_backend(room, roster, hw_config=hw_2q)
    expected = {
        "fl": "cluster_A.module8.complex_output_0",
        "xy1": "cluster_A.module10.complex_output_0",
        "xy2": "cluster_A.module10.complex_output_1",
        "z1": "cluster_A.module6.real_output_0",
        "z2": "cluster_A.module6.real_output_1",
        "zc": "cluster_A.module6.real_output_2",
    }
    assert set(expected) == set(roster.lines())
    assert wired.line_ports() == expected
    assert wired.device.line_ports() == expected  # the backend hook delegates

    # the minimal fixture does not wire q2:mw: xy2 is omitted, the rest stand
    # (fl still resolves through q1:res, which it does wire)
    minimal = make_backend(tmp_path, roster)
    assert minimal.line_ports() == {k: v for k, v in expected.items() if k != "xy2"}

    # the same elements with no HardwareAgent: no graph to read, no raise
    bare = QbloxDeviceModel(minimal._hw_agent.quantum_device, roster)
    assert bare.line_ports() == {}


def test_real_fixture_dut_config_parses(tmp_path):
    """The lab's real dut config (SCQO/tests/demo_instr_config) deserializes through
    the same path the factory uses (parse-grade; the fixture has no hw_config)."""
    import pytest

    src = FIXTURES / "QBlox_Scheduler" / "dut_config_AS_QRC.json"
    if not src.is_file():
        pytest.skip("SCQO checkout with demo_instr_config not found side-by-side")
    shutil.copy(src, tmp_path / "dut_config.json")

    import scqo_qblox.elements  # noqa: F401  register custom element types
    from qblox_scheduler import QuantumDevice

    device = QuantumDevice.from_json_file(str(tmp_path / "dut_config.json"))
    assert device.elements  # the real device tree deserialized
