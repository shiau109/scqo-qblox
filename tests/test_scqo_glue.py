"""Driver-side scqo glue: the `scqo` CLI works in THIS venv + the qblox factory.

The real CLI coverage lives in SCQO/tests (test_cli_*.py) against the built-in
simulated backend; this smoke test only proves the driver-side glue: the `scqo`
command runs end-to-end in the qblox venv, the per-repo demo scripts import
cleanly from scqo.cli, and the `scqo.backends` entry point resolves to a working
factory (``build_backend(cfg, setup, roster)`` — the setup is a NAMED record,
backend + note plus the DERIVED "instrument_config" vendor folder injected by
scqo, and the roster is the device's authority on which entities exist).

Greenfield: the temp lab writes a schema-3 components.toml (modes + lines; the
readout rider mints q0_res/q0_ro, the drive rider q0_xy) plus the design.toml the
simulated vendor seeds its knobs from — without a datasheet no knob has a
standing value and every run fails pre-probe.
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

#: the whole served surface: one view class + one binding table per CHANNEL KIND
SERVED_KINDS = {"drive", "readout", "flux"}


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
        'readout = ["q0"]\n'      # mints q0_res (mode) + q0_ro (channel)
        '[lines.xy0]\n'
        'drive = ["q0"]\n',       # mints q0_xy
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
    declared Unrealized entries cover EXACTLY scqo's KNOB fields of that kind (a new
    core knob fails here until this driver binds or declines it — the combo-release
    alarm; monitors and facts are never pushed and must appear in neither), coupled
    names are real sibling knobs, the vendor-only inventory collides with no neutral
    field name, and the module is pure data (importable without qblox_scheduler —
    enforced on its import statements)."""
    import ast

    from scqo.catalog import ALL_STATIC_FIELDS, CHANNELS

    from scqo_qblox.backend import fieldmap

    assert set(fieldmap.FIELD_BINDINGS) == SERVED_KINDS
    assert set(fieldmap.UNREALIZED) <= SERVED_KINDS
    for kind in SERVED_KINDS:
        knobs = {f for f, spec in CHANNELS[kind].fields.items()
                 if spec.role == "knob"}
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

    assert not set(fieldmap.VENDOR_ONLY) & ALL_STATIC_FIELDS
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
        assert set(v.coupled) <= (set(fieldmap.VENDOR_ONLY) | ALL_STATIC_FIELDS) - {name}, name
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
    """``components()`` is the doctor's WITNESS: exactly the channel entities this
    backend serves a view for, each reported with the ROSTER's kind (a kind
    disagreement is a FAIL in scqo.checks.vendor_checks) and its derived
    operation. The pair composite is absent — Qblox exposes no gate-macro
    surface — which the doctor reports as an ordinary WARN, never a failure."""
    import pytest

    pytest.importorskip("qblox_scheduler")
    from conftest import make_backend
    from scqo.checks import FAIL, vendor_checks

    backend = make_backend(tmp_path, roster)
    inventory = backend.device.components()

    # every roster channel targets an element of the fixture dut, so all of them
    # are realized; the composite is not
    assert set(inventory) == set(roster.channels())
    for name, info in inventory.items():
        entity = roster.entities[name]
        assert info.kind == entity.kind
        assert info.target == entity.target and info.line == entity.line
    assert inventory["q1_ro"].operations == ("readout",)
    assert inventory["q1_xy"].operations == ("rx",)
    assert inventory["c12_z"].operations == ("flux_bias",)

    checks = vendor_checks(roster, inventory)
    assert not [c for c in checks if c.status == FAIL], checks
    missing = [c for c in checks if "does not realize" in c.message]
    assert len(missing) == 1 and "q1_q2" in missing[0].message  # the composite only


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
