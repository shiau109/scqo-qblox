"""vendor_config_snapshot: the setup snapshot IS what save() would write, from memory.

Offline (no cluster): the real dut fixture + the minimal hw config. The hook must be
deterministic (a content hash rests on it), must never write, must show the config
the run EXECUTES (the agent's hardware_configuration, not the disk copy) and must
never carry the explicit nulls that crash the compiler on reload.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import HW_MIN, make_backend  # noqa: E402

from scqo_qblox.backend.qblox_backend import QbloxDeviceModel  # noqa: E402


def _walk_nulls(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_nulls(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_nulls(v, f"{path}[{i}]")
    elif obj is None:
        yield path


def _listing(folder: Path) -> dict[str, float]:
    return {p.name: p.stat().st_mtime_ns for p in folder.iterdir() if p.is_file()}


def test_snapshot_is_deterministic_and_writes_nothing(tmp_path, roster):
    backend = make_backend(tmp_path, roster)
    before = _listing(tmp_path)
    first = backend.vendor_config_snapshot()
    second = backend.vendor_config_snapshot()
    assert set(first) == {"dut_config.json", "hw_config.json"}
    assert first == second  # byte-identical: the content hash is stable
    for text in first.values():
        assert text.endswith("\n") and "\r" not in text
        json.loads(text)
    assert _listing(tmp_path) == before  # nothing created, nothing touched
    assert "scqo-qblox" in backend.versions()


def test_snapshot_shows_the_executed_config_not_the_disk_copy(tmp_path, roster):
    """A power written in memory (not saved) is what the next run compiles against
    - and what the snapshot reports - while hw_config.json on disk still holds
    the fixture's attenuation. The hook must not re-point qd.hardware_config."""
    backend = make_backend(tmp_path, roster)
    qd = backend._hw_agent.quantum_device
    embedded_before = qd.hardware_config
    backend.device.component("fl.q1").readout_power_dbm = -20.0  # solves att 18

    snap = backend.vendor_config_snapshot()
    hw = json.loads(snap["hw_config.json"])
    dut = json.loads(snap["dut_config.json"])
    assert hw["hardware_options"]["output_att"]["q1:res-q1.ro"] == 18
    assert dut["hardware_config"]["hardware_options"]["output_att"]["q1:res-q1.ro"] == 18
    on_disk = json.loads((tmp_path / "hw_config.json").read_text(encoding="utf-8"))
    fixture = json.loads(HW_MIN.read_text(encoding="utf-8"))
    assert on_disk == fixture  # untouched
    assert qd.hardware_config is embedded_before  # read-only hook


def test_snapshot_carries_no_null_channel_descriptions(tmp_path, roster):
    """The chipA 2026-07-16 crash trigger can never reach a snapshot: the embedded
    hardware_config is the exclude_none dump, even from a null-poisoned file."""
    hw = json.loads(HW_MIN.read_text(encoding="utf-8"))
    hw["hardware_description"]["cluster_A"]["modules"]["8"].update(
        {"complex_output_0": None, "complex_input_0": None,
         "digital_output_0": None, "digital_output_1": None})
    backend = make_backend(tmp_path, roster, hw_config=hw)
    snap = backend.vendor_config_snapshot()
    assert list(_walk_nulls(json.loads(snap["hw_config.json"]))) == []
    dut = json.loads(snap["dut_config.json"])
    assert list(_walk_nulls(dut["hardware_config"])) == []


def test_snapshot_equals_the_files_save_writes(tmp_path, roster):
    """The snapshot is the save() serialisation: after a save the texts parse equal
    to both files on disk (save() shares config_texts, so they cannot diverge)."""
    backend = make_backend(tmp_path, roster)
    backend.device.component("fl.q1").readout_power_dbm = -20.0
    backend.device.component("xy1.q1").drive_power_dbm = -33.0
    snap_before = backend.vendor_config_snapshot()
    backend.device.save()
    for name, text in snap_before.items():
        assert json.loads(text) == json.loads((tmp_path / name).read_text(encoding="utf-8")), name
    assert backend.vendor_config_snapshot() == snap_before  # save() changed nothing in memory


def test_snapshot_degrades_without_a_hardware_agent(roster):
    """No agent = no hardware config to snapshot: {} rather than an error."""
    from qblox_scheduler import QuantumDevice
    from qcodes import Instrument

    Instrument.close_all()
    model = QbloxDeviceModel(QuantumDevice("bare"), roster, config_file=None,
                             hw_agent=None, hw_config_file=None)
    assert model.config_texts() == {}
