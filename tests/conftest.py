"""Shared fixtures for the Qblox driver tests (greenfield entity model).

The backend now takes the device ROSTER: a driver serves a view per ENTITY,
addressed since SCQO 4.0.0 by its ADDRESS — the channels ``<line>.<target>``
(``fl.q1`` / ``xy1.q1`` / ``z1.q1``) and the flux LINE itself (``z1``, the
owner of the standing bias) — and only the roster says what those names mean:
which kind's knobs they carry and which vendor element they land on.
:data:`ROSTER_TOML` describes the real dut fixture
(``SCQO/tests/demo_instr_config`` — elements q1, q2 and the tunable coupler
c12) in the schema-3 components.toml vocabulary (unchanged by 4.0.0).

Probes read their neutral state through ``self.device.channel(target, kind)``
(the standing flux bias through ``self.device.flux_line(target)``), which the
Session serves as a :class:`~scqo.device.RecordingDevice` over the
vendor tree; :func:`recording_device` builds the same surface for a test that
drives a probe directly (store path ``None`` = validated, not persisted).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO.parents[0] / "SCQO" / "tests" / "demo_instr_config"
DUT_FIXTURE = FIXTURES / "QBlox_Scheduler" / "dut_config_AS_QRC.json"
HW_MIN = REPO / "tests" / "fixtures" / "hw_config_min.json"

#: The fixture chip in the greenfield schema: ONE multiplexed feedline (the
#: readout riders mint q1_res/fl.q1 and q2_res/fl.q2), a drive and a flux wire
#: per qubit (xy1.q1 + the line z1 over its channel z1.q1, likewise xy2.q2 and
#: z2/z2.q2), and the tunable coupler c12 — an ordinary flux_transmon mode with
#: its own flux wire (zc/zc.c12), referenced as the pair's coupler. Every drive
#: line also LENDS itself to the other drivable modes (the borrowed xy1.q2,
#: xy1.c12, xy2.q1, xy2.c12), which no element realizes. The pair's declared
#: operation (q1_q2.cz) carries no Qblox surface (no gate macros), which is
#: exactly what components() must report.
ROSTER_TOML = """\
schema = 3

[modes.q1]
kind = "flux_transmon"

[modes.q2]
kind = "flux_transmon"

[modes.c12]
kind = "flux_transmon"

[composites.q1_q2]
kind       = "qubit_pair"
high       = "q2"
low        = "q1"
coupler    = "c12"
operations = ["cz"]

[lines.fl]
readout = ["q1", "q2"]

[lines.xy1]
drive = ["q1"]

[lines.xy2]
drive = ["q2"]

[lines.z1]
flux = ["q1"]

[lines.z2]
flux = ["q2"]

[lines.zc]
flux = ["c12"]
"""


@pytest.fixture(autouse=True)
def _fresh_instruments():
    """qblox's QuantumDevice registers as a qcodes Instrument singleton — close
    them after every test so the next backend can claim the same names."""
    yield
    try:
        from qcodes import Instrument
    except ModuleNotFoundError:  # no vendor stack installed: nothing to close
        return
    Instrument.close_all()


@pytest.fixture()
def roster():
    """The fixture chip's validated roster (parsed by the real loader)."""
    from scqo.roster import parse_components

    return parse_components(ROSTER_TOML)


def make_backend(tmp_path: Path, roster, *, hw_config: dict | None = None):
    """A QbloxBackend over a COPY of the real dut fixture + a hw config.

    ``hw_config`` defaults to the minimal fixture; pass a mutated dict to test
    a poisoned file. No cluster is contacted (HardwareAgent validates only).
    """
    if not DUT_FIXTURE.is_file():
        pytest.skip("SCQO checkout with demo_instr_config not found side-by-side")
    from scqo_qblox.backend.qblox_backend import QbloxBackend

    shutil.copy(DUT_FIXTURE, tmp_path / "dut_config.json")
    hw = hw_config if hw_config is not None else json.loads(
        HW_MIN.read_text(encoding="utf-8"))
    (tmp_path / "hw_config.json").write_text(json.dumps(hw, indent=2),
                                             encoding="utf-8")
    return QbloxBackend(
        hardware_config=str(tmp_path / "hw_config.json"),
        device_config=str(tmp_path / "dut_config.json"),
        output_dir=str(tmp_path / "out"),
        roster=roster,
    )


def recording_device(backend, roster):
    """The device surface an experiment reads through — what the Session hands
    to ``exp.device``: channel and flux-line views over the vendor tree, knobs
    seeded from the instrument (pull), backed by an in-memory store."""
    from scqo.device import RecordingDevice
    from scqo.stores import state_store

    return RecordingDevice(backend.device, roster, state_store(None, roster))


def make_experiment(cls, backend, roster, params):
    """An experiment wired the way the Session wires one: the backend for
    acquisition, the recording device for state."""
    exp = cls(backend, params)
    exp.device = recording_device(backend, roster)
    return exp


def compile_probe(backend, exp):
    """Compile a probe's Schedule exactly as ``HardwareAgent.run`` would.

    Building a Schedule proves almost nothing: the 1 ns time grid and the latched
    parameter alignment are enforced inside the COMPILER, so a probe that only
    ever gets built fails for the first time on the instrument. Every probe test
    goes through here.

    The compiler does NOT enforce the DAC range for a swept amplitude domain —
    measured 2026-07-30, do not assume otherwise. A +/-0.9 domain (= +/-2.25 V on
    a QCM) compiles clean, and +/-3.0 dies with an internal numpy
    ``ufunc 'absolute' ... StrDType`` that names no port and no voltage. The real
    range guard is ``scqo_qblox/experiments/_flux_limits.py``, which runs BEFORE this.
    """
    from qblox_scheduler.backends.graph_compilation import SerialCompiler

    exp.sweep_axes = exp.define_sweep()
    schedule = exp.probe()
    qd = backend._hw_agent.quantum_device
    qd.hardware_config = backend._hw_agent.hardware_configuration
    return SerialCompiler().compile(schedule=schedule,
                                    config=qd.generate_compilation_config())
