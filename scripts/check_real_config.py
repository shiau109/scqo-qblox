"""Self-test the scqo stack against a REAL Qblox device config — no hardware needed.

    python scripts/check_real_config.py D:\\qpu_data\\SQ_demo\\QBLOX_config
    python scripts/check_real_config.py <config_dir> --targets q1 q2

Loads your lab's dut/hw config (any ``dut_config*.json`` + ``hw_config*.json`` in the
given folder), then runs the full scqo pipeline with SIMULATED data over the REAL
device tree: read neutral fields -> run experiments -> fit -> write results back ->
save in vendor format -> reload and compare. Everything happens on a temporary copy;
your original files are never opened for writing.

Needs an environment with qblox_scheduler (this repo's own ``.venv``, or the lab's
``.venv-qblox``); scqo/scqat
are picked up from the sibling repos automatically if not installed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # scqo_qblox without pip install
try:
    import scqo  # noqa: F401
except ImportError:  # sibling-repo layout (D:/github)
    sys.path[:0] = [str(REPO.parent / "SCQO"), str(REPO.parent / "scqat")]


def _pick(source: Path, pattern: str) -> Path:
    hits = [p for p in sorted(source.glob(pattern)) if "viewable" not in p.name]
    if not hits:
        raise SystemExit(f"no {pattern} found in {source}")
    return hits[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_dir", help="folder holding dut_config*.json + hw_config*.json")
    parser.add_argument("--targets", nargs="+", help="components to exercise (default: elements named q*)")
    args = parser.parse_args()

    source = Path(args.config_dir)
    work = Path(tempfile.mkdtemp(prefix="scqo_qblox_selftest_"))
    shutil.copy(_pick(source, "dut_config*.json"), work / "dut_config.json")
    shutil.copy(_pick(source, "hw_config*.json"), work / "hw_config.json")
    print(f"sandbox: {work}")
    print("  (temporary self-test copies + throwaway run data: your originals and your")
    print("   real data_root are NOT touched; real measurements use `scqo run`)")

    try:
        import scqo_qblox.elements  # noqa: F401  register the lab's custom element types
        from qblox_scheduler import QuantumDevice
    except ModuleNotFoundError as err:
        raise SystemExit(
            f"missing package: {err.name}\n"
            "This self-test needs the Qblox stack. Either install it into this environment:\n"
            "    uv pip install --python <your-venv-python> -e .\n"
            "or activate an environment that already has it (the lab uses .venv-qblox)."
        )

    qd = QuantumDevice.from_json_file(str(work / "dut_config.json"))
    print(f"[1/6] loaded device '{qd.name}' | elements: {list(qd.elements)}")

    qubits = args.targets or [n for n in qd.elements if n.startswith("q")]
    if not qubits:
        raise SystemExit(f"no q* elements in this config ({list(qd.elements)}) "
                         f"— name the ones to exercise with --targets")

    from scqo_qblox.backend.qblox_backend import QbloxDeviceModel
    from scqo.testing import demo_components

    # The driver resolves every name through the ROSTER (fl.q1 -> the readout
    # knobs of vendor element q1), so it is needed before the device model.
    # This throwaway self-test derives a chipT-shaped roster for the discovered
    # elements — one multiplexed feedline + a drive wire each; the REAL roster
    # lives in <data_root>/<device>/components.toml and is what `scqo run` uses.
    roster = demo_components(tuple(qubits), pair=False)

    dm = QbloxDeviceModel(qd, roster, config_file=str(work / "dut_config.json"))
    snap = dm.snapshot()
    for name, fields in snap.items():  # keyed by entity (fl.q1, xy_q1.q1)
        print(f"      {name}: {fields}")
    print(f"[2/6] snapshot OK | testing targets: {qubits}")

    import scqo_qblox.experiments  # noqa: F401
    from scqo import Session
    from scqo.testing import SimulatedBackend

    sess = Session(SimulatedBackend(dm), roster,
                   data_root=work / "data", device_name="selftest", state_sync="push")
    before = {name: dict(v) for name, v in sess.device_state().items()}
    failures = []
    for experiment in ("resonator_spectroscopy", "qubit_power_rabi"):
        # update="apply": this self-test exists to exercise writeback (scqo v0.6.0
        # defaults to suggest-only, which would leave the device tree untouched).
        result = sess.run(experiment, {"targets": qubits}, update="apply", tags=["selftest"])
        ok = all(result["outcomes"].get(q) == "successful" for q in qubits) and not result.get("error")
        print(f"[3/6] {experiment}: {result['outcomes']}" + (f" error={result['error']}" if result.get("error") else ""))
        if not ok:
            failures.append(experiment)

    # the knobs those two experiments write live on the CHANNEL entities:
    # readout_freq_hz on the readout channel, pi_amp on the drive channel - each
    # qubit's default channel of the kind, from the roster (never from strings)
    after = sess.device_state()
    ro = {q: roster.default_channel(q, "readout") for q in qubits}
    xy = {q: roster.default_channel(q, "drive") for q in qubits}
    touched = [ro[q] for q in qubits] + [xy[q] for q in qubits]
    moved = [name for name in touched if after[name] != before[name]]
    print(f"[4/6] writeback reached the real device tree for: {moved or 'NONE'}")
    if set(moved) != set(touched):
        failures.append("writeback")

    dm.save()
    reloaded = QuantumDevice.from_json_file(str(work / "dut_config.json"))
    dm2 = QbloxDeviceModel(reloaded, roster)
    round_trip = all(
        abs(dm2.snapshot()[ro[q]]["readout_freq_hz"]
            - after[ro[q]]["readout_freq_hz"]) < 1e-3 for q in qubits
    )
    print(f"[5/6] vendor-format save/reload round-trip: {'OK' if round_trip else 'MISMATCH'}")
    if not round_trip:
        failures.append("save-roundtrip")

    # OFFLINE COMPILE gate. Parse-grade validation above cannot catch configs that
    # VALIDATE but cannot COMPILE (e.g. explicit-null channel descriptions crash the
    # sequencer builder — chipA 2026-07-16): build the real resonator-spectroscopy
    # probe Schedule against the full backend (HardwareAgent + hw_config.json) and
    # compile it with no cluster attached — the same SerialCompiler call
    # HardwareAgent.run makes after connecting. The qblox mirror of the QM side's
    # offline generate_config() gate.
    from qcodes import Instrument

    Instrument.close_all()  # QuantumDevice is a qcodes singleton; free 'selftest' names
    try:
        from qblox_scheduler.backends.graph_compilation import SerialCompiler

        from scqo_qblox.backend.qblox_backend import QbloxBackend
        from scqo.device import RecordingDevice
        from scqo.experiments import get
        from scqo.stores import state_store

        backend = QbloxBackend(hardware_config=str(work / "hw_config.json"),
                               device_config=str(work / "dut_config.json"),
                               output_dir=str(work / "compile_out"),
                               roster=roster)
        exp_cls = get("resonator_spectroscopy")
        exp = exp_cls(backend, exp_cls.Parameters(targets=qubits, num_readout_freq_points=11, num_averages=2))
        # the probe reads its neutral state through the recording surface, the
        # way a Session hands it over (in-memory store: nothing persists here)
        exp.device = RecordingDevice(backend.device, roster,
                                     state_store(None, roster))
        exp.sweep_axes = exp.define_sweep()
        schedule = exp.probe()
        qd_real = backend._hw_agent.quantum_device
        qd_real.hardware_config = backend._hw_agent.hardware_configuration  # runtime truth
        SerialCompiler().compile(schedule=schedule, config=qd_real.generate_compilation_config())
        print("[6/6] offline schedule COMPILE against this config: OK")
    except Exception as err:  # noqa: BLE001 — a self-test reports, never crashes
        print(f"[6/6] offline schedule COMPILE FAILED: {type(err).__name__}: {err}")
        failures.append("compile")

    print(f"\nruns saved + indexed under {work / 'data'}: "
          f"{[r['run_id'] for r in sess.find_runs(tag='selftest')]}")
    print("\nPASS - scqo works against this real config" if not failures
          else f"\nFAIL - problems in: {', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
