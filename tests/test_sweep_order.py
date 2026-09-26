"""A descending window reaches the sequencer as a descending sweep.

scqo's flux, detuning and amplitude windows are a TRAVERSAL ORDER (decided
2026-09-26): ``start`` -> ``end`` in either direction, never re-sorted. On this backend
every such probe builds its loop domain from the axis ENDPOINTS,
``linspace(axis[0], axis[-1], n)``, so the order survives only if nothing on the
way takes a min/max or re-centers — and the compiler must accept the negative
step. Checked on every Qblox carrier of those windows (derived from the
Parameters mixins, so a new carrier is covered automatically), at both levels:

* the BUILT schedule: each swept domain of the window's kind starts at the
  ``start`` edge and runs down;
* the COMPILED Q1ASM: a sweep register is initialized and then decremented
  (``sub``) — what the sequencer actually runs. An ascending sweep ``add``s.

The scqo side proves the stored axis keeps the order and that no estimator can
tell; this file proves the instrument really walked it.
"""

from __future__ import annotations

import re

import pytest

pytest.importorskip("qblox_scheduler")

from conftest import compile_probe, make_backend, make_experiment  # noqa: E402
from test_probe_surface import QBLOX_PROBES, _params  # noqa: E402

import scqo_qblox.experiments  # noqa: E402,F401  (import side effect: @register)
from scqo.experiments import get  # noqa: E402
from scqo.experiments._capabilities import (  # noqa: E402
    AmplitudeSweepParameters,
    DriveDetuningSweepParameters,
    FluxSweepParameters,
    ReadoutDetuningSweepParameters,
)
from scqo_qblox.backend.qblox_backend import SELF_ACQUIRING_ATTR  # noqa: E402

#: window mixin -> (start field, end field, the loop-domain dtype it sweeps)
WINDOWS = {
    FluxSweepParameters: ("start_flux_v", "end_flux_v", "amplitude"),
    DriveDetuningSweepParameters: (
        "start_drive_detuning_hz", "end_drive_detuning_hz", "frequency"),
    ReadoutDetuningSweepParameters: (
        "start_readout_detuning_hz", "end_readout_detuning_hz", "frequency"),
    AmplitudeSweepParameters: ("start_amp_factor", "end_amp_factor", "amplitude"),
}

CASES = [
    pytest.param(name, *fields, id=f"{name}-{fields[0]}")
    for name in QBLOX_PROBES
    for mixin, fields in WINDOWS.items()
    if issubclass(get(name).Parameters, mixin)
    and not getattr(get(name), SELF_ACQUIRING_ATTR, None)
]


def test_every_window_has_a_qblox_carrier():
    """Guard against the derivation silently finding nothing (flux, both
    detuning frames and amplitude all have Qblox probes)."""
    assert {c.values[1] for c in CASES} == {f[0] for f in WINDOWS.values()}


def _descending(cls, start, end):
    """The small compile-sized params, with this window walked high -> low."""
    params = _params(cls)
    low, high = sorted((getattr(params, start), getattr(params, end)))
    return params.model_copy(update={start: high, end: low})


def _domains(schedule, dtype):
    """Every loop domain of ``dtype`` in a BUILT schedule tree."""
    found, seen = [], set()

    def walk(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        for domain in (getattr(node, "domain", None) or {}).values():
            if str(getattr(domain.dtype, "value", domain.dtype)).lower() == dtype:
                found.append(domain)
        for child in (getattr(node, "operations", None) or {}).values():
            walk(child)
        if getattr(node, "body", None) is not None:
            walk(node.body)

    walk(schedule)
    return found


def _programs(compiled):
    for cluster in compiled.compiled_instructions.values():
        if not isinstance(cluster, dict):
            continue
        for module in cluster.values():
            for sequencer in (module.get("sequencers", {}) if isinstance(module, dict)
                              else {}).values():
                program = (getattr(sequencer, "sequence", None) or {}).get("program")
                if program:
                    yield program


def _counts_down(program) -> bool:
    """True when some SWEEP register is decremented: ``move <v>,Rk # Initialize
    sweep var`` followed by ``sub Rk,<step>,Rk`` (an ascending sweep ``add``s)."""
    swept = set(re.findall(r"move\s+-?\d+,(R\d+)\s*#\s*Initialize sweep var", program))
    return any(re.search(rf"^\s*sub\s+{reg},\d+,{reg}\b", program, re.MULTILINE)
               for reg in swept)


@pytest.mark.parametrize("name,start,end,dtype", CASES)
def test_a_descending_window_is_swept_descending(tmp_path, roster, name, start, end, dtype):
    cls = get(name)
    backend = make_backend(tmp_path, roster)
    params = _descending(cls, start, end)
    exp = make_experiment(cls, backend, roster, params)
    # the two-tone probes play the drive chain's residual, which the fixture
    # leaves unseeded (NaN); the core run() solves it before probing
    exp.device.channel("q1", "drive").drive_power_dbm = -33.0
    exp.device.channel("q1", "readout").readout_depletion_s = 1e-6

    exp.sweep_axes = exp.define_sweep()
    domains = _domains(exp.probe(), dtype)
    assert domains, f"{name}: no {dtype} loop in the built schedule"
    swept = [d for d in domains if d.num > 1]
    assert swept and all(d.start > d.stop for d in swept), (
        f"{name}: {start}={getattr(params, start)} -> {end}={getattr(params, end)} "
        f"reached the schedule as {[(d.start, d.stop) for d in swept]}")

    compiled = compile_probe(backend, exp)
    assert any(_counts_down(p) for p in _programs(compiled)), (
        f"{name}: no sweep register counts down in the compiled Q1ASM")


def test_an_uneven_benchmarking_list_is_refused_by_name(tmp_path, roster):
    """The loop domain is rebuilt from the list's ENDPOINTS, so an uneven list
    would play different amplitudes than the dataset labels - refused before any
    instrument time. An evenly spaced list is legal in either direction."""
    cls = get("qubit_deterministic_benchmarking")
    backend = make_backend(tmp_path, roster)

    uneven = make_experiment(cls, backend, roster,
                             _params(cls).model_copy(update={"amp_prefactors": [0.5, 0.3, 0.4]}))
    with pytest.raises(ValueError, match="not evenly spaced"):
        uneven.define_sweep()

    even = make_experiment(cls, backend, roster,
                           _params(cls).model_copy(update={"amp_prefactors": [0.5, 0.4, 0.3]}))
    assert list(even.define_sweep()["amp_prefactor"]) == [0.5, 0.4, 0.3]
