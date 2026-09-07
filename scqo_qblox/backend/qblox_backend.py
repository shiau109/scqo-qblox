"""Qblox backend: maps the scqo abstractions onto qblox_scheduler.

``qblox_scheduler`` is imported lazily (inside methods) so that ``import scqo_qblox`` works
without the Qblox stack installed, and so the simulated path never needs it.

Since the greenfield model a driver serves a view PER CHANNEL ENTITY, not per
qubit: the roster's ``q1_xy`` / ``q1_ro`` / ``q1_z`` each carry their own
function's knobs, and all three resolve onto the SAME qblox_scheduler
``DeviceElement`` (the channel's single target). ``QbloxDeviceModel.component``
does that resolution through the roster — never by parsing names.

Neutral-name mapping: declared ONCE in ``scqo_qblox/backend/fieldmap.py`` (the catalog
``scqo state --fields`` renders; drift-tested per channel kind against scqo's knob
fields). The executable conversions are the three channel views below.
"""

from __future__ import annotations

import json
import math
import warnings
from contextlib import contextmanager
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import xarray as xr
from scqo.backend import Backend
from scqo.catalog import derived_op
from scqo.device import ComponentInfo, DeviceModel, EntityView, make_view_base
from scqo.entities import Channel
from scqo.fieldmap import OperatorCommand, Unrealized, VendorBinding, VendorOnly

from scqo_qblox.backend.fieldmap import (
    FIELD_BINDINGS,
    OPERATOR_COMMANDS,
    UNREALIZED,
    VENDOR_ONLY,
)

#: A probe whose ``probe()`` EXECUTES on the cluster and returns a ready Dataset
#: (instead of a native ``Schedule``) sets this class attribute to a one-line
#: WHY -- the truthy value is the flag, the text is the refusal reason. Same
#: spelling and same default-ALLOW polarity as scqo-qm's ``SELF_ACQUIRING_ATTR``
#: (the two backends are independent, so the constant is duplicated, not
#: imported). Census in tests/test_preview.py.
SELF_ACQUIRING_ATTR = "probe_self_acquires"


def _self_acquires(experiment: "Experiment") -> "str | None":
    """The declared reason this probe acquires inside ``probe()``, or None."""
    return getattr(type(experiment), SELF_ACQUIRING_ATTR, None)


@contextmanager
def _relaxed_grid_tolerance(tolerance_ns: float = 0.05):
    """Widen ``qblox_scheduler``'s grid-time tolerance FOR ONE COMPILE.

    The vendor default is 1.1e-3 ns, which rejects durations that float
    arithmetic lands a picosecond off the 1 ns grid -- the sub-band sweeps and
    the Clifford sequences both build their timings by accumulation, so they
    trip it. The constant is a module GLOBAL, so setting it and walking away
    would quietly relax validation for every later experiment in the process
    (and this repo has already lost a run to a schedule that compiled clean
    offline and died on the cluster). Restore it on the way out, always.
    """
    import qblox_scheduler.backends.qblox.constants as qblox_constants

    previous = qblox_constants.GRID_TIME_TOLERANCE_TIME
    qblox_constants.GRID_TIME_TOLERANCE_TIME = tolerance_ns
    try:
        yield
    finally:
        qblox_constants.GRID_TIME_TOLERANCE_TIME = previous


if TYPE_CHECKING:
    from scqo.experiment import Experiment
    from scqo.roster import Roster

#: Rendered-shot ceiling for ``preview`` (sweep points x schedule.repetitions).
#: The vendor pulse diagram UNROLLS every loop before sampling — profiled
#: 2026-08-09 on chipA: a 51-point thermal ramsey at 400 averages spent 9.4 min
#: inside sample_schedule, 84% of it in deepcopy under
#: _unroll_single_loop (103k operation clones, 61,201 sampled pulses), and
#: ``x_range`` does not bound that cost. Cost is linear in points x reps
#: (~30 rendered shots draw in about a second), so 256 keeps a preview around
#: ten seconds; anything bigger is refused by name with the exact remedy.
_PREVIEW_MAX_RENDERED_SHOTS = 256

#: timing_table.html row ceiling — the table unrolls with the same loops.
_PREVIEW_TIMING_ROWS = 2000

#: Nominal QRM-RF full-scale output (dBm) at pulse_amp=1.0, output_att=0 — the
#: datasheet maximum (+5 dBm into 50 Ohm). Frequency/LO/mixer-dependent in reality,
#: so absolute powers derived from it are good to ±a few dB; a per-setup
#: photon-number anchor (AC-Stark) is the Phase-3 refinement.
QBLOX_NOMINAL_FULL_SCALE_DBM = 5.0
#: The canonical digital operating point: keep the pulse amplitude <= 0.5 full
#: scale (shared by the readout AND drive chain solves).
_CANONICAL_MAX_AMP = 0.5
#: What to assume for a port whose module has never been asked. The maximum
#: output attenuation is NOT a constant of the product line — qblox_instruments
#: builds the ``out<k>_att`` validator from a live SCPI query
#: (``_get_max_out_att``), and chipA has a 60 dB QRM-RF (slot 8, ISA 2.0) beside
#: a 30 dB QCM-RF (slot 4, ISA 2.1). 60 keeps the full dynamic range on the
#: modules that have it; a port that turns out to be narrower is corrected the
#: first time a run connects (see ``QbloxBackend._sync_att_limits``) and
#: remembered in ``att_limits.json`` from then on.
_DEFAULT_MAX_OUTPUT_ATT = 60
#: An attenuation no Qblox RF output refuses, so one solved at or below it needs
#: no hardware question at all. The narrowest attenuator in the range is the
#: QRC's, whose ceiling moves with the centre frequency and bottoms out at
#: 20.5 dB (qblox_instruments' own out<k>_att docstring).
_UNIVERSALLY_SAFE_ATT = 20
#: Sidecar holding the discovered per-output maxima, beside mixer_cal.json.
ATT_LIMITS_FILE = "att_limits.json"
#: Deadline for ONE ``Cluster.close()`` in ``release_instruments``. Generous:
#: releasing is a courtesy, so a slow close should be waited out rather than
#: abandoned, and the caller is about to ask a human a question anyway. Past it
#: we warn and keep the socket, which is what happens today in any case.
_CLUSTER_CLOSE_TIMEOUT_S = 15.0
#: FLOOR for the run timeout, in SECONDS (the unit ``HardwareAgent.run`` takes;
#: its own default is 10 s). The actual deadline is computed per experiment by
#: ``_run_timeout_s`` and never goes below this.
#:
#: KEEP THIS A MULTIPLE OF 60. The seconds only look like seconds: the
#: instrument coordinator hands the cluster ``timeout_sec // 60`` MINUTES
#: (qblox_scheduler ``instrument_coordinator/components/qblox.py``), floor
#: division, so the real granularity is one minute and anything under 60 s
#: truncates to a ZERO-minute deadline. That is also why the error names
#: minutes while this constant is in seconds.
#:
#: Why 300: the parity monitors are parameterized by RECORD TIME and play one
#: long uninterrupted train of single shots, bounded by Qblox's 3e6 acquisition
#: bins — ~145 s of sequencer time at chipA's ``idle_multiple=3`` shot. The old
#: fixed 120 s sat inside that band and killed a 30 s-record run on 2026-08-01;
#: 300 s clears the bin-limited maximum about 2x over. A FIXED number was
#: defended on the grounds that the bin ceiling bounds every schedule — false
#: for FPGA-averaged sweeps, whose num_averages x sweep points x thermalization
#: is unbounded and killed long runs a second time (issue #24). Hence the
#: per-experiment scaling; this constant survives only as the proven-safe floor.
_RUN_TIMEOUT_S = 300


def _max_thermalization_s(experiment: "Experiment") -> float:
    """The longest per-shot thermal wait across the experiment's targets — 0.0
    for targets without a readable drive channel (pairs, unseeded knobs); the
    caller's sequence margin keeps the estimate nonzero regardless."""
    worst = 0.0
    for target in getattr(experiment.params, "targets", None) or []:
        try:
            chan = experiment.device.channel(target, "drive")
            worst = max(worst, float(chan.thermalization_time_s))
        except Exception:
            continue
    return worst


def _run_timeout_s(experiment: "Experiment") -> int:
    """Wall-clock deadline for ``HardwareAgent.run``, scaled to the experiment.

    A fixed number was outgrown twice (120 s by a 30 s parity record, 300 s by
    long averaged sweeps — issue #24), so the deadline is estimated from the
    experiment itself and floored at ``_RUN_TIMEOUT_S``:

    - the parity monitors publish their EXACT per-shot period during ``probe()``
      (``probe_shot_period_s``), so shots x the slowest period IS the run;
    - everything else is repetitions x sweep points x (thermalization + a 1 ms
      margin for the played sequence and ms-scale swept idles);
    - anything unestimable falls back to the floor.

    The estimate is doubled (arming, transfers, serialization are not in the
    product), then rounded UP to a whole minute — the vendor's ``// 60`` floor
    division would silently drop a fractional minute. Call it AFTER ``probe()``:
    the shot periods only exist once the probe has resolved them.
    """
    est = 0.0
    try:
        periods = getattr(experiment, "probe_shot_period_s", None) or {}
        if periods:
            shots = getattr(experiment, "resolved_num_shots", None)
            shots = shots() if callable(shots) else experiment.params.num_shots
            est = float(shots) * max(periods.values())
        else:
            params = experiment.params
            reps = (getattr(params, "num_averages", None)
                    or getattr(params, "num_shots", None))
            if reps:
                axes = getattr(experiment, "sweep_axes", None) or experiment.define_sweep()
                points = 1
                for name, values in axes.items():
                    # shot_idx repeats what `reps` already counts
                    if name != "shot_idx":
                        points *= max(1, len(values))
                est = float(reps) * points * (_max_thermalization_s(experiment) + 1e-3)
    except Exception:
        est = 0.0
    seconds = max(float(_RUN_TIMEOUT_S), 2.0 * est)
    return int(math.ceil(seconds / 60.0)) * 60


def chunk_timeout_s(experiment: "Experiment", *, shots: int, points: int) -> int:
    """Deadline for ONE sub-run of a probe that acquires inside ``probe()``.

    :func:`_run_timeout_s` estimates the WHOLE sweep from ``sweep_axes``; a
    self-acquiring probe launches that sweep in pieces and needs the same
    arithmetic applied to the piece it is about to run. Same shape, same
    doubling, same floor and — the part every hand-rolled formula in these
    probes got wrong — the same REAL thermalization time instead of a magic
    per-shot period. A 300 us reset x 2000 shots is ten minutes on its own,
    which is precisely the underestimate issue #24 reported.

    Public (no leading underscore) because the probes are its callers.
    """
    est = float(shots) * max(1, int(points)) * (_max_thermalization_s(experiment) + 1e-3)
    seconds = max(float(_RUN_TIMEOUT_S), 2.0 * est)
    return int(math.ceil(seconds / 60.0)) * 60


def _solve_att(name: str, target: float, what: str,
               max_att: int = _DEFAULT_MAX_OUTPUT_ATT) -> tuple[int, float]:
    """Solve an output chain for an absolute port power: the largest EVEN
    attenuation in ``[0, max_att]`` keeping the amplitude <= 0.5 (the module
    validator is Multiples(2) — an odd value would only fail later, at instrument
    prepare); the amplitude absorbs the exact residual.

    ``max_att`` is a property of the individual module output, not of the driver
    (see ``_DEFAULT_MAX_OUTPUT_ATT``). Lowering it never changes the power that
    reaches the port — the amplitude takes over what the attenuator cannot — it
    only costs DAC resolution, which is why clamping is the right response to a
    narrower attenuator and refusing would not be.
    """
    if target > QBLOX_NOMINAL_FULL_SCALE_DBM:
        raise ValueError(
            f"{name}: target {target} dBm exceeds the chain maximum "
            f"(+{QBLOX_NOMINAL_FULL_SCALE_DBM} dBm at amplitude 1, output_att=0)"
        )
    att_max = QBLOX_NOMINAL_FULL_SCALE_DBM - target + 20.0 * math.log10(_CANONICAL_MAX_AMP)
    att = int(min(int(max_att), max(0, 2 * math.floor(att_max / 2.0))))
    amp = 10.0 ** ((target - QBLOX_NOMINAL_FULL_SCALE_DBM + att) / 20.0)
    if amp > _CANONICAL_MAX_AMP:  # only when att=0 cannot absorb it (target > ~-1 dBm)
        warnings.warn(
            f"{name}: hitting {target} dBm needs {what}={amp:.3f} > "
            f"{_CANONICAL_MAX_AMP} (output_att already 0) — above the canonical "
            f"operating point"
        )
    return att, amp


def _port_outputs(hw_config: Any) -> dict[str, tuple[str, int, int]]:
    """``port -> (cluster, slot, complex-output index)`` from the connectivity graph.

    The physical identity of an output is what an attenuation limit belongs to,
    and the graph is the only place the config states it. (``scripts/
    calibrate_mixers.py`` parses the same edges for the same reason; it works off
    the raw JSON, this off the validated model, and neither may import the other
    — a backend must not depend on an operations script.)
    """
    # The validated model carries a networkx Graph, whose plain iteration yields
    # NODES; only `.edges` gives the pairs. A raw dict-loaded config is already a
    # list of pairs, so accept both rather than depending on which one a caller has.
    graph = getattr(getattr(hw_config, "connectivity", None), "graph", None)
    edges = getattr(graph, "edges", None) if graph is not None else None
    if edges is None:
        edges = graph or []
    found: dict[str, tuple[str, int, int]] = {}
    for edge in edges:
        try:
            a, b = edge
        except (TypeError, ValueError):
            continue
        for node, other in ((a, b), (b, a)):
            parts = str(node).split(".")
            if len(parts) != 3 or not parts[1].startswith("module"):
                continue
            cluster, module, channel = parts
            if not channel.startswith("complex_output_"):
                continue
            try:
                slot = int(module[len("module"):])
                output = int(channel.rsplit("_", 1)[1])
            except ValueError:
                continue
            found[str(other)] = (cluster, slot, output)
    return found


def _module_max_att(module: Any, output: int) -> int | None:
    """One module output's maximum attenuation in dB, or None if unknowable.

    Two shapes: QCM/QRM build a ``Multiples(2, 0, max)`` validator from the live
    query at construction, so the ceiling is readable off the parameter; QRC
    validates only the 0.5 dB step and exposes ``get_max_out<k>_att()`` because
    its ceiling moves with the centre frequency. Floored to an even integer
    either way — this driver's solve emits even dB.
    """
    getter = getattr(module, f"get_max_out{output}_att", None)
    if callable(getter):
        try:
            return int(2 * math.floor(float(getter()) / 2.0))
        except Exception:  # noqa: BLE001 - an unreachable module is "unknowable"
            return None
    param = getattr(module, f"out{output}_att", None)
    ceiling = getattr(getattr(param, "vals", None), "max_value", None)
    if ceiling is None:
        return None
    return int(2 * math.floor(float(ceiling) / 2.0))


def _att_limits(hw_agent: Any) -> dict[str, int]:
    """This driver's per-port-clock max-attenuation cache, carried on the agent.

    An annotation on the vendor object rather than driver state, because the
    channel VIEWS need it at solve time and the only thing they hold is the
    agent. Keyed by port-clock (what the solve has in hand); the sidecar on disk
    is keyed physically, by slot/output.
    """
    limits = getattr(hw_agent, "_scqo_qblox_att_limits", None)
    if limits is None:
        limits = {}
        hw_agent._scqo_qblox_att_limits = limits
    return limits


def _read(owner: Any, name: str) -> float:
    """Read a scheduler parameter across API generations: the legacy QCoDeS style
    exposes a callable (``param()``), the pydantic-model style a plain attribute."""
    attr = getattr(owner, name)
    return float(attr() if callable(attr) else attr)


def _write(owner: Any, name: str, value: float) -> None:
    attr = getattr(owner, name)
    if callable(attr):
        attr(float(value))
    else:
        setattr(owner, name, float(value))


def _grid_s(value: float, what: str) -> float:
    """Validate a seconds duration as a positive multiple of 4 ns and return it
    re-derived from the integer grid (so both backends store the identical
    canonical float). 4 ns is the PORTABLE neutral contract — QM's
    pulse/integration-weights resolution; the scheduler itself is finer, but an
    off-grid value here would be unrealizable on the QM backend."""
    ns = float(value) * 1e9
    grid = round(ns)
    if abs(ns - grid) > 1e-3 or grid <= 0 or grid % 4:
        raise ValueError(
            f"{what}={value!r} s: must be a positive multiple of 4 ns (the "
            f"portable pulse/weights grid; no silent rounding)")
    # / 1e9 (exact), never * 1e-9: division rounds correctly, so the stored float
    # equals the parsed literal (2000 -> exactly 2e-6) on BOTH backends.
    return grid / 1e9


def snap_ns(value: float, what: str, *, grid_ns: int = 1) -> float:
    """A positive seconds duration ROUNDED onto the scheduler's time grid.

    The rounding sibling of :func:`_grid_s`. Which one a field gets is a policy
    choice, already settled per field and mirrored on QM:

    * REFUSE (``_grid_s``) for values a HUMAN sets and a calibration depends on
      — an off-grid readout pulse would silently de-calibrate the stored
      amplitude, so it must be rejected loudly.
    * ROUND (here) for POLICY waits an automated fit produces. A T1-derived
      reset (``thermalization_factor x t1_s``) is never on the grid by luck, and
      a sub-nanosecond difference in a ~1 ms wait changes nothing measurable;
      refusing would make the calibration loop unable to write its own result.
      scqo-qm's ``set_thermalization_time`` already decided this.

    ``qblox_scheduler`` refuses anything off its 1 ns ``GRID_TIME`` (tolerance
    1.1e-3 ns) at COMPILE time, with a raw vendor traceback pointing at an
    absolute timestamp rather than at the field that caused it — so snapping
    here is also what keeps the error surface honest.
    """
    ns = float(value) * 1e9
    if not (ns > 0):
        raise ValueError(f"{what}={value!r} s: must be positive")
    snapped = max(grid_ns, round(ns / grid_ns) * grid_ns)
    # / 1e9, never * 1e-9 — same exact-float reason as _grid_s.
    return snapped / 1e9


class _QbloxChannelView:
    """Shared plumbing of the three channel views.

    ``name`` is the ROSTER ENTITY name (``q1_ro``) — what scqo addresses and what
    every error message must cite; the VENDOR element name (``q1``) is
    ``_element.name`` and is what the port-clock keys are built from. The two were
    the same string in the pre-greenfield model, which is why the split matters.

    ``hw_agent`` (the backend's HardwareAgent) is needed only by the absolute-power
    knobs: an output attenuation lives in the hardware compilation config, one level
    above the element. Its ``hardware_configuration`` dict is the AUTHORITATIVE
    runtime surface — every ``run()`` recompiles from it and re-pushes
    ``out<n>_att`` to the module, so writes there survive; a direct qcodes ``.set()``
    would be overwritten.
    """

    def __init__(self, name: str, element: Any, hw_agent: Any = None) -> None:
        self.name = name
        self._element = element
        self._hw_agent = hw_agent

    @property
    def _qubit(self) -> str:
        """The vendor element name behind this channel (the port-clock key half)."""
        return self._element.name

    def _output_att(self, port_clock: str) -> int:
        """Current output attenuation (dB) of a line from the hardware config
        (missing key -> 0 dB, the instrument default)."""
        if self._hw_agent is None:
            raise RuntimeError(
                f"{self.name}: no HardwareAgent attached — output_att (and so the "
                f"absolute powers) needs the hardware compilation config"
            )
        opts = self._hw_agent.hardware_configuration.hardware_options
        att_map = getattr(opts, "output_att", None) or {}
        return int(att_map.get(port_clock, 0))

    def _write_output_att(self, port_clock: str, att: int) -> None:
        if self._hw_agent is None:
            raise RuntimeError(
                f"{self.name}: no HardwareAgent attached — cannot write output_att"
            )
        opts = self._hw_agent.hardware_configuration.hardware_options
        if opts.output_att is None:
            opts.output_att = {}
        opts.output_att[port_clock] = att  # authoritative: recompiled+pushed each run

    def _max_output_att(self, port_clock: str) -> int:
        """This line's attenuator ceiling in dB — a property of the physical
        module output, learned from the instrument and cached (see
        ``_DEFAULT_MAX_OUTPUT_ATT``). Solving against the default on a narrower
        port is not silently wrong: the run that follows corrects it before
        anything is pushed."""
        if self._hw_agent is None:
            return _DEFAULT_MAX_OUTPUT_ATT
        return int(_att_limits(self._hw_agent).get(port_clock, _DEFAULT_MAX_OUTPUT_ATT))


class QbloxReadoutChannel(_QbloxChannelView, make_view_base("readout")):
    """The scqo READOUT channel view (``q1_ro``) over the target's ``DeviceElement``.

    Carries the dispersive-readout knobs: tone frequency, pulse amplitude/power,
    the pulse/window pair, and the discriminator (rotation + threshold, realized
    on acq_rotation/acq_threshold — only the RUS loop exit is Unrealized, see
    fieldmap.UNREALIZED). The monitors (fidelity_g/e, blob positions) are never
    pushed and so have no vendor surface at all.
    """

    @property
    def readout_freq_hz(self) -> float:
        return _read(self._element.clock_freqs, "readout")

    @readout_freq_hz.setter
    def readout_freq_hz(self, value: float) -> None:
        _write(self._element.clock_freqs, "readout", value)

    @property
    def readout_amp(self) -> float:
        return _read(self._element.measure, "pulse_amp")

    @readout_amp.setter
    def readout_amp(self, value: float) -> None:
        _write(self._element.measure, "pulse_amp", value)

    # ------------------------------------------- readout duration / window
    @property
    def readout_duration_s(self) -> float:
        return _read(self._element.measure, "pulse_duration")

    @readout_duration_s.setter
    def readout_duration_s(self, value: float) -> None:
        new_s = _grid_s(value, "readout_duration_s")
        _write(self._element.measure, "pulse_duration", new_s)
        # Portable contract (QM parity — its weights cannot span past the pulse):
        # the window never outlives the pulse, so a shrink clamps it down; the
        # scqo layer re-reads it and records the echo as a COUPLED change.
        if _read(self._element.measure, "integration_time") > new_s:
            _write(self._element.measure, "integration_time", new_s)

    @property
    def readout_integration_s(self) -> float:
        return _read(self._element.measure, "integration_time")

    @readout_integration_s.setter
    def readout_integration_s(self, value: float) -> None:
        new_s = _grid_s(value, "readout_integration_s")
        duration = _read(self._element.measure, "pulse_duration")
        if new_s > duration + 1e-12:
            raise ValueError(
                f"{self.name}: readout_integration_s={value!r} s exceeds the "
                f"readout pulse ({duration} s). The hardware here would allow "
                f"it, but QM cannot integrate past the pulse, so the portable "
                f"contract is window <= duration - raise readout_duration_s "
                f"first (or set both in one command; the pulse pushes first)")
        _write(self._element.measure, "integration_time", new_s)

    # ------------------------------------------------------------ absolute power
    def _port_clock(self) -> str:
        """The hardware-options key of this element's readout line, e.g. 'q1:res-q1.ro'."""
        port = getattr(self._element.ports, "readout")
        port = port() if callable(port) else port
        return f"{port}-{self._qubit}.ro"

    @property
    def readout_power_dbm(self) -> float:
        amp = _read(self._element.measure, "pulse_amp")
        if amp <= 0:  # log10 domain — the absolute power is undefined, not zero
            raise ValueError(
                f"{self.name}: readout pulse_amp is {amp} — absolute power undefined"
            )
        return (QBLOX_NOMINAL_FULL_SCALE_DBM - self._output_att(self._port_clock())
                + 20.0 * math.log10(amp))

    @readout_power_dbm.setter
    def readout_power_dbm(self, value: float) -> None:
        port_clock = self._port_clock()
        att, amp = _solve_att(self.name, float(value), "pulse_amp",
                              self._max_output_att(port_clock))
        self._write_output_att(port_clock, att)
        _write(self._element.measure, "pulse_amp", amp)

    # The photon-depletion wait, on the lab element's own `depletion` submodule
    # (elements.py) — the scheduler has no slot for it, unlike QM's
    # resonator.depletion_time. Absolute seconds both sides, so no conversion;
    # NaN = never calibrated, and it is left as NaN rather than coerced so the
    # consumers can tell "no wait" from "nobody measured this resonator".
    @property
    def readout_depletion_s(self) -> float:
        return _read(self._element.depletion, "duration")

    @readout_depletion_s.setter
    def readout_depletion_s(self, value: float) -> None:
        # ZERO IS LEGAL HERE, unlike every other wait on this backend: it means
        # "measured, and this resonator needs no settle", which is a different
        # claim from the NaN default ("nobody has measured it") and the two must
        # stay distinguishable — active reset runs on the first and refuses the
        # second. So snap_ns, which refuses non-positive durations, is only
        # reached for a real wait.
        if float(value) == 0.0:
            _write(self._element.depletion, "duration", 0.0)
            return
        # ROUNDED like thermalization_time_s, and for the same reason: this is a
        # policy wait resonator_spectroscopy writes as a factor over 1/(2 pi
        # kappa), never on the grid by luck.
        _write(self._element.depletion, "duration",
               snap_ns(value, f"{self.name}: readout_depletion_s"))

    # ---------------------------------------------------- readout discriminator
    # The two knobs `use_state_discrimination` runs on. The compiler reads them off
    # the element (they are `measure` factory kwargs), so a probe only has to ask
    # for acq_protocol="ThresholdedAcquisition" — see experiments/_state.py.
    #
    # The vendor knob is DEGREES and OPPOSITE-HANDED; the neutral field is RADIANS
    # in QM's integration_weights_angle convention (the catalog names the field
    # readout_rotation_rad). Qblox's acq_rotation "rotates the threshold line
    # clockwise" (qblox_scheduler acquisitions tutorial), i.e. the sequencer's
    # decision value is Re(z * e^{+i*theta}) — the DATA turned counterclockwise —
    # while a QM weights angle turns the data clockwise. So this boundary NEGATES
    # as well as converts; both together are what keep
    # `scqo set q1_ro.readout_rotation_rad=...` meaning the same rotation on both
    # backends. Missing the negation is not a fit-quality nit: chipA 2026-08-08
    # calibrated at 2.15 rad and the mirrored rotation put BOTH blob centres below
    # acq_threshold, so a discriminated qubit_relaxation read ~0 population at
    # every wait (run 20260808-123326-971). The threshold needs no conversion —
    # it lives in the rotated frame, which both conventions agree on — see fieldmap.
    #: decimals the radian read-back is quantized to. `radians(degrees(x))` is not
    #: exactly `x`, and RecordingDevice._sync_coupled compares vendor read-back to
    #: its cached config with EXACT float equality (deliberately — that strictness
    #: is the staleness guard's rule). Without this, every write to any OTHER
    #: readout knob emitted a junk `coupled_to` history row differing in the 17th
    #: digit. 1e-12 rad is ~6e-11 deg, twelve orders below anything the vendor
    #: resolves; the point is only that read(write(x)) == x.
    _ROTATION_DECIMALS = 12

    # A rotation is periodic, and the two sides pick DIFFERENT representatives of
    # the same angle: the sequencer takes degrees in [0, 360] and refuses anything
    # outside (constants.MIN/MAX_PHASE_ROTATION_ACQ — chipA 2026-07-26 died at
    # compile time on "-21.712643880286674 ... requires it to be between 0 and
    # 360"), while the neutral field keeps the mathematical convention (-pi, pi] so
    # a small correction reads as -0.39 rather than 5.89. Both directions fold, so
    # any real the caller writes is legal and reads back as its equivalent angle.
    @property
    def readout_rotation_rad(self) -> float:
        rad = -math.radians(float(_read(self._element.measure, "acq_rotation")) % 360.0)
        if rad <= -math.pi:
            rad += 2.0 * math.pi
        return round(rad, self._ROTATION_DECIMALS)

    @readout_rotation_rad.setter
    def readout_rotation_rad(self, value: float) -> None:
        deg = -math.degrees(round(float(value), self._ROTATION_DECIMALS)) % 360.0
        _write(self._element.measure, "acq_rotation", deg)

    @property
    def readout_threshold(self) -> float:
        return float(_read(self._element.measure, "acq_threshold"))

    @readout_threshold.setter
    def readout_threshold(self, value: float) -> None:
        _write(self._element.measure, "acq_threshold", float(value))

    # ------------------------------------------------------- unrealized knobs
    # A repeat-until-success LOOP EXIT threshold, which Qblox active reset has no
    # use for: it plays a fixed number of ConditionalResets rather than looping
    # (experiments/_reset.py), so acq_threshold alone decides every branch. There
    # is no second threshold to write, so this one stays Unrealized
    # (fieldmap.UNREALIZED) and needs a concrete raising pair — make_view_base
    # declares an abstract property per knob of the kind.
    _RUS_UNREALIZED = (
        "readout_rus_threshold is Unrealized on the Qblox backend: it is a "
        "repeat-until-success loop-exit threshold, and active reset here is not "
        "a loop — it plays a fixed number of ConditionalResets, each branching "
        "on acq_threshold. Feedback itself IS supported (conditional playback); "
        "the rotation and threshold are realized (acq_rotation/acq_threshold)"
    )

    @property
    def readout_rus_threshold(self) -> float:
        raise NotImplementedError(f"{self.name}: {self._RUS_UNREALIZED}")

    @readout_rus_threshold.setter
    def readout_rus_threshold(self, value: float) -> None:
        raise NotImplementedError(f"{self.name}: {self._RUS_UNREALIZED}")


class QbloxDriveChannel(_QbloxChannelView, make_view_base("drive")):
    """The scqo DRIVE channel view (``q1_xy``) over the target's ``DeviceElement``.

    Carries the xy knobs: drive frequency, the calibrated pi pulse (amplitude and
    length), and the saturation-drive pair behind the absolute drive power.
    """

    @property
    def drive_freq_hz(self) -> float:
        return _read(self._element.clock_freqs, "f01")

    @drive_freq_hz.setter
    def drive_freq_hz(self, value: float) -> None:
        _write(self._element.clock_freqs, "f01", value)

    @property
    def pi_amp(self) -> float:
        return _read(self._element.rxy, "amp180")

    @pi_amp.setter
    def pi_amp(self, value: float) -> None:
        _write(self._element.rxy, "amp180", value)

    @property
    def pi_duration_s(self) -> float:
        return _read(self._element.rxy, "duration")

    @pi_duration_s.setter
    def pi_duration_s(self, value: float) -> None:
        # No 4 ns grid guard: the portable pulse/weights grid is the READOUT
        # pulse/window contract (QM's integration weights); an x180 length is a
        # plain seconds value on both backends, and refusing off-grid ones here
        # would invent a contract the neutral catalog does not state.
        # The 1 ns SCHEDULER grid is not a contract we invent, though — it is
        # what this compiler enforces, and a hand-set 32.5 ns pi used to be
        # accepted here and then crash at compile time with a vendor traceback.
        # REFUSED rather than rounded (unlike the reset wait): a pi length is a
        # calibrated quantity whose stored pi_amp was measured against it.
        if abs(float(value) * 1e9 - round(float(value) * 1e9)) > 1e-3:
            raise ValueError(
                f"{self.name}: pi_duration_s={value!r} s is not a whole number "
                f"of nanoseconds (the qblox_scheduler 1 ns time grid). No "
                f"silent rounding: the stored pi_amp was calibrated against a "
                f"specific pulse length")
        _write(self._element.rxy, "duration", value)

    # Lives on the element's IdlingReset submodule, not on a drive parameter:
    # the wait is the qubit's own relaxation, and the scheduler's Reset() gate
    # compiles to IdlePulse(reset.duration). The drive channel is the neutral
    # home because scqo knobs live on channels, never on modes. Both sides are
    # absolute seconds, so this is a pass-through with no conversion — the one
    # backend where the neutral field maps 1:1.
    @property
    def thermalization_time_s(self) -> float:
        return _read(self._element.reset, "duration")

    @thermalization_time_s.setter
    def thermalization_time_s(self, value: float) -> None:
        # ROUNDED to the 1 ns scheduler grid, not refused: this is a policy wait
        # written by qubit_relaxation as thermalization_factor x t1_s, which is
        # never on the grid by luck. Writing it raw compiles into an off-grid
        # IdlePulse that poisons every later timestamp in the schedule.
        _write(self._element.reset, "duration",
               snap_ns(value, f"{self.name}: thermalization_time_s"))

    # ------------------------------------------------------------ absolute power
    # The drive twin, anchored to the stored SATURATION (spec) amplitude
    # (element.spec.spec_amp — the CW VoltageOffset the qubit_spectroscopy probe
    # plays). The drive output_att is PORT-level and shared by every xy pulse:
    # while it is off its standing value the stored pi_amp means a different
    # power (qubit_spectroscopy's run() sets and exactly reverts it; the even
    # discrete att + verbatim amplitude restore make the revert lossless).
    def _drive_port_clock(self) -> str:
        """The hardware-options key of this element's drive line, e.g. 'q1:mw-q1.01'."""
        port = getattr(self._element.ports, "microwave")
        port = port() if callable(port) else port
        return f"{port}-{self._qubit}.01"

    @property
    def drive_amp(self) -> float:
        amp = _read(self._element.spec, "spec_amp")
        if math.isnan(amp):
            raise ValueError(
                f"{self.name}: spec_amp is unset (NaN) — seed it (or set "
                f"drive_power_dbm, which writes it as the chain residual)"
            )
        return amp

    @drive_amp.setter
    def drive_amp(self, value: float) -> None:
        _write(self._element.spec, "spec_amp", value)

    @property
    def drive_power_dbm(self) -> float:
        amp = _read(self._element.spec, "spec_amp")
        # NaN would silently propagate through log10 into the config — refuse it
        # exactly like a zero/negative amplitude (power undefined, not a number).
        if not (amp > 0) or not math.isfinite(amp):
            raise ValueError(
                f"{self.name}: spec_amp is {amp} — absolute drive power undefined"
            )
        return (QBLOX_NOMINAL_FULL_SCALE_DBM - self._output_att(self._drive_port_clock())
                + 20.0 * math.log10(amp))

    @drive_power_dbm.setter
    def drive_power_dbm(self, value: float) -> None:
        port_clock = self._drive_port_clock()
        att, amp = _solve_att(self.name, float(value), "spec_amp",
                              self._max_output_att(port_clock))
        self._write_output_att(port_clock, att)
        _write(self._element.spec, "spec_amp", amp)

    # ------------------------------------------------------------ DRAG (pi)
    # rxy.beta is the derivative scaling in SECONDS; the neutral knob speaks ns
    # so it lands on the same numeric range as the shell's min_beta/max_beta
    # window (see fieldmap.FIELD_BINDINGS["drive"]["drag_beta"]).
    @property
    def drag_beta(self) -> float:
        val = _read(self._element.rxy, "beta")
        return float(val) * 1e9 if val is not None else 0.0

    @drag_beta.setter
    def drag_beta(self, value: float) -> None:
        _write(self._element.rxy, "beta", float(value) * 1e-9)

    # ------------------------------------------------------- unrealized knobs
    # The pi/2 pair. Qblox DERIVES X90 from rxy.amp180 (amp180*theta/180) and
    # rxy carries a SINGLE beta, so neither x90 knob has a home of its own that
    # anything PLAYS. Binding them to the x180 slots (amp180 = 2*pi_amp_x90,
    # the shared beta) would let an x90 calibration silently overwrite the
    # calibrated pi gate -- the failure issue #24 reported on the QM side.
    # Concrete raising pairs required because make_view_base declares the
    # abstract property for every knob. See fieldmap.UNREALIZED for the
    # promotion condition.
    _X90_UNREALIZED = (
        "the pi/2 gate has no independently calibrated knob on the Qblox backend: "
        "X90 is derived from rxy.amp180 and rxy has one shared beta, so writing "
        "either x90 knob would overwrite the x180 calibration. Set pi_amp / "
        "drag_beta instead"
    )

    @property
    def pi_amp_x90(self) -> float:
        raise NotImplementedError(f"{self.name}: {self._X90_UNREALIZED}")

    @pi_amp_x90.setter
    def pi_amp_x90(self, value: float) -> None:
        raise NotImplementedError(f"{self.name}: {self._X90_UNREALIZED}")

    @property
    def drag_beta_x90(self) -> float:
        raise NotImplementedError(f"{self.name}: {self._X90_UNREALIZED}")

    @drag_beta_x90.setter
    def drag_beta_x90(self, value: float) -> None:
        raise NotImplementedError(f"{self.name}: {self._X90_UNREALIZED}")


class QbloxFluxChannel(_QbloxChannelView, make_view_base("flux")):
    """The scqo FLUX channel view (``q1_z``) over the target's ``DeviceElement``.

    ``idle_flux`` is the standing bias every relative-frame flux sweep is measured
    FROM: the probes emit ``VoltageOffset(idle_flux)`` and play their excursion on
    top of it, exactly as QM's ``initialize_qpu`` + ``play("const")`` does.

    It was ``Unrealized`` here until 2026-07-30 while the probes read
    ``element.flux_params.sweet_spot`` through a vendor helper — i.e. the vendor
    home existed and the neutral surface was simply routed around it, which is
    the failure mode ``Unrealized`` is supposed to prevent (unsettable knob,
    invisible in pull mode, standing bias readable behind the API's back but
    never governed).

    An uncalibrated sweet spot reads as NaN and REFUSES rather than defaulting to
    0.0. Zero is precisely the value at which the absolute and relative frames
    coincide, so silently substituting it hides a wrong-frame sweep instead of
    surfacing it.

    The transfer-function FACTS (flux_offset, flux_per_phi0) are physical.json
    content and never push, so they bind nothing here.
    """

    @property
    def idle_flux(self) -> float:
        value = _read(self._element.flux_params, "sweet_spot")
        if not math.isfinite(value):
            raise ValueError(
                f"{self.name}: idle_flux (element.flux_params.sweet_spot) is not "
                f"calibrated. A relative-frame flux sweep is measured FROM this "
                f"bias, so running one without it would silently emit an "
                f"ABSOLUTE sweep. Calibrate the sweet spot, or set it to 0.0 to "
                f"declare the line genuinely parked at the DAC zero.")
        return value

    @idle_flux.setter
    def idle_flux(self, value: float) -> None:
        _write(self._element.flux_params, "sweet_spot", value)

    # ------------------------------------------------------- unrealized knobs
    # The flux line's output delay vs the drive line. The vendor home exists —
    # hardware_options.latency_corrections[<port-clock>] (seconds) — but no Qblox
    # experiment writes it yet (qubit_xyz_delay is QM-only), so it stays
    # Unrealized (fieldmap.UNREALIZED) with a concrete raising pair, since
    # make_view_base declares an abstract property per knob of the kind.
    _FLUX_DELAY_UNREALIZED = (
        "flux_delay_s is Unrealized on the Qblox backend: no XY-Z delay probe is "
        "wired here yet. The home exists — hardware_options.latency_corrections "
        "per port-clock (seconds) — and this promotes to a real binding when a "
        "Qblox qubit_xyz_delay probe lands"
    )

    @property
    def flux_delay_s(self) -> float:
        raise NotImplementedError(f"{self.name}: {self._FLUX_DELAY_UNREALIZED}")

    @flux_delay_s.setter
    def flux_delay_s(self, value: float) -> None:
        raise NotImplementedError(f"{self.name}: {self._FLUX_DELAY_UNREALIZED}")


#: channel kind -> the view class this backend serves for it. A kind absent here
#: (``pump``) and every non-channel entity (modes, lines, composites) is a KeyError
#: from ``component()`` — the contract scqo degrades gracefully against.
_CHANNEL_VIEWS: dict[str, type[_QbloxChannelView]] = {
    "drive": QbloxDriveChannel,
    "readout": QbloxReadoutChannel,
    "flux": QbloxFluxChannel,
}


def _read_or_none(view: EntityView, field: str) -> float | None:
    """Read a neutral knob, returning None if this element doesn't carry it.

    ValueError covers readout_power_dbm on a zero/unset pulse_amp; RuntimeError
    covers a device model constructed without a HardwareAgent (no hardware config
    to read the attenuation from) — and NotImplementedError, the Unrealized
    knobs' signal, is a RuntimeError subclass."""
    try:
        return getattr(view, field)
    except (TypeError, AttributeError, KeyError, ValueError, RuntimeError):
        return None


def _derived_operations(roster: "Roster", channel: Channel) -> tuple[str, ...]:
    """The single-mode operation this channel derives on its target — read from
    scqo's (channel kind x target kind) table, never declared here."""
    target = roster.entities.get(channel.target[0])
    if target is None:
        return ()
    try:
        op = derived_op(channel.kind, target.kind)
    except KeyError:
        return ()
    return (op,) if op else ()


class QbloxDeviceModel(DeviceModel):
    """Wraps a qblox_scheduler ``QuantumDevice`` (+ optionally its HardwareAgent).

    Entity names are ROSTER names: ``component("q1_ro")`` resolves the channel
    entity through the roster, takes its KIND and its single TARGET, fetches the
    vendor element for that target, and returns the matching channel view. The
    roster is therefore not optional — without it the driver cannot tell what
    ``q1_ro`` means.

    The agent reference (and its hardware config file path) enables the
    ``*_power_dbm`` surfaces — an output attenuation lives in the hardware
    compilation config, not on the element.
    """

    def __init__(self, quantum_device: Any, roster: "Roster",
                 config_file: str | None = None,
                 hw_agent: Any = None, hw_config_file: str | None = None) -> None:
        self._qd = quantum_device
        self._roster = roster
        self._config_file = config_file
        self._hw_agent = hw_agent
        self._hw_config_file = hw_config_file

    @property
    def roster(self) -> "Roster":
        """The device's authority on which entities exist (read-only)."""
        return self._roster

    def component(self, name: str) -> EntityView:
        """The view for one vendor-realized entity, addressed by ROSTER name.

        KeyError for everything this backend does not realize — an unknown name,
        a mode/line/composite (Qblox exposes no gate-macro surface), a pump or
        multi-target channel, and a channel whose target has no element in the
        dut config. That is the contract: scqo degrades gracefully and the doctor
        reports the gap against ``components()``.
        """
        e = self._roster.entities.get(name)
        if e is None:
            raise KeyError(
                f"unknown entity {name!r} — not in this device's roster")
        if not isinstance(e, Channel):
            channels = [c.name for c in self._roster.channels_of(name)]
            raise KeyError(
                f"{name!r} is a {type(e).__name__.lower()}; the Qblox backend "
                f"serves channel entities only (knobs live on channels) — "
                f"address {channels or '(none wired)'}")
        view_cls = _CHANNEL_VIEWS.get(e.kind)
        if view_cls is None:
            raise KeyError(
                f"{name!r} is a {e.kind} channel — the Qblox backend realizes "
                f"{sorted(_CHANNEL_VIEWS)} channels only")
        if len(e.target) != 1:
            raise KeyError(
                f"{name!r} is a multi-target {e.kind} channel {e.target} — the "
                f"Qblox backend serves one element per channel")
        try:
            element = self._qd.get_element(e.target[0])
        except KeyError:
            raise KeyError(
                f"{name!r} targets {e.target[0]!r}, which is not an element of "
                f"the loaded dut config (elements: {sorted(self._qd.elements)})"
            ) from None
        return view_cls(name, element, hw_agent=self._hw_agent)

    def components(self) -> dict[str, ComponentInfo]:
        """Derived inventory (the doctor's WITNESS, never truth): every roster
        channel this backend actually serves a view for, with the kind it serves
        it AS — ``vendor_checks`` FAILS on a kind disagreement, so this reports
        the channel's kind, not an element type (element_type cannot distinguish
        a coupler from a qubit; the roster arbitrates). Composites are absent:
        no Qblox gate-macro surface exists yet.
        """
        elements = set(self._qd.elements)
        out: dict[str, ComponentInfo] = {}
        for name, ch in self._roster.channels().items():
            if ch.kind not in _CHANNEL_VIEWS or len(ch.target) != 1:
                continue
            if ch.target[0] not in elements:
                continue
            out[name] = ComponentInfo(
                kind=ch.kind, target=tuple(ch.target), line=ch.line,
                operations=_derived_operations(self._roster, ch))
        return out

    def config_texts(self) -> dict[str, str]:
        """The two vendor config files as ``save()`` would write them, from memory:
        ``{"dut_config.json": text, "hw_config.json": text}`` — the agent's
        ``hardware_configuration`` is the RUNTIME truth, so it is what lands in
        hw_config.json AND replaces the copy embedded in the dut (``to_json`` would
        re-embed the stale copy WITH explicit nulls, the chipA 2026-07-16 crash
        trigger — see save()). Read-only: ``qd.hardware_config`` is NOT re-pointed
        here (save() does that), nothing is written, no cluster is touched. ``{}``
        without an agent (no hardware config to snapshot)."""
        hw = self._hw_agent.hardware_configuration if self._hw_agent is not None else None
        if hw is None:
            return {}
        # exclude_none is LOAD-BEARING: pydantic's default dump writes every unset
        # Optional as an explicit null, and the qblox compiler treats an
        # explicitly-null channel description as "user set None" (it lands in
        # model_fields_set on reload) and crashes sequencer compilation — whereas
        # an ABSENT key gets a default ChannelDescription. Never write nulls.
        hw_text = hw.model_dump_json(indent=2, exclude_none=True)
        data = json.loads(self._qd.to_json())
        if "hardware_config" in data:
            data["hardware_config"] = json.loads(hw.model_dump_json(exclude_none=True))
        return {"dut_config.json": json.dumps(data, indent=4) + "\n",
                "hw_config.json": hw_text + "\n"}

    def save(self) -> None:
        # Write back to the EXACT files the device was loaded from. (to_json_file
        # writes <device_name>.json, which silently diverges from the dut_config.json
        # the backend loads — calibrations would be stale after a restart.)
        hw = self._hw_agent.hardware_configuration if self._hw_agent is not None else None
        if hw is not None:
            # The separate hw_config.json is the RUNTIME truth (connect_clusters
            # overwrites qd.hardware_config from it on every run); keep the copy
            # embedded in dut_config.json in step BEFORE serializing the quantum
            # device, so the two files can never drift through a save.
            self._qd.hardware_config = hw
        texts = self.config_texts()  # the same two texts a setup snapshot stores
        if hw is not None and self._hw_config_file is not None:
            with open(self._hw_config_file, "w", encoding="utf-8") as f:
                f.write(texts["hw_config.json"])
        if self._config_file is not None:
            with open(self._config_file, "w", encoding="utf-8") as f:
                f.write(texts["dut_config.json"] if texts else self._qd.to_json())

    def snapshot(self) -> dict:
        """``{entity: {knob: value}}`` over the channel entities this backend
        realizes, reporting only the knobs the fieldmap declares BOUND (the
        Unrealized ones have no vendor value to seed from). None for a knob a
        given element cannot answer — a real lab device tree carries elements
        without a spec slot, and provenance must never crash a session."""
        state: dict[str, dict] = {}
        for name, ch in self._roster.channels().items():
            try:
                view = self.component(name)
            except KeyError:
                continue  # not realized here: absent from the snapshot entirely
            state[name] = {
                field: _read_or_none(view, field)
                for field in FIELD_BINDINGS.get(ch.kind, {})
            }
        return state


class QbloxBackend(Backend):
    """scqo Backend over a Qblox cluster (or dummy connections for dry runs)."""

    def __init__(self, hardware_config: str, device_config: str,
                 output_dir: str | None = None, *, roster: "Roster") -> None:
        # Lazy import keeps `import scqo_qblox` free of qblox_scheduler. The elements import
        # registers the lab's custom element types (FluxTunableTransmonElement) so the
        # dut config can be deserialized.
        import scqo_qblox.elements  # noqa: F401
        from qblox_scheduler import HardwareAgent

        self._roster = roster
        self._hw_agent = HardwareAgent(
            hardware_configuration=hardware_config,
            quantum_device_configuration=device_config,
            output_dir=output_dir,
        )
        # HardwareAgent parses a config PATH into a plain dict; the validated model
        # only appears once connect_clusters runs. Validate it NOW (the very call
        # connect_clusters makes) so the hardware config is readable/writable with
        # no cluster attached — readout_power_dbm and save() need it. A later
        # connect_clusters re-validates the same object harmlessly.
        import json as _json

        from qblox_scheduler.backends.qblox_backend import QbloxHardwareCompilationConfig

        validated = QbloxHardwareCompilationConfig.model_validate(
            self._hw_agent._hardware_configuration
        )
        # NORMALIZE through an exclude_none round-trip: a file carrying explicit
        # nulls (e.g. one written by a pre-fix save(), or the QBLOX_training
        # placeholder dumps) marks those fields as SET-to-None, and the compiler's
        # channel-description lookup then passes None into the sequencer config and
        # crashes. Dropping the nulls here takes them OUT of model_fields_set, so
        # poisoned configs load, compile, and self-heal on the next save().
        self._hw_agent._hardware_configuration = QbloxHardwareCompilationConfig.model_validate(
            _json.loads(validated.model_dump_json(exclude_none=True))
        )
        self._hw_config_file = hardware_config
        self._device = QbloxDeviceModel(
            self._hw_agent.quantum_device,
            roster,
            config_file=device_config,
            hw_agent=self._hw_agent,
            hw_config_file=hardware_config,
        )
        # Attenuator ceilings learned by an earlier run, so the very first solve
        # of this process is already right (see _sync_att_limits).
        self._load_att_limits()

    @classmethod
    def load(cls, config_dir: str = "./qblox_state", output_dir: str | None = None,
             *, roster: "Roster") -> "QbloxBackend":
        """Construct from the repo's standard config locations."""
        return cls(
            hardware_config=f"{config_dir}/hw_config.json",
            device_config=f"{config_dir}/dut_config.json",
            output_dir=output_dir,
            roster=roster,
        )

    @property
    def device(self) -> QbloxDeviceModel:
        return self._device

    @property
    def roster(self) -> "Roster":
        """The device roster this backend was built against."""
        return self._roster

    def field_bindings(self) -> dict[str, dict[str, VendorBinding]]:
        """The declared per-CHANNEL-KIND neutral-knob catalog (scqo_qblox.backend.fieldmap)
        — the conversion CODE is the channel views above; this is its description."""
        return {kind: dict(bindings) for kind, bindings in FIELD_BINDINGS.items()}

    def unrealized(self) -> dict[str, dict[str, Unrealized]]:
        """Knobs this backend cannot realize, per channel kind (see fieldmap)."""
        return {kind: dict(entries) for kind, entries in UNREALIZED.items()}

    def vendor_only(self) -> dict[str, VendorOnly]:
        """Qblox-unique calibration knobs, vendor-owned (see fieldmap)."""
        return dict(VENDOR_ONLY)

    def operator_commands(self) -> tuple[OperatorCommand, ...]:
        """This driver's vendor operator CLIs (see fieldmap) — the other half of
        "what can I reach on THIS instrument that is not a scqo command".

        The tuple is returned as-is, unlike ``vendor_only``'s defensive
        ``dict()``: a tuple of frozen dataclasses is already immutable, and
        keeping the same object keeps the tests' unbound equality check exact.
        """
        return OPERATOR_COMMANDS

    def _default_view(self, target: str, kind: str) -> EntityView | None:
        """The target's DEFAULT channel view of one kind, or None when the roster
        or the vendor tree cannot serve it (provenance must never fail a run)."""
        try:
            return self._device.component(
                self._roster.default_channel(target, kind))
        except Exception:
            return None

    def distortion_apply_command(self, target: str,
                                 run_id: str | None = None) -> str:
        """The command that turns a cryoscope's measured taps into THIS
        instrument's predistortion filter — scqo's duck-typed hint hook
        (``scqo.experiments._distortion_hint``), printed by both cryoscopes
        right after they propose ``distortion_amp``/``distortion_tau_s``.

        Run-addressed whenever the run is known: ``--run`` names exactly which
        measurement feeds the filter, so ``scqo accept`` order cannot change
        what lands in the hardware config (the two cryoscopes share ONE fact
        slot with REPLACE semantics — see
        :mod:`scqo_qblox.backend.apply_distortion`). Without a run id the
        command reads the accepted facts instead, the CLI's own default.
        """
        run = f" --run {run_id}" if run_id else ""
        return f"python -m scqo_qblox.backend.apply_distortion --target {target}{run}"

    def vendor_config_snapshot(self) -> dict[str, str]:
        """The vendor config as held in memory, as the two files ``save()`` would
        write (``QbloxDeviceModel.config_texts``): scqo's setup snapshot, stored per
        run under ``<device>/setup_snapshots/``. Provenance only: never writes,
        never touches the cluster, degrades to ``{}`` on any failure. Note that
        ``acquire()`` may legitimately change ``output_att`` between the run-start
        and run-end snapshots (``_sync_att_limits`` re-solves a chain under a
        newly learned module ceiling) — scqo reports that as ``setup_snapshot.drift``.
        """
        try:
            return self._device.config_texts()
        except Exception:  # provenance must never fail a run
            return {}

    def versions(self) -> dict[str, str]:
        """``{distribution: version}`` of this driver and its vendor libs, stamped
        into the setup snapshot's manifest so a later ``scqo restore`` can warn when
        the environment differs."""
        out: dict[str, str] = {}
        for name in ("scqo-qblox", "qblox-scheduler", "qblox-instruments"):
            try:
                out[name] = _dist_version(name)
            except Exception:  # not installed here
                continue
        return out

    def power_context(self, qubits: list[str]) -> dict:
        """Raw readout + drive chain values per qubit (run-record provenance only).

        Addressed by MODE name (``params.targets``): each target's default readout
        and drive channels are resolved through the roster, so the two blocks stay
        independent — an element without a spec slot still reports its readout chain.
        """
        out: dict = {}
        for name in qubits:
            out[name] = {}
            view = self._default_view(name, "readout")
            try:
                out[name] = {
                    "output_att_db": view._output_att(view._port_clock()),
                    "pulse_amp": _read(view._element.measure, "pulse_amp"),
                    "nominal_full_scale_dbm": QBLOX_NOMINAL_FULL_SCALE_DBM,
                    "readout_power_dbm": view.readout_power_dbm,
                    "note": "power derived from the nominal +5 dBm full scale "
                            "(frequency-dependent, ±a few dB)",
                }
                # The readout LO the data was taken at: a hand-edited lo_freq is
                # otherwise invisible in provenance (readout_freq_hz alone cannot
                # explain a jump across the IF window). Only when configured — a
                # missing entry must not degrade the rest of the context.
                opts = self._hw_agent.hardware_configuration.hardware_options
                mf = getattr(opts, "modulation_frequencies", None) or {}
                lo = getattr(mf.get(view._port_clock()), "lo_freq", None)
                if lo is not None:
                    out[name]["readout_lo_freq_hz"] = float(lo)
            except Exception:  # provenance must never fail a run
                out[name] = {}
            # The drive chain behind drive_power_dbm — same never-fail rule, and
            # independent of the readout block.
            try:
                view = self._default_view(name, "drive")
                out[name].update({
                    "drive_output_att_db": view._output_att(view._drive_port_clock()),
                    "spec_amp": _read(view._element.spec, "spec_amp"),
                    "drive_power_dbm": view.drive_power_dbm,
                })
                opts = self._hw_agent.hardware_configuration.hardware_options
                mf = getattr(opts, "modulation_frequencies", None) or {}
                lo = getattr(mf.get(view._drive_port_clock()), "lo_freq", None)
                if lo is not None:
                    out[name]["drive_lo_freq_hz"] = float(lo)
            except Exception:  # provenance must never fail a run
                pass
        return out

    def _drive_views(self, targets: list[str]) -> dict[str, Any]:
        """Every run target's DEFAULT drive view, keyed by channel name.

        A composite target (a pair) has no drive channel of its own, so it
        expands to its MEMBER modes' drive channels — the entities the reset
        actually happens on. Roster-resolved; never string arithmetic."""
        from scqo.entities import Composite

        views: dict[str, Any] = {}
        for target in targets:
            entity = self._roster.entities.get(target)
            modes = [target]
            if isinstance(entity, Composite):
                modes = [m for names in entity.roles.values() for m in names]
            for mode in modes:
                try:
                    name = self._roster.default_channel(mode, "drive")
                except Exception:
                    continue
                if name not in views:
                    views[name] = self.device.component(name)
        return views

    @contextmanager
    def _thermalization_override(self, experiment: "Experiment"):
        """Per-run override of the thermal-reset wait: recorded set -> run ->
        exact revert.

        The STANDING wait already lives in the dut config (scqo pushes the
        neutral ``thermalization_time_s`` knob onto ``element.reset.duration``),
        so ``Reset()`` needs no argument and no probe changes. Only an explicit
        per-run override acts — and unlike QM, where the wait is baked in at
        program-BUILD time, the scheduler expands ``Reset`` into
        ``IdlePulse(reset.duration)`` during COMPILATION, which happens inside
        ``run()``. So this must bracket the whole acquire, not just probe().

        Writes go straight to the vendor tree, NOT through the recording
        device: a per-run override must leave no ChangeRecord and no store row.

        THERMAL ONLY. ``ConditionalReset`` never reads ``element.reset.duration``,
        so under ``reset_method='active'`` this would set and revert a value
        nothing plays. ``experiments/_reset.py::check_reset_method`` refuses that
        combination upstream — same discipline as the RuntimeError below."""
        override_ns = getattr(experiment.params, "thermalization_time_ns", None)
        if override_ns is None:
            yield
            return
        views = self._drive_views(list(experiment.params.targets))
        if not views:
            raise RuntimeError(
                f"thermalization_time_ns={override_ns} was set but no drive "
                f"channel resolved for targets {list(experiment.params.targets)} "
                f"— refusing to run with the override silently ignored")
        previous = {n: v.thermalization_time_s for n, v in views.items()}
        for view in views.values():
            view.thermalization_time_s = float(override_ns) / 1e9
        try:
            yield
        finally:
            for name, view in views.items():
                view.thermalization_time_s = previous[name]

    # ------------------------------------------------- output-attenuator limits
    @property
    def _att_limits_path(self) -> Path | None:
        """The sidecar beside hw_config.json (mixer_cal.json's neighbour)."""
        if self._hw_config_file is None:
            return None
        return Path(self._hw_config_file).parent / ATT_LIMITS_FILE

    def _load_att_limits(self) -> None:
        """Seed the port-clock cache from the sidecar, if one was written before.

        Physical keys on disk (``slot4/out0``), port-clock keys in memory: the
        limit belongs to the module output, but the solve only ever has a
        port-clock, and rewiring must not carry a stale ceiling to a new port.
        """
        path = self._att_limits_path
        if path is None or not path.is_file():
            return
        import json as _json

        try:
            stored = _json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # a corrupt sidecar is a cache miss, never a failed run
        outputs = _port_outputs(self._hw_agent.hardware_configuration)
        limits = _att_limits(self._hw_agent)
        for port_clock in self._known_port_clocks():
            key = self._physical_key(outputs, port_clock)
            if key is not None and isinstance(stored.get(key), int):
                limits[port_clock] = int(stored[key])

    @staticmethod
    def _physical_key(outputs: dict, port_clock: str) -> str | None:
        entry = outputs.get(port_clock.split("-", 1)[0])
        return None if entry is None else f"slot{entry[1]}/out{entry[2]}"

    def _known_port_clocks(self) -> dict[str, Any]:
        """``port_clock -> the view that owns it`` for every output chain whose
        power this driver solves (readout tones and saturation drives)."""
        found: dict[str, Any] = {}
        for name in self._device.components():
            try:
                view = self._device.component(name)
            except KeyError:
                continue
            for getter in ("_port_clock", "_drive_port_clock"):
                resolve = getattr(view, getter, None)
                if resolve is None:
                    continue
                try:
                    found[resolve()] = view
                except Exception:  # noqa: BLE001 - an element without that port
                    continue
        return found

    def _suspect_chains(self) -> dict[str, Any]:
        """``port_clock -> view`` for every chain whose stored attenuation might
        be more than its output can actually do.

        A chain is suspect only when its ceiling is UNKNOWN and the solve went
        above ``_UNIVERSALLY_SAFE_ATT``. That keeps the hardware query rare and
        purposeful: a modest attenuation is legal on every Qblox RF output ever
        shipped, so asking about it would cost a cluster connection to learn
        nothing — and it is what keeps ``acquire()`` from dialling the cluster in
        offline tests that stub ``run()``.
        """
        limits = _att_limits(self._hw_agent)
        return {
            port_clock: view
            for port_clock, view in self._known_port_clocks().items()
            if self._output_att_of(port_clock)
            > limits.get(port_clock, _UNIVERSALLY_SAFE_ATT)
        }

    def _sync_att_limits(self) -> None:
        """Learn each suspect output's real attenuator ceiling, then re-solve any
        chain that was solved against a wider one.

        WHY THIS IS NOT IN THE SOLVE. ``out<k>_att``'s ceiling comes from a live
        SCPI query — qblox_instruments builds the validator from it — so it is
        unknowable until something connects, and nothing connects until a run.
        Solving optimistically and correcting here is what keeps the full 60 dB
        on the modules that have it while never pushing an illegal value to one
        that does not (chipA: 60 dB on slot 8, 30 dB on slot 4, both RF).

        The correction is POWER-PRESERVING and needs no target: writing a
        chain's current ``*_power_dbm`` back re-solves it under the new ceiling,
        and the attenuator's give is taken up by the amplitude. Runs before
        ``probe()``, because the probe plays ``drive_amp``.
        """
        import json as _json

        suspect = self._suspect_chains()
        if not suspect:
            return  # nothing on air could be refused; do not touch the cluster

        try:
            clusters = self._hw_agent.get_clusters()
        except Exception as err:  # noqa: BLE001 - offline/dummy: keep the default
            warnings.warn(
                f"could not read the output-attenuation limits of "
                f"{', '.join(sorted(suspect))} ({type(err).__name__}: {err}) — "
                f"leaving the solve at {_DEFAULT_MAX_OUTPUT_ATT} dB; a narrower "
                f"output will refuse the value at instrument prepare"
            )
            return

        outputs = _port_outputs(self._hw_agent.hardware_configuration)
        limits = _att_limits(self._hw_agent)
        discovered: dict[str, int] = {}
        for port_clock, view in suspect.items():
            entry = outputs.get(port_clock.split("-", 1)[0])
            if entry is None:
                continue
            cluster_name, slot, output = entry
            module = getattr(clusters.get(cluster_name), f"module{slot}", None)
            ceiling = None if module is None else _module_max_att(module, output)
            if ceiling is None:
                continue
            limits[port_clock] = ceiling
            discovered[f"slot{slot}/out{output}"] = ceiling
            solved = self._output_att_of(port_clock)
            if solved <= ceiling:
                continue
            field = ("readout_power_dbm" if port_clock.endswith(".ro")
                     else "drive_power_dbm")
            standing = self._device.component(view.name)  # raw view: no ChangeRecord
            power = getattr(standing, field)
            warnings.warn(
                f"{view.name}: {port_clock} was solved for {solved} dB but slot "
                f"{slot} output {output} attenuates at most {ceiling} dB — "
                f"re-solving at {power:.2f} dBm, which the amplitude now carries "
                f"(same power at the port, less DAC range)"
            )
            setattr(standing, field, power)

        path = self._att_limits_path
        if discovered and path is not None:
            try:
                merged = {}
                if path.is_file():
                    try:
                        merged = _json.loads(path.read_text(encoding="utf-8"))
                    except ValueError:
                        merged = {}
                merged.update(discovered)
                path.write_text(_json.dumps(merged, indent=2) + "\n", encoding="utf-8")
            except OSError:
                pass  # a read-only config folder costs a re-query, not a run

    def _output_att_of(self, port_clock: str) -> int:
        opts = self._hw_agent.hardware_configuration.hardware_options
        return int((getattr(opts, "output_att", None) or {}).get(port_clock, 0))

    def acquire(self, experiment: "Experiment") -> xr.Dataset:
        # The reset-method backstop. add_reset() already refuses inside every
        # probe, but that only fires if the probe remembers to call it; this
        # fires for anything carrying the neutral field, before we touch the
        # instrument. check_reset_method is side-effect free, so being called
        # twice costs nothing.
        from scqo_qblox.experiments._reset import check_reset_method

        check_reset_method(experiment)

        # Before probe(): the probe plays the amplitude this may re-solve.
        self._sync_att_limits()
        with self._thermalization_override(experiment), _relaxed_grid_tolerance():
            # A self-acquiring probe compiles and RUNS its own schedules inside
            # probe() (it has to: it steps LO/DRAG between acquisitions), so it
            # hands back the finished Dataset. The declared attribute decides,
            # not the return type -- preview() reads the same flag, and an
            # undeclared probe returning a Dataset is a bug worth surfacing.
            res = experiment.probe()
            reason = _self_acquires(experiment)
            if reason:
                if not isinstance(res, xr.Dataset):
                    raise TypeError(
                        f"{type(experiment).__name__} declares "
                        f"{SELF_ACQUIRING_ATTR} ({reason}) but probe() returned "
                        f"{type(res).__name__}, not a Dataset")
                return res
            schedule = res  # native qblox_scheduler.Schedule
            timeout_s = _run_timeout_s(experiment)  # after probe(): needs the shot periods
            # retrieve_acquisition() re-waits against the coordinator's OWN
            # timeout parameter (default 60 s) — keep that second gate no
            # tighter than the first.
            self._hw_agent.instrument_coordinator.timeout(timeout_s)
            try:
                raw = self._hw_agent.run(schedule, timeout=timeout_s)
            finally:
                # The clusters only exist once run() has connected them, and the loops
                # this quiets are the ones qblox_instruments leaves open until
                # interpreter shutdown — so install here, and in `finally` so a run
                # that raises still gets it. Idempotent; see _asyncio_noise.
                from scqo_qblox.backend._asyncio_noise import silence_proactor_self_pipe_noise

                silence_proactor_self_pipe_noise(self._hw_agent)
        return self._to_canonical(raw, experiment)

    def release_instruments(self) -> list[str]:
        """Hand the cluster back; the names released. The ``scqo.Backend`` hook.

        This backend is the reason the hook exists: ``HardwareAgent.run()``
        calls ``connect_clusters()`` and the vendor exposes NO disconnect, so a
        process that has acquired once holds four sockets per cluster until it
        exits. The CLI calls this before blocking on the accept prompt — an
        operator who walks away from an overnight campaign's "apply which
        updates?" would otherwise pin the cluster all night, and that
        contention degrades every later run (CLAUDE.md, the zombie-process
        paragraph). Safe there: a decision only writes the in-memory config and
        the JSON files, and the next ``acquire()`` reconnects from scratch.

        Reads ``_clusters`` DIRECTLY and returns ``[]`` when it is empty.
        ``get_clusters()`` would be the obvious accessor and is the wrong one:
        it calls ``connect_clusters()`` when nothing is connected, so asking to
        disconnect would connect.

        ``Cluster.close()`` can block forever (``scripts/calibrate_mixers.py``
        ``close_cluster`` says why, and why it cannot safely be moved
        off-thread), so each close runs on a DAEMON thread that is joined with a
        deadline. That bounds the caller: the worst case is a warning and a
        socket still held — exactly today's behaviour — never a hang. The
        script's ``_hard_exit`` answer is deliberately NOT reused: killing the
        process is legitimate at the end of a standalone tool, never inside a
        live session that still owes the operator a decision.
        """
        import threading

        components = dict(getattr(self._hw_agent, "_clusters", None) or {})
        if not components:
            return []  # nothing ever connected: never dial in order to hang up
        coordinator = self._hw_agent.instrument_coordinator
        released: list[str] = []
        for name, component in components.items():
            try:
                coordinator.remove_component(component.name)
            except Exception as err:  # already gone: closing is still worth trying
                warnings.warn(f"{name}: removing the instrument-coordinator component "
                              f"failed ({type(err).__name__}: {err})", stacklevel=2)
            closer = threading.Thread(target=self._close_cluster, args=(name, component),
                                      name=f"release-{name}", daemon=True)
            closer.start()
            closer.join(_CLUSTER_CLOSE_TIMEOUT_S)
            if closer.is_alive():
                warnings.warn(
                    f"{name}: Cluster.close() did not return within "
                    f"{_CLUSTER_CLOSE_TIMEOUT_S:g} s; its sockets stay open until this "
                    f"process exits", stacklevel=2)
                continue
            released.append(name)
        # Cleared even for a close that timed out: the components are no longer
        # registered, so a later run must rebuild them. connect_clusters()
        # closes a same-named stale ClusterComponent itself.
        self._hw_agent._clusters.clear()
        return released

    @staticmethod
    def _close_cluster(name: str, component: Any) -> None:
        """Close one ClusterComponent's underlying Cluster. Never raises: it runs
        on a daemon thread whose exception would only print a stray traceback."""
        try:
            component.cluster.close()
        except Exception as err:  # noqa: BLE001 - a failed close must not kill the CLI
            warnings.warn(f"{name}: Cluster.close() raised "
                          f"({type(err).__name__}: {err})", stacklevel=2)

    def preview(self, experiment: "Experiment", out_dir: Path,
                **options: Any) -> list[Path]:
        """Render the COMPILED schedule to files — no cluster, nothing saved.

        The scqo ``Backend.preview`` hook (``Session.preview`` /
        ``scqo run <name> --preview``): the same construction slice as
        ``acquire()`` minus the instrument — the reset backstop, the
        thermalization bracket around probe()+compile (the wait is baked in
        at build/compile time), and the exact ``hardware_config`` mutation
        ``HardwareAgent.run`` performs. It must NEVER call
        ``_sync_att_limits`` (a network call that can also write
        att_limits.json — preview stays fully offline). Both artifacts need
        the compiled timeline: ``plot_pulse_diagram``/``timing_table``
        require absolute timing, so compilation is mandatory, and a failure
        in either fails the preview — they ARE the product.

        An OVERSIZED schedule is refused by name BEFORE compiling: the vendor
        sampler draws every sweep point of EVERY repetition (and ``x_range``
        does not bound the cost), so the render is linear in
        points x repetitions and a lab-default sweep is hours — see
        ``_PREVIEW_MAX_RENDERED_SHOTS``. No gate-level ``circuit_diagram`` on
        purpose: the vendor's circuit renderer cannot draw a hardware-loop
        swept schedule (uncompiled it dies comparing a symbolic duration,
        compiled it dies in ``_draw_loop`` — measured 2026-08-09), and every
        registered probe sweeps.
        """
        from scqo.backend import PreviewWarning

        from scqo_qblox.experiments._reset import check_reset_method

        # Backend-specific preview options: `no_simulate` is a harmless
        # request here (this preview is offline by construction), everything
        # else is refused by name — never silently ignored.
        options.pop("no_simulate", None)
        if options:
            raise ValueError(
                f"the qblox backend has no simulator — preview option(s) "
                f"{', '.join(sorted(options))} are not realized here (the "
                f"compiled pulse diagram is already the full picture)")
        name = getattr(type(experiment), "name", type(experiment).__name__)
        reason = _self_acquires(experiment)
        if reason:
            raise ValueError(
                f"{name} cannot be previewed on the Qblox backend: {reason} — "
                f"it acquires inside probe(), so there is no single standalone "
                f"schedule to render; run it for real, or preview another "
                f"experiment")
        check_reset_method(experiment)
        # The rendered-shot guard runs BEFORE probe(): the averaging rides a
        # LoopOperation inside the schedule (verified by the 61,201-pulse
        # profile = points x averages x 3 exactly), so the schedule object
        # itself reports repetitions=1 and the honest count is the one scqo
        # already knows — the sweep grid times the shot count.
        n_points = max(1, math.prod(
            len(values) for values in experiment.sweep_axes.values()))
        reps = int(getattr(experiment.params, "num_averages", None)
                   or getattr(experiment.params, "num_shots", None) or 1)
        rendered = n_points * max(1, reps)
        if rendered > _PREVIEW_MAX_RENDERED_SHOTS:
            raise ValueError(
                f"{name}: preview refuses to render this schedule — the "
                f"vendor pulse diagram unrolls and draws every repetition, "
                f"and {n_points} sweep points x {reps} shots = {rendered} "
                f"rendered shots (cap {_PREVIEW_MAX_RENDERED_SHOTS}; ~20k "
                f"shots measured at 10+ minutes). The per-shot sequence "
                f"does not change with the counts — preview it small: "
                f"--set num_averages=2 plus this experiment's point-count "
                f"parameter (num_points / num_readout_freq_points / ... — see "
                f"`scqo run {name} --help`); the real run is unaffected")
        with self._thermalization_override(experiment), _relaxed_grid_tolerance():
            schedule = experiment.probe()  # native qblox_scheduler.Schedule
            from qblox_scheduler.backends.graph_compilation import SerialCompiler

            qd = self._hw_agent.quantum_device
            # the same mutation HardwareAgent.run performs before compiling
            qd.hardware_config = self._hw_agent.hardware_configuration
            compiled = SerialCompiler().compile(
                schedule=schedule, config=qd.generate_compilation_config())

        out_dir.mkdir(parents=True, exist_ok=True)
        pulse_html = out_dir / "pulse_diagram.html"
        pulse_fig = compiled.plot_pulse_diagram(plot_backend="plotly")
        pulse_fig.write_html(str(pulse_html))

        timing_html = out_dir / "timing_table.html"
        timing_df = compiled.timing_table.data  # plain DataFrame, no Styler
        body = timing_df.head(_PREVIEW_TIMING_ROWS).to_html()
        if len(timing_df) > _PREVIEW_TIMING_ROWS:
            body = (f"<p><b>truncated:</b> showing the first "
                    f"{_PREVIEW_TIMING_ROWS} of {len(timing_df)} rows</p>"
                    + body)
            warnings.warn(PreviewWarning(
                f"timing_table.html truncated to the first "
                f"{_PREVIEW_TIMING_ROWS} of {len(timing_df)} rows"))
        timing_html.write_text(body, encoding="utf-8")
        return [pulse_html, timing_html]

    @staticmethod
    def _to_canonical(raw: xr.Dataset, experiment: "Experiment") -> xr.Dataset:
        """Relabel a raw Qblox dataset into scqo's convention: dims (target, <sweep>),
        vars I/Q — or the discriminated variable when the run was thresholded.

        The probes label every acquisition ``acq_channel=f"S_21_{qubit}"`` with the
        sweep loop variable as a per-point coordinate (cal02 reference pattern), so
        the hardware returns one array per qubit over the swept axis (repetitions
        averaged on the cluster). The canonical sweep values come from
        ``experiment.sweep_axes`` — the probe built its loop from exactly those.
        On a structure mismatch the raw dataset is pickled for offline inspection.

        THRESHOLDED runs (``use_state_discrimination``, experiments/_state.py) come
        back on the SAME variable but REAL: the FPGA already compared each shot
        against acq_threshold. Which mode it was is read off the vendor's own
        ``acq_protocol`` attribute rather than the parameter — the attribute
        describes what the hardware actually did and cannot disagree with it.
        Getting this wrong is silent, not loud: taking ``.real``/``.imag`` of a
        thresholded result would put the population in ``I``, leave ``Q`` at zero,
        still satisfy the ``("I","Q")`` contract, and hand scqat a degenerate blob
        to reduce.

        The discriminated VARIABLE NAME follows the readout schema (SCQO TUTORIAL
        §11): with a ``shot_idx`` sweep the values are per-shot outcomes and stay
        ``state``; without one the cluster averaged the thresholded shots, so the
        values are probabilities and land on ``population``.
        """
        import numpy as np

        qubits = list(experiment.params.targets)  # type: ignore[attr-defined]
        axes = {name: np.asarray(values) for name, values in experiment.sweep_axes.items()}
        readout = getattr(experiment, "readout_coords", None)
        if readout is not None:
            axes.update({name: np.asarray(values) for name, values in readout().items()})
        shape = tuple(len(v) for v in axes.values())
        rows: list = []
        discriminated = False
        try:
            for name in qubits:
                key = f"S_21_{name}"
                if key not in raw.data_vars:
                    raise KeyError(
                        f"acquisition channel {key!r} not in raw dataset "
                        f"(data_vars={list(raw.data_vars)}) — probe/hardware mismatch"
                    )
                discriminated = raw[key].attrs.get("acq_protocol") == "ThresholdedAcquisition"
                values = np.asarray(raw[key].values).squeeze()
                if values.shape != shape:
                    if values.ndim == len(shape) and values.shape == tuple(reversed(shape)):
                        values = values.T  # labeled axes returned in reversed order
                    elif values.size == int(np.prod(shape)):
                        # flat bin order follows the probe's loop nesting (outer axis
                        # first), which matches sweep_axes insertion order
                        values = values.reshape(shape)
                    else:
                        raise ValueError(
                            f"{key}: expected shape {shape} for axes {list(axes)}, "
                            f"got {values.shape} (dims={dict(raw.sizes)})"
                        )
                rows.append(values)
        except (KeyError, ValueError) as err:
            raise type(err)(f"{err}; {_dump_raw(raw)}") from err
        dims = ("target", *axes.keys())
        coords = {"target": qubits, **axes}
        if discriminated:
            # The compiler substitutes -1 for a NaN threshold result
            # (compiler.py: `n if not isnan(n) else -1`). Left alone it reads as a
            # perfectly valid population 100% below the floor; as NaN the fitter
            # skips the point.
            state = np.stack(rows).real.astype(float)
            state[state == -1.0] = np.nan
            var = "state" if "shot_idx" in axes else "population"
            return xr.Dataset({var: (dims, state)}, coords=coords)
        stacked = np.stack(rows)
        return xr.Dataset(
            {"I": (dims, stacked.real), "Q": (dims, stacked.imag)}, coords=coords
        )


def _dump_raw(raw: xr.Dataset) -> str:
    """Pickle the raw hardware dataset for offline inspection (netCDF can't hold the
    complex S21 data); a failed bring-up run is never wasted."""
    import pickle
    import tempfile
    from datetime import datetime
    from pathlib import Path

    dump = Path(tempfile.gettempdir()) / f"qblox_raw_{datetime.now():%Y%m%d-%H%M%S}.pkl"
    try:
        with open(dump, "wb") as f:
            pickle.dump(raw, f)
        return f"raw dataset pickled to {dump}"
    except Exception as err:
        return f"raw dataset could not be pickled ({type(err).__name__}: {err})"
