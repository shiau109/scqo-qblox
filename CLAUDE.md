# scqo-qblox — Qblox backend for the `scqo` experiment API

## What this repo is
The Qblox sibling of scqo-qm. It implements the **`scqo`** instrument-agnostic
experiment API ([SCQO](https://github.com/shiau109/SCQO), a hard dependency resolved as the
sibling checkout `../SCQO`) against the **Qblox** control stack
(`qblox_scheduler`: `Schedule`, `HardwareAgent`, `QuantumDevice`).

## Four design rules (do not break these)
1. **Independent of Quantum Machines.** Never import `qm`, `quam`, `quam_builder`,
   `qualibrate`, or `qualibration_libs`. The only shared code is `scqo`, which is
   itself vendor-free. (See `pyproject.toml` — no QM packages.)
2. **The common API lives in `scqo`, not here.** Parameters, Result, `estimate`,
   `simulate`, `update`, registry and `Session` come from `scqo`. This repo adds
   only the Qblox-specific halves: `probe()` per experiment and the backend/device adapter.
3. **Runs manually and via AI through the same `scqo.Session`.** `Session.catalog()` /
   `Session.run()` / `Session.device_state()` are plain JSON in/out.
4. **Backend parity — the rule lives in `SCQO\CLAUDE.md` (*Backend parity*).** Given
   one Parameters object, this `probe()` and the QM one must realize the SAME
   sequence: same pulse order, same pulses present, same tones on during
   acquisition. Only vendor idiom may differ (ASAP chaining and `rel_time` here
   against `align()`/`wait()` there; a stitched AWG-offset pair against a rendered
   waveform). A field description saying the other backend "ignores" a parameter is
   the counter-example, not an exemption — that sentence is what let
   `qubit_spectroscopy` latch a continuous drive here while QM played a finite
   pulse, so the same command measured a Stark-shifted line on one instrument and a
   bare one on the other, and both wrote the same `drive_freq_hz`.
   `tests/test_sequential_timing.py` is this repo's half of the pin. An OPTIONAL
   CAPABILITY a backend cannot realize is the exception, and must refuse BY NAME
   (see `experiments/_reset.py`).

## Layout
```
scqo_qblox/
  backend/qblox_backend.py   # QbloxBackend (scqo.Backend) + QbloxDeviceModel + ONE view class
                             #   per CHANNEL KIND: QbloxDriveChannel / QbloxReadoutChannel /
                             #   QbloxFluxChannel (subclass scqo.device.make_view_base("drive"|
                             #   "readout"|"flux")); all three resolve onto the SAME
                             #   qblox_scheduler DeviceElement (the channel's single target)
                             #   wraps qblox_scheduler.HardwareAgent + QuantumDevice
  backend/_distortion.py     # flux-distortion facts -> the 4-stage QCM exp-bank dict (pure)
  backend/apply_distortion.py  # operator CLI: python -m scqo_qblox.backend.apply_distortion
                             #   (QbloxBackend.distortion_apply_command hands scqo's two
                             #   cryoscopes this command line as their writeback hint)
  experiments/
    __init__.py              # imports each experiment module so @register runs (populates catalog;
                             #   completeness enforced by tests/test_experiment_registration.py)
    _vendor.py               # the probes' one door out of the neutral surface: the raw
                             #   DeviceElement behind a target's default channel (ports,
                             #   flux sweet spot), addressed through the ROSTER
    _reset.py, _state.py, _flux_limits.py, _amp_limits.py   # the shared guard/branch helpers
    <name>.py                # ONE module per experiment this backend realizes: Qblox<Name>(<Name>)
                             #   with only probe() — e.g. resonator_spectroscopy, qubit_ramsey.
                             #   NOT every scqo experiment is realized here (several are QM-only);
                             #   __init__.py's __all__ is the authoritative list
qblox_config/                # placeholder (README only): the device-model config is the setup's
                             #   dut_config.json under <data_root>/<device>/<cid>/<setup>/backend_config/
qblox_state/                 # placeholder (README only): the real dut_config.json + hw_config.json live in
                             #   that backend_config/ folder; every scqo run snapshots them from memory
                             #   (QbloxBackend.vendor_config_snapshot -> <device>/setup_snapshots/)
scqo_qblox/scqo_backend.py            # the `scqo.backends` entry-point factory
                                 #   build_backend(cfg, setup, roster): loads the SELECTED
                                 #   named setup's vendor folder (setup["instrument_config"],
                                 #   DERIVED <cid>/<setup>/backend_config since scqo v0.9;
                                 #   canonical names dut_config.json + hw_config.json;
                                 #   loud SystemExit when missing) and threads the device
                                 #   ROSTER into the backend (entity-name resolution needs
                                 #   it); vendor imports stay lazy
scripts/                         # check_real_config.py + ai_loop_demo.py (a worked Session
                                 #   example) + calibrate_mixers.py/.ipynb (see below)
```
Students use the **`scqo` command** and edit **nothing** here: select a setup
(`scqo user --device <name> [--setup <name>]`) and run. With no config everything runs
simulated and saves nothing. Setup/labconfig detail lives in `SCQO\INSTALL.md` §2.

**State authority (`state_sync`)**: sessions on this backend run `"pull"` — the vendor config
(`dut_config.json` + `hw_config.json`) is the truth at startup and scqo pushes only what it
freshly measures. `"push"` is TEMPORARILY refused for every hardware backend by scqo's
`make_session` (a push would seed the vendor from `scqo_state.json` with no history rows and
clobber hand edits such as the LO / attenuation values); this factory never reads
`cfg.state_sync`, so nothing here needs lifting when that changes. `scripts/check_real_config.py`
builds `Session(..., state_sync="push")` directly on a temp COPY of the real config — deliberate,
and outside `make_session`.

## Adding an experiment
1. Subclass the backend-free experiment from `scqo.experiments.<name>`.
2. Implement only `probe()` using `qblox_scheduler` (import the vendor lib *inside* the
   method / backend so `import scqo_qblox` stays light and the simulated path needs no Qblox).
   Read device state through the CHANNEL that owns the knob —
   `self.device.channel(target, "readout").readout_freq_hz`, `...("drive").pi_amp` —
   never `backend.device.component(<qubit>)`; vendor-only bits (ports, the flux sweet
   spot) come from `_vendor.vendor_element(self, target, kind)`.
3. `@register` the subclass and import the module in `scqo_qblox/experiments/__init__.py`
   (manual — `tests/test_experiment_registration.py` refuses a module missing its line —
   and keep `__all__` in step with it; `test_probe_surface.py` compares it to the catalog).
Everything else (parameters, fitting, writeback, simulation) is inherited from `scqo`.

## Reference
- Terminology (Experiment = probe + estimator; "protocol" retired): SCQO's `CLAUDE.md` -> **Terminology**.
- Shared API + patterns: SCQO's `CLAUDE.md` (the sibling checkout, or github.com/shiau109/SCQO).
- Qblox usage examples: `QBLOX_training`, the vendor's read-only example repo (`docs/applications/superconducting`). A LOCAL reference checkout on the lab machine; not needed to build or test this repo.
- QM sibling (do not import from it): [scqo-qm](https://github.com/shiau109/scqo-qm).

## Hardware invariants
- `scqo_qblox/elements.py` vendors the lab's element types and deliberately EXTENDS the
  QBLOX_training copy: `LCHTransmonElement` adds the `spec` submodule (`spec_amp`,
  the saturation-drive slot behind `drive_amp`/`drive_power_dbm`);
  `FluxTunableTransmonElement` subclasses it. `QbloxBackend` must register them
  BEFORE `QuantumDevice.from_json_file`, or the device tree won't deserialize.
  A dut config missing the `spec` block still loads (spec_amp defaults NaN =
  field unknown until seeded).
- The channel views read/write BOTH scheduler API generations (legacy QCoDeS
  callables and the pydantic-model plain attributes).
- `QbloxDeviceModel.component()` takes a ROSTER ENTITY name (`q1_ro`, `q1_xy`,
  `q1_z`) and resolves it through the roster (kind -> view class, single target ->
  vendor element). Everything the vendor does not realize — modes, lines,
  composites, pump/multi-target channels, a target with no element — is a KeyError.
  A view's `.name` is the ENTITY name; `_element.name` is the vendor element.
- The agent's `hardware_configuration` dict is AUTHORITATIVE: every run recompiles from
  it and re-pushes attenuations, so a direct qcodes `.set()` is overwritten.
- `save()` writes BOTH config files (`dut_config.json` + `hw_config.json`); the dut's
  embedded `hardware_config` copy is synced first so they cannot diverge. Both texts come
  from `QbloxDeviceModel.config_texts()`, which `vendor_config_snapshot()` returns WITHOUT
  writing - the per-run setup snapshot (`<device>/setup_snapshots/`, `scqo restore`) is
  therefore exactly what a save would have produced at run start. `_sync_att_limits` may
  legitimately change `output_att` during `acquire()`; scqo reports that as `setup_snapshot.drift`.
- `readout_power_dbm` ↔ readout `output_att` + `measure.pulse_amp`;
  `drive_power_dbm` ↔ drive-port `output_att` + `element.spec.spec_amp`
  (`output_att` takes EVEN integers; both solves keep the amplitude ≤ 0.5, the
  amplitude carries the exact residual).
- **The attenuator's CEILING is per module output, and only the instrument knows
  it.** qblox_instruments builds the `out<k>_att` validator from a live SCPI
  query (`_get_max_out_att`), so it is neither a constant nor a function of the
  module type: chipA runs a 60 dB QRM-RF (slot 8, ISA 2.0) beside a **30 dB**
  QCM-RF (slot 4, ISA 2.1). A hardcoded 0–60 killed a run on 2026-07-29 (`38 is
  invalid: must be between 0 and 30 inclusive`). Since nothing connects until a
  run, `_solve_att` stays optimistic (`_DEFAULT_MAX_OUTPUT_ATT`) and
  `acquire()`'s `_sync_att_limits` corrects before `probe()`: it asks the cluster
  only about chains solved above `_UNIVERSALLY_SAFE_ATT` (20 dB, the QRC's worst
  case — below that no output can refuse, and asking would put a network call in
  the offline tests), then RE-SOLVES an over-solved chain by writing its current
  `*_power_dbm` back. That correction is power-preserving by construction (the
  amplitude takes up what the attenuator cannot) and costs only DAC range, which
  is why clamping is right and refusing would not be. What it learns lands in
  `<config_dir>/att_limits.json`, keyed physically by `slot<N>/out<K>` so
  rewiring a port cannot inherit a stale ceiling, and is loaded at construction
  so the next process solves right the first time.
- `readout_duration_s` ↔ `measure.pulse_duration`, `readout_integration_s` ↔
  `measure.integration_time`: both positive multiples of 4 ns (REFUSED otherwise),
  window ≤ pulse (QM-portability contract — the hardware here would allow more),
  and a pulse shrink clamps the window down with it.
- `readout_rotation_rad` ↔ `measure.acq_rotation` (**radians ↔ DEGREES and
  NEGATED**, both in the view — the vendors are opposite-handed: `acq_rotation`
  turns the data counterclockwise ("threshold line clockwise", vendor tutorial;
  cal16 writes `degrees(mod(-angle(e-g), 2π))` directly), while the neutral field
  keeps QM's `integration_weights_angle` convention, which turns it clockwise —
  and one `scqo set` must mean the same rotation on both backends; the missed
  negation mirrored the frame and classified every chipA shot |g>, 2026-08-08)
  and **FOLDED**: the sequencer takes degrees in [0, 360] and refuses anything
  outside, while the neutral field keeps (−π, π], so both directions wrap; `readout_threshold` ↔ `measure.acq_threshold`,
  unconverted (same normalized frame the probes acquire in) and unfolded (its
  limits are ±1.7e7, which a real threshold never approaches).
  These two arm `use_state_discrimination` on the four coherent-drive probes:
  `experiments/_state.py` asks for `acq_protocol="ThresholdedAcquisition"` and the
  compiler reads the numbers off the element. Vendor default `0.0` = UNCALIBRATED and
  the probes refuse it by name. Calibrated by `single_shot_readout`, whose Qblox
  `update()` proposes both. `readout_rus_threshold` stays Unrealized — it is a
  repeat-until-success LOOP EXIT and active reset here is not a loop, so there is no
  second threshold to write. Qblox absolutely does have feedback; it just isn't that.
- **Discriminated variable naming follows the readout schema** (SCQO TUTORIAL §11 /
  CLAUDE.md digest): a thresholded run with a `shot_idx` sweep decodes to per-shot
  `state` (integer outcomes — the parity monitors; qubit_parity_switch_discrete adds a
  `meas_idx` axis, its two per-cycle measurements riding one `S_21_<q>` channel as
  labeled bins); without one the cluster averaged the thresholded shots, so the decode
  lands on `population` (a probability). `state` never means an averaged value.
- **Active reset** (`reset_method="active"`) lives in `experiments/_reset.py`, the ONE
  door every probe builds its reset through (`add_reset`) — enforced by
  `test_no_probe_constructs_the_reset_gate_directly`. Vendor `ConditionalReset` =
  thresholded `Measure` + `ConditionalOperation(X)` at a fixed `TRIGGER_DELAY` of
  364 ns; ~18.8 µs against chipA's 1.86 ms thermal wait. Four rules, each a SILENT
  failure if broken:
  1. Opt-in is per probe (`supports_active_reset`, default DENY) and limited to the
     four coherent-drive carriers. Everything else refuses BY NAME — the readout-sweep
     probes because the discriminator is only valid at the calibrated point,
     `single_shot_readout` because it IS that calibration, `qubit_spectroscopy` because
     its `Reset` is a driven dwell. `QbloxBackend.acquire` re-checks before `probe()`.
  2. The conditional measurement takes `acq_channel=f"cond_{q}"`. On the probe's own
     `S_21_<q>` the schedule still compiles and the bins MERGE (5-point sweep → 10 bins,
     dying later in `_to_canonical`).
  2b. The settle after the reset is the readout channel's `readout_depletion_s` KNOB
     (`element.depletion.duration`, a lab addition in `elements.py`), resolved through
     scqo's `_depletion.depletion_wait_ns` — never a number this driver picks.
     `resonator_spectroscopy` calibrates it as `depletion_factor / (2π·kappa_tot_hz)`.
     **NaN (never calibrated) refuses; 0 runs** — "no settle needed" and "nobody
     measured this resonator" are different claims, which is why the setter bypasses
     `snap_ns` (it rejects non-positive) for the 0 case.
  3. The discriminator guard does NOT ride on `use_state_discrimination` — active reset
     with averaged I/Q readout is legal, and an uncalibrated `ConditionalReset` compiles
     clean and thresholds every shot against zero.
  4. Targets stay SEQUENTIAL (one outstanding feedback label). The compiler catches only
     the *interleaved* shape; two whole `ConditionalReset`s overlapping compile clean and
     are a hardware hazard. `thermalization_time_ns` + `active` is refused, not ignored.
  COMPILE TIME grows with `num_averages` under active reset only —
  `compile_conditional_playback` unrolls every loop in Python. Measured (51 points, 1
  target): thermal flat ~0.3 s; active 0.94 / 3.79 / 14.11 s at 100 / 1000 / 4000. It is
  spent inside `acquire()` before anything reaches the cluster, so a big run looks hung.
- Readout/drive LO = `hw_config.json` `hardware_options.modulation_frequencies`
  (PORT-level, shared by every element on the output; untracked wiring). Hand-edit
  only while NO session is live — `save()` rewrites the file from the in-memory
  config and would silently revert the edit — and restart notebook kernels after.
  `power_context` stamps the readout LO into every run record, and the setup snapshot
  now carries the whole `hw_config.json` per run.
- **Mixer calibration** (`scripts/calibrate_mixers.py`, notebook wrapper alongside) is an
  OPERATIONS tool, not part of the scqo surface: it drives the RF modules' built-in AMC
  straight through `qblox_instruments`, so no session/HardwareAgent may hold the cluster
  while it runs. It is listed — with that precondition, which the script itself only prints
  at RUNTIME — in `fieldmap.OPERATOR_COMMANDS`, rendered by `scqo state --fields`: an
  operator cannot find a non-scqo command through `scqo -h`, so that inventory is the door. It reads the config folder — connectivity graph for port -> (slot, output),
  `modulation_frequencies` for the LO, `dut_config` clocks for `NCO = clock_freq - lo_freq`.
  **`sideband_cal()` fails silently and non-deterministically.** It returns having changed
  nothing while the firmware reports success, on roughly half to three-quarters of calls
  (chipA 2026-07-27). You cannot lean on the LO cal to notice: LO leakage is DC mixer
  feedthrough, so `out{k}_lo_cal()` yields plausible offsets *with no tone playing at all*.
  When the sideband cal does land the value is solid, so the TRIGGERING is unreliable, not
  the measurement. Hence the design — **verify, then retry**, never "trust a condition":
  a result still on the vendor defaults is a `no-op` that is never cached and exits
  non-zero; a CACHED entry on the defaults is treated as a miss, so one bad run cannot
  poison the file; the check is TOLERANCE-based, not `==` (a failed cal can null out to
  1.000031); and `--attempts` (default 12) retries until the values move. That knob is the
  reliability lever — 5 was not enough. Attempt counts go to `mixer_cal.json` `history`;
  a rising trend is the number for a Qblox support ticket.
  Do NOT infer a cause from a single pass/fail. Attenuation, `clear_sequencer_flags()`, cal
  ordering, the `mixer_corr_*` write and the QRM-RF acquisition connection were each
  "confirmed" from one A/B and each later contradicted. They remain in the code because
  they are harmless and match the Qblox tutorial, but they are UNPROVEN. `--diagnose`
  re-runs the five-trial matrix on one port-clock when a chip misbehaves.
  **The process must not linger** (this one IS established): `close_cluster()` bounds
  `Cluster.close()` with a watchdog and `_hard_exit()` ends it with `os._exit`, because a
  "finished" run that never exits keeps four sockets on the cluster, and that contention
  degrades every later run — eleven such zombies once produced garbled SCPI reads
  (`int('')` at connect).
  LO cal is per OUTPUT (`out{k}_offset_path0/1`, module state); sideband cal is per
  SEQUENCER (`mixer_corr_gain_ratio` / `mixer_corr_phase_offset_degree`) and must run on the
  index the compiler will allocate — lowest free sequencer per module, port-clocks in config
  order (seq0 for one port-clock per module; override on a multiplexed module). Results live
  only in VOLATILE cluster state, so `<config_dir>/mixer_cal.json` caches them keyed by
  `(slot, output, lo_freq)` / `(slot, sequencer, nco_freq)` and validates against the LIVE
  values — a reboot reads as a miss, never as a skip.
  **It survives a run only because hw_config has no `hardware_options.mixer_corrections`
  block**: the instrument coordinator pushes `mixer_corr_*` / `out{k}_offset_path*` only for
  port-clocks that declare them (defaults are `None`), and never resets the module. Adding
  such a block overwrites the AMC on the next upload — then use the scheduler's own
  `auto_lo_cal` / `auto_sideband_cal` instead of this script.
- **The Windows shutdown traceback belongs to us, not to scqo.** Every `scqo run` used to end
  with `OSError: [WinError 87]` / "Error on reading from the event loop self pipe" AFTER
  `saved:` — qblox's `Transport.__init__` makes a bare `ProactorEventLoop` per transport when
  no loop is running and never closes it, and asyncio reports the dead self-pipe through
  `call_exception_handler`. `backend/_asyncio_noise.py` installs a narrow handler (that one
  message + `OSError` + winerror 87) on each cluster's transport loop; `acquire()` calls it in
  a `finally`, since the loops exist only once `HardwareAgent.run` has connected and a run
  that RAISES has opened them too. Slot transports share the cluster's loop
  (`loop_from=self._transport`), so it is one loop per Cluster. The fix lives HERE, not in
  SCQO's CLI: that CLI is vendor-neutral, and an `os._exit` there would skip dataset flushes
  for every backend. Closing the loop at `atexit` was rejected too — `atexit` is LIFO and
  qcodes registers its own instrument-closing hooks.
- **`release_instruments()` hands the cluster back before an interactive prompt** — the
  `scqo.Backend` hook, implemented HERE and nowhere else (QM closes its QuantumMachine per
  acquisition; this backend has no disconnect at all, so one `acquire()` pins four sockets
  per cluster until the process exits). SCQO's `_review_core` calls it before blocking on
  "apply which updates?", which is safe because deciding a suggestion writes only the
  in-memory config and the JSON files — the values reach hardware at the next run, which
  reconnects. Two traps it is written against: `get_clusters()` CONNECTS when `_clusters` is
  empty (so asking to disconnect would connect — read `_clusters` directly), and
  `Cluster.close()` can block forever (so each close runs on a daemon thread joined with
  `_CLUSTER_CLOSE_TIMEOUT_S`; past it we warn and keep the socket, never hang). Deliberately
  NOT `calibrate_mixers.py`'s `_hard_exit` answer: killing the process is legitimate at the
  end of a standalone tool, never inside a session that still owes the operator a decision.
  `tests/test_release_instruments.py` pins all of it with fakes — no cluster needed.
- **Flux is in VOLTS on the neutral surface and a FRACTION on the wire.** Every Qblox sequencer
  operand is a fraction of full scale — `VoltageOffset`'s operand is documented "the unitless
  amplitude", `offset_awg_path{x}` is `Numbers(-1.0, 1.0)` with `unit=""`, and the scheduler's
  `max_awg_output_voltage` is DEAD metadata (declared five times, read zero times). Nothing in the
  stack converts. Until 2026-07-30 the flux probes passed `flux_bias_v` straight through, so a
  requested 0.3 V left a QCM as **0.75 V** and every fitted `flux_offset` was wrong by the module's
  full-scale factor. `experiments/_flux_limits.py::to_dac_fraction` is the conversion, kept
  separate from the checks on purpose — a function that validates *and* silently rescales is two
  jobs, and the rescale must be visible at the call site.
- **The flux rail is fixed by the MODULE** (no direct/amplified analogue): baseband **QCM 5 Vpp =
  ±2.5 V peak into 50 Ω**, QRM ±0.5 V. The spec is PEAK-TO-PEAK and the guard compares a
  single-ended voltage, so it takes half — reading 5 there is a silent factor of two. RF modules
  cannot emit DC at all, so flux on one is refused as a WIRING error (the vendor config validator
  catches it first in practice; the branch remains for unvalidated dicts and unknown module types).
  **Nothing downstream refuses** — measured, not assumed: a ±0.9 domain (±2.25 V) compiles clean
  and ±3.0 dies with an internal numpy `ufunc 'absolute' ... StrDType` naming no port. The guard
  must run BEFORE compilation, which is why `tests/conftest.compile_probe` no longer claims the
  compiler enforces the DAC range.
- **Two flux frames, and the `_pulse` suffix is a promise.** `resonator_spectroscopy_flux` is
  ABSOLUTE (the swept value IS the line voltage; `VoltageOffset` replaces the standing bias);
  `qubit_spectroscopy_flux_pulse` is RELATIVE (the swept value is an excursion the DAC adds to
  `idle_flux`). Until 2026-07-30 both emitted absolutely, so the `_pulse` probe — which scqo
  test-enforces as relative and whose estimator re-references `old_idle_flux + fitted` — was
  silently absolute. It was invisible only because the old vendor helper defaulted an uncalibrated
  sweet spot to **0.0**, exactly where the two frames coincide. The relative probe now shifts its
  loop DOMAIN by the anchor: writing `idle_flux + flux` at the use site instead reaches the
  compiler as a BinaryExpression and dies in `expand_awg_from_normalised_range`.
- **`idle_flux` is a REALIZED knob** (`element.flux_params.sweet_spot`), not Unrealized. It reads
  through `flux_anchor_v`, the same call `estimate()` uses to record `old_idle_flux`, so the bias a
  probe emits from and the one the fit re-references cannot drift apart. NaN means uncalibrated and
  REFUSES; only the absolute probe's end-of-schedule park falls back to 0 V, because that park is a
  courtesy and not part of any measurement.
- Placement rule (which store owns which value): `scqo state --rule` / SCQO TUTORIAL §10.
  A vendor copy of a neutral/physical value is legal only as a CACHE with a named
  refresh trigger — the SCQO stores are truth.
- Two readout-power probes: `resonator_spectroscopy_power_chain` sweeps power with a
  Python loop (one 1D detuning scan per point); `resonator_spectroscopy_power_amp` is a
  single-program FPGA sweep over Python-UNROLLED geometric amplitude blocks, giving a
  uniform-dBm axis.
- **The cryoscope probes (2026-08-24).** Three compiler facts shape them, each measured:
  every same-sequencer gap compiles to a `wait` that must be 0 or ≥ 4 ns; `VoltageOffset`
  and `play` each consume an intrinsic 4 ns before that wait; and `Schedule.loop` UNROLLS
  every TIME domain in Python anyway (schedule.py:1103), so a Python loop over durations
  is compile-identical to a "hardware" one. Hence `qubit_ramsey_cryoscope` COMPOSES each
  1-ns duration as `n = 4k + r`: a VoltageOffset segment of 4k ns plus a 1/2/3-sample
  SquarePulse remainder at the same timestamp the offset returns (three waveforms total;
  every flux wait a multiple of 4 by construction), with the frame axis a realtime PHASE
  loop driving `ShiftClockPhase` (+360·frame — tomography, NOT qubit_ramsey's negative
  ramp) before a fixed X90. A variable `Rxy(phi=...)` does NOT compile in a realtime loop
  (`pulses.py:264` whitelists only voltage_offset/phase_shift/frequency). Ceiling:
  `max_duration_ns` ≤ 512 (the READOUT QRM's 12288-instruction budget fills first;
  640 exceeds it) — refused by name. `qubit_spectroscopy_cryoscope` is square-drive-only
  (refuses cosine/gaussian by name; π-area amplitude from sampling the rxy DRAG envelope,
  σ = duration/8) with the log wait axis Python-unrolled inside a realtime FREQUENCY loop.
- **Distortion apply** (`python -m scqo_qblox.backend.apply_distortion --target q1
  [--run <id>|--extend|--clear|--dry-run]`): accepted cryoscope facts → ONE
  `QbloxHardwareDistortionCorrection` under
  `hardware_options.distortion_corrections["<flux_port>-cl0.baseband"]` (flux plays on
  the baseband IDENTITY clock — there is no `<q>.flux` clock; a real output refuses a
  LIST, that shape is complex-channel). The 4-stage bank is hard hardware: overflow taps
  warn LOUDLY (kept = most significant by |A|), and `--extend` merges + re-partitions
  rather than appending. Saved via `QbloxDeviceModel.save()` (both config files);
  compiled + pushed on the next run that plays flux — only QCM compilers apply it. Reached
  two ways: the cryoscopes print it at writeback (`distortion_apply_command`, run-addressed
  and fully resolved), and `scqo state --fields` lists it as a `<placeholder>` template —
  two artifacts on purpose, pinned to the same module by `test_scqo_glue.py`.
- **Probes that acquire inside `probe()`** (`probe_self_acquires = "<why>"`, spelled by
  `backend.SELF_ACQUIRING_ATTR`, same default-ALLOW polarity as scqo-qm's): the two
  broadband sweeps, `qubit_drag_equator` and `qubit_tomography`. They step an LO or the
  DRAG coefficient BETWEEN acquisitions, so they compile and run their own schedules and
  hand `acquire()` a finished Dataset; `preview` refuses them by name. Three rules:
  1. **Each exposes its schedule builders** (`build_sub_schedule`, `build_schedule`,
     `build_training_schedule`/`build_tomography_schedule`) and is listed in
     `tests/test_probe_surface.SELF_ACQUIRING_BUILDS`, which COMPILES every one.
     `test_every_self_acquiring_probe_declares_its_schedules` fails a probe that is
     absent from the map — the early-return that skipped them silently left ~400 lines
     of probe uncompiled, in the one file whose whole point is that compiling is what
     catches the instrument's rules.
  2. **Deadlines come from `qblox_backend.chunk_timeout_s`**, never a hand-rolled
     formula: it applies `_run_timeout_s`'s arithmetic to the ONE piece about to run and
     counts the REAL thermalization. A magic 500 us shot period ignores chipA's 2.7 ms
     reset — the underestimate issue #24 reported.
  3. **Whatever they step, they restore in a `finally`** (LO, drive/readout clock, DRAG
     beta), and a FAILED step raises: swallowing it measures a sub-band at the previous
     LO and labels it with this one's frequency.
  `broadband_qubit_spectroscopy` measures ONE target — it steps a single drive
  port-clock's LO — and refuses a second by name; `broadband_resonator_spectroscopy`
  legitimately reports the same feedline trace for every target, exactly as the neutral
  `simulate()` broadcasts it.
- **`drag_beta` is REALIZED** (`element.rxy.beta`) now that `qubit_drag_equator` lands
  here: the view speaks **ns**, the vendor stores **seconds**. It is deliberately NOT the
  same quantity as QM's `DragCosinePulse.alpha` — the catalog marks `drag_beta`
  `portable=False` precisely so each backend may define it in its own convention.
  `rxy` carries ONE beta and X90 is DERIVED from `rxy.amp180` (`amp180*theta/180`), so
  **`pi_amp_x90` and `drag_beta_x90` stay Unrealized**: binding them to the x180 slots
  (`amp180 = 2*pi_amp_x90`, the shared beta) would let a pi/2 calibration silently
  overwrite the calibrated pi gate — the failure issue #24 reported on the QM side.
  `qubit_deterministic_benchmarking` therefore reads `amp_reference_field()` and refuses
  `target_gate=x90` in `define_sweep()`, before the neutral layer reaches for the
  unrealized anchor and reports a bare "has no value yet". Promotion means moving
  `pi_half` from `FluxTunableTransmonElement` up to `LCHTransmonElement`, binding
  `element.pi_half.amp90`, and teaching the probes to PLAY it.
- **Sequence preview** (`scqo run <name> --preview` → `QbloxBackend.preview`): compiles
  the probe's Schedule OFFLINE (the same `hardware_config` mutation + SerialCompiler call
  `HardwareAgent.run` makes) and writes `pulse_diagram.html` (plotly) +
  `timing_table.html` — both MANDATORY, both need the compiled timeline. It must never
  call `_sync_att_limits` (network + att_limits.json write; test-pinned). **The render
  is capped at `_PREVIEW_MAX_RENDERED_SHOTS` (256 = sweep points × repetitions), refused
  by name with the exact `--set` remedy**: the vendor visualizer UNROLLS every loop by
  deepcopy before sampling (profiled 2026-08-09 — 51 pts × 400 avg = 9.4 min in
  sample_schedule, 84% deepcopy; `x_range` does NOT bound it), so lab-default counts are
  hours. Preview small (`--set num_averages=2` + the experiment's point-count
  parameter, which is per-capability: `num_points` on the time sweeps,
  `num_readout_freq_points` on the readout-detuning ones) — the per-shot
  sequence is identical. No gate-level circuit diagram on purpose: the vendor renderer
  cannot draw hardware-loop swept schedules (dies uncompiled AND compiled, measured
  2026-08-09), and every registered probe sweeps.

## Tests
`tests/conftest.py` holds the shared fixture chip — a schema-3 roster for the demo
dut config (`ROSTER_TOML`: q1/q2 + coupler c12, one multiplexed feedline, a drive and
a flux wire each) plus `make_backend` / `make_experiment` (the latter attaches the
`RecordingDevice` a Session would). `tests/test_scqo_glue.py` (scqo↔backend glue, the
per-kind fieldmap drift alarm, the `components()` witness), `tests/test_qblox_power.py`
(the readout AND drive absolute-power paths), `tests/test_probe_surface.py` (EVERY
registered probe COMPILES its Schedule against the channel-entity surface — building
one proves nothing, since the time grid, the DAC range and the latched-parameter
alignment all live in the compiler; `conftest.compile_probe` is the shared door).

**One vendor version, both venvs.** `uv run pytest` uses `.venv`; `scqo run` on the
cluster uses the shared `.venv-qblox`. They must hold the same `qblox-scheduler` - on
2026-07-26 they did not (b4 vs b6) and the two versions *disagreed about whether a
schedule is legal*: `readout_frequency` compiled clean offline and died on hardware.
Both are now 1.0.0b6 + qblox_instruments 1.3.0. After changing either, run the suite
in the lab venv too, with its interpreter directly:
`<.venv-qblox>/Scripts/python.exe -m pytest tests/ -q`.

### Testing discipline — here, just run the whole thing
`uv run pytest tests/ -q` (plain `uv run` is correct: `scqo` is a hard dependency in
`pyproject.toml`, so uv's sync keeps it). This suite is small enough that a selection map would
cost more attention than it saves — unlike SCQO's, which is minutes — so the full run IS the
targeted run. Run it before every commit. No test counts or timings are quoted here on purpose:
the previous ones rotted into three mutually contradictory figures for this repo alone. Each
release records what it actually ran, in the `OFFLINE-VALIDATED` line of SCQO's `RELEASES.toml`.

The one narrowing worth knowing: **`test_scqo_glue.py` is the slowest file by a wide margin** — it shells out to
the real `scqo` CLI and runs the AI-loop demo end-to-end. While iterating on a probe, loop on
`uv run pytest tests/test_probe_surface.py tests/test_time_grid.py -q` — per-test time is
milliseconds there, so almost all of it is fixture + `qblox_scheduler` import — and pick the glue
test back up before you commit. Once you are down to the import cost there is nothing left to win;
don't over-narrow.

| File | Covers |
|---|---|
| `test_probe_surface.py` | every registered probe **compiles** its Schedule on the channel-entity surface |
| `test_time_grid.py` | the specific swept WINDOWS whose naive linspace step was fractional |
| `test_state_discrimination.py` | `use_state_discrimination`: the two knobs, the thresholded probes, the `state` decode, the single_shot_readout proposal |
| `test_active_reset.py` | `reset_method="active"`: the opt-in census, the four refusals, the acq-channel rule, rounds/settle, the sequential-feedback rule (needs `fixtures/hw_config_2q.json`) |
| `test_qblox_power.py` | output-att solves, the hardware-config write surface, dual-file save, `power_context` |
| `test_vendor_snapshot.py` | `vendor_config_snapshot`: deterministic, writes nothing, the EXECUTED config (not the disk copy), no nulls, parsed-equal to `save()`, degrades without an agent |
| `test_att_limits.py` | the per-module attenuator ceiling: clamp-not-refuse, when the cluster is asked and when it must NOT be, the power-preserving re-solve, the sidecar |
| `test_sequential_timing.py` | the BACKEND-PARITY half: `qubit_spectroscopy`'s drive ends at the readout tone's START (`readout_overlap=false`) or its END (`true`), at all three emission shapes — asserted on the COMPILED tree with accumulated absolute times |
| `test_qblox_reset.py` | `thermalization_time_s` as a neutral drive-channel knob |
| `test_flux_limits.py` | the flux rail per MODULE, the volts→DAC-fraction conversion, the two frames, the RF-wiring refusal |
| `test_readout_duration.py` | duration/window knobs on the readout view (pure stubs, no qblox_scheduler) |
| `test_hw_config_serialization.py` | explicit nulls never written or trusted |
| `test_mixer_calibration.py` | `scripts/calibrate_mixers.py`: the pure plan (config -> LO/NCO groups) + the AMC control flow on a `dummy_cfg` cluster (channel map, tone, sequencer snapshot/restart, cache) |
| `test_asyncio_noise.py` | the WinError-87 shutdown suppressor: what it swallows, what it must NOT, idempotence, and that `acquire()` installs it even when the run raises |
| `test_preview.py` | `QbloxBackend.preview`: both compiled artifacts render offline, no `_sync_att_limits` call, pinned `--out` dir overwrites in place, the rendered-shot cap refuses lab-sized schedules by name |
| `test_new_probe_contracts.py` | the two silent-wrong-data regressions the 2026-08 probe batch shipped with: benchmarking sweeps `amp_reference_field()`'s knob and refuses a pi/2 gate BY NAME, the x90 knobs stay Unrealized, broadband-qubit refuses a second target (its resonator sibling's broadcast is pinned as CORRECT), `chunk_timeout_s` counts the reset |
| `test_experiment_registration.py` | every experiment module has its `__init__` import line (both directions) |
| `test_scqo_glue.py` | the `scqo` CLI works in THIS venv + the qblox factory (slow — see above) |
